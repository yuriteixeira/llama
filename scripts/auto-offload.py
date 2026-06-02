#!/usr/bin/env python3
"""Auto-calculate, smoke-test, and self-repair buun offload params."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from bench_buun_tui.config_store import load_config, save_config
from bench_buun_tui.offload import OffloadError, calculate_offload
from bench_buun_tui.profile import Profile
from bench_buun_tui.repair import diagnose_failure, next_repair_step
from bench_buun_tui.smoke import run_smoke_test, smoke_test_context


def main() -> int:
    args = parse_args()
    stop_llama_servers()
    cfg = load_config(args.config) if args.config else None
    profile = selected_profile(args, cfg)
    desired_context = args.ctx or smoke_test_context(profile)

    print(f"model: {profile.model}")
    print(f"context: {desired_context}")
    if args.vram_budget_mib:
        print(f"calculation VRAM budget override: {args.vram_budget_mib} MiB")
    warn_gpu_processes()
    try:
        rec = calculate_offload(profile, desired_context=desired_context, vram_budget_bytes=args.vram_budget_mib * 1024 * 1024 if args.vram_budget_mib else None)
    except OffloadError as exc:
        print(f"auto-offload failed: {exc}", file=sys.stderr)
        return 2

    profile.gpu_layers = rec.gpu_layers
    profile.n_cpu_moe = rec.n_cpu_moe
    print(f"initial recommendation: gpu_layers={profile.gpu_layers} n_cpu_moe={profile.n_cpu_moe} confidence={rec.confidence}")

    try:
        passed = try_and_repair(profile, desired_context, rec.model.block_count, rec.model.supports_moe, args.max_attempts, args.timeout)
    finally:
        stop_llama_servers()
    if not passed:
        print("no passing smoke-test candidate found", file=sys.stderr)
        return 1

    print(f"passing values: gpu_layers={profile.gpu_layers} n_cpu_moe={profile.n_cpu_moe}")
    if args.apply_config:
        if cfg is None or args.config is None:
            print("--apply-config requires --config", file=sys.stderr)
            return 2
        save_config(cfg, args.config)
        print(f"updated config: {args.config}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="GGUF model path, relative to repo root or absolute")
    source.add_argument("--config", type=Path, help="saved TUI INI config to load")
    parser.add_argument("--profile", default="profile_a", help="profile section when using --config (default: profile_a)")
    parser.add_argument("--ctx", type=int, help="desired max context for calculation and smoke test")
    parser.add_argument("--batch-size", default="1024")
    parser.add_argument("--ubatch-size", default="256")
    parser.add_argument("--threads", default="12")
    parser.add_argument("--cache-strategy", default="turbo", choices=["turbo", "q8", "f16", "unset"])
    parser.add_argument("--device", default="CUDA0")
    parser.add_argument("--flash-attn", default="1")
    parser.add_argument("--depths", default="32768")
    parser.add_argument("--max-attempts", type=int, default=40, help="maximum smoke-test attempts during repair")
    parser.add_argument("--timeout", type=int, help="seconds before killing each smoke-test attempt")
    parser.add_argument("--vram-budget-mib", type=int, help="override usable VRAM budget for calculation; smoke test still needs that VRAM actually free")
    parser.add_argument("--apply-config", action="store_true", help="write passing values back to --config")
    return parser.parse_args()


def selected_profile(args: argparse.Namespace, cfg) -> Profile:
    if cfg is not None:
        for profile in cfg.profiles:
            if profile.name == args.profile:
                return profile
        raise SystemExit(f"profile not found in config: {args.profile}")
    return Profile(
        args.profile,
        model=args.model,
        cache_strategy=args.cache_strategy,
        depths=args.depths,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        threads=args.threads,
        device=args.device,
        flash_attn=args.flash_attn,
    )


def stop_llama_servers() -> None:
    candidates = [
        ["./manage-server.sh", "buun", "stop"],
        ["./manage-server.sh", "vanilla", "stop"],
        ["systemctl", "--user", "stop", "llama-server-buun.service"],
        ["systemctl", "--user", "stop", "llama-server.service"],
    ]
    stopped_any = False
    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15, check=False)
        except Exception:
            continue
        if proc.returncode == 0:
            stopped_any = True
    if stopped_any:
        print("stopped llama-server before testing")


def warn_gpu_processes() -> None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except Exception:
        return
    if proc.stdout.strip():
        print("warning: active GPU compute processes may reduce available VRAM for smoke tests:")
        for line in proc.stdout.strip().splitlines():
            print(f"  {line}")


def try_and_repair(profile: Profile, ctx: int, block_count: int, supports_moe: bool | None, max_attempts: int, timeout: int | None) -> bool:
    attempts = 0
    tried: set[tuple[str, str]] = set()
    diagnosis = "unknown"

    def attempt(label: str) -> tuple[bool, str]:
        nonlocal attempts
        key = (profile.gpu_layers, profile.n_cpu_moe)
        if key in tried or attempts >= max_attempts:
            return False, diagnosis
        tried.add(key)
        attempts += 1
        print(f"\nattempt {attempts}/{max_attempts}: {label}: gpu_layers={profile.gpu_layers} n_cpu_moe={profile.n_cpu_moe}")
        result = run_smoke_test(profile, ctx, timeout=timeout)
        if result.output.strip():
            print(tail_text(result.output, 3000))
        diagnosis = diagnose_failure(result.output)
        print(f"returncode={result.returncode} diagnosis={diagnosis}")
        return result.passed, diagnosis

    passed, diagnosis = attempt("recommended")
    if passed:
        return True

    while attempts < max_attempts:
        step = next_repair_step(profile, block_count, supports_moe, diagnosis, tried)
        if step is None:
            return False
        profile.gpu_layers = step.gpu_layers
        profile.n_cpu_moe = step.n_cpu_moe
        passed, diagnosis = attempt(step.label)
        if passed:
            return True
    return False


def tail_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else "...\n" + text[-max_chars:]


def int_or_default(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())

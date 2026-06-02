from __future__ import annotations

from pathlib import Path

from .paths import BENCH, ROOT, SERVER
from .profile import CACHE_STRATEGIES, Profile


def is_set(value: str) -> bool:
    return bool(value.strip()) and value.strip().lower() not in {"unset", "none", "default"}


def add_arg(cmd: list[str], flag: str, value: str) -> None:
    if is_set(value):
        cmd.extend([flag, value])


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def flash_attn_server_value(value: str) -> str:
    return "on" if truthy(value) else "off"


def profile_server_command(profile: Profile) -> list[str]:
    ctk, ctv = CACHE_STRATEGIES.get(profile.cache_strategy, CACHE_STRATEGIES["turbo"])
    model_path = Path(profile.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    cmd = [
        str(SERVER),
        "--host", "0.0.0.0",
        "--port", "8080",
        "--model", str(model_path),
        "--alias", Path(profile.model).stem or "selected",
        "--ctx-size", best_server_ctx_size(profile),
    ]
    if is_set(ctk):
        cmd.extend(["--cache-type-k", ctk])
    if is_set(ctv):
        cmd.extend(["--cache-type-v", ctv])
    add_arg(cmd, "--threads", first_csv_value(profile.threads))
    add_arg(cmd, "--threads-batch", first_csv_value(profile.threads))
    cmd.extend(["--threads-http", "4"])
    add_arg(cmd, "--batch-size", first_csv_value(profile.batch_size))
    add_arg(cmd, "--ubatch-size", first_csv_value(profile.ubatch_size))
    add_arg(cmd, "--n-cpu-moe", first_csv_value(profile.n_cpu_moe))
    add_arg(cmd, "--n-gpu-layers", first_csv_value(profile.gpu_layers))
    add_arg(cmd, "--device", profile.device)
    if is_set(profile.flash_attn):
        cmd.extend(["--flash-attn", flash_attn_server_value(profile.flash_attn)])
    if is_set(profile.no_mmap) and truthy(profile.no_mmap):
        cmd.append("--no-mmap")
    cmd.extend(["--parallel", "1", "--metrics", "--reasoning", "off", "--no-mmproj"])
    return cmd


def first_csv_value(value: str) -> str:
    if not is_set(value):
        return value
    return value.split(",", 1)[0].strip()


def best_server_ctx_size(profile: Profile) -> str:
    if not is_set(profile.depths):
        return "262144"
    vals = [int(p) for p in profile.depths.split(",") if p.strip().isdigit()]
    return str(max(vals) if vals else 262144)


def profile_command(profile: Profile, raw_path: Path) -> list[str]:
    ctk, ctv = CACHE_STRATEGIES.get(profile.cache_strategy, CACHE_STRATEGIES["turbo"])
    model_path = Path(profile.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    cmd = [
        str(BENCH),
        "--model", str(model_path),
        "--output", "jsonl",
        "--progress",
    ]
    add_arg(cmd, "--n-depth", profile.depths)
    add_arg(cmd, "--n-prompt", profile.prompt)
    add_arg(cmd, "--n-gen", profile.gen)
    add_arg(cmd, "--batch-size", profile.batch_size)
    add_arg(cmd, "--ubatch-size", profile.ubatch_size)
    add_arg(cmd, "--threads", profile.threads)
    add_arg(cmd, "--n-cpu-moe", profile.n_cpu_moe)
    add_arg(cmd, "--n-gpu-layers", profile.gpu_layers)
    if is_set(ctk):
        cmd.extend(["--cache-type-k", ctk])
    if is_set(ctv):
        cmd.extend(["--cache-type-v", ctv])
    add_arg(cmd, "--device", profile.device)
    add_arg(cmd, "--flash-attn", profile.flash_attn)
    if is_set(profile.no_mmap) and truthy(profile.no_mmap):
        cmd.extend(["--mmap", "0"])
    add_arg(cmd, "--repetitions", profile.repetitions)
    if profile.warmup.lower() not in {"1", "true", "yes", "on", "unset", "none", "default"}:
        cmd.append("--no-warmup")
    return cmd

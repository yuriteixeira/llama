#!/usr/bin/env python3
"""Update buun llama-server model presets from the best recorded benchmark results."""

from __future__ import annotations

import argparse
import importlib.util
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "bench-results" / "buun-tui"
PRESET_PATH = ROOT / "versions" / "buun" / "model-presets.ini"
FIXED_ARGS_PATH = ROOT / "versions" / "buun" / "model-preset-fixed-args.txt"
RANK_SCRIPT = ROOT / "scripts" / "rank-buun-results.py"

sys.path.insert(0, str(ROOT / "scripts"))
from bench_buun_tui.models import find_models, model_label, model_path  # noqa: E402


@dataclass(frozen=True)
class PresetRow:
    section: str
    model: str
    ctx_size: int
    cache_type_k: str
    cache_type_v: str
    threads: int | None
    n_batch: int | None
    n_ubatch: int | None
    n_cpu_moe: int | None
    n_gpu_layers: int | None
    flash_attn: str
    score: float | None
    pp_tps: float | None
    tg_tps: float | None
    sample_count: int | None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULT_DIR, help="benchmark results directory")
    parser.add_argument("--preset", type=Path, default=PRESET_PATH, help="model-presets.ini path to update")
    parser.add_argument("--min-samples", type=int, default=1, help="minimum paired samples required per model")
    parser.add_argument("--score", choices=["normalized-geomean", "simple-geomean"], default="normalized-geomean")
    parser.add_argument("--include-test-runs", action="store_true", help="include run directories whose name contains 'test'")
    parser.add_argument("--latest-only", action="store_true", help="only consider the latest run directory")
    parser.add_argument("--max-models", type=int, default=0, help="limit number of ranked model presets written; 0 means all")
    parser.add_argument("--fixed-args-file", type=Path, default=FIXED_ARGS_PATH, help="file with llama-server args to include in every preset")
    parser.add_argument("--fixed-arg", action="append", default=[], help="additional llama-server arg snippet to include in every preset; may be repeated")
    parser.add_argument("--dry-run", action="store_true", help="print the generated preset without writing it")
    args = parser.parse_args()

    ranking = load_rank_module()
    summaries = ranking.discover_summary_files(args.input.resolve(), args.include_test_runs, args.latest_only)
    rows = ranking.load_rows(summaries)
    samples = ranking.build_paired_samples(rows)
    if not samples:
        print(f"No paired benchmark samples found under {args.input}", file=sys.stderr)
        return 1

    # Mutates samples with normalized scores so overall ranking uses the same score semantics as rank-buun-results.py.
    ranking.rank_by_depth(samples, score_mode=args.score)
    aggregates = ranking.rank_best_model_configs(samples, min_samples=max(1, args.min_samples), score_mode=args.score)
    if args.max_models > 0:
        aggregates = aggregates[: args.max_models]

    fixed_params = load_fixed_params(args.fixed_args_file, args.fixed_arg)
    preset_rows: list[PresetRow] = []
    used_sections: set[str] = {"default"}
    warnings: list[str] = []
    for agg in aggregates:
        resolved_model = resolve_model(str(agg.model_path or ""), str(agg.model or ""), warnings)
        if not resolved_model:
            continue
        section = unique_section_name(slugify(Path(resolved_model).stem), used_sections)
        used_sections.add(section)
        preset_rows.append(
            PresetRow(
                section=section,
                model=str(model_path(resolved_model).resolve()),
                ctx_size=int(ranking.max_depth_for_config(samples, agg.config_key) or 262144),
                cache_type_k=str(agg.type_k or ""),
                cache_type_v=str(agg.type_v or ""),
                threads=agg.n_threads,
                n_batch=agg.n_batch,
                n_ubatch=agg.n_ubatch,
                n_cpu_moe=agg.n_cpu_moe,
                n_gpu_layers=agg.n_gpu_layers,
                flash_attn=ranking.normalize_bool_text(agg.flash_attn or "on") or "on",
                score=median(agg.scores) if agg.scores else None,
                pp_tps=median(agg.pp_values) if agg.pp_values else None,
                tg_tps=median(agg.tg_values) if agg.tg_values else None,
                sample_count=agg.sample_count,
            )
        )

    if not preset_rows:
        print("No ranked rows could be resolved to local model files.", file=sys.stderr)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        return 1

    content = render_presets(preset_rows, fixed_params)
    if args.dry_run:
        print(content, end="")
    else:
        args.preset.parent.mkdir(parents=True, exist_ok=True)
        args.preset.write_text(content)
        print(f"Updated {args.preset} with {len(preset_rows)} ranked model preset(s).")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def load_rank_module() -> Any:
    spec = importlib.util.spec_from_file_location("rank_buun_results", RANK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RANK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_model(model_value: str, display_value: str, warnings: list[str]) -> str:
    for candidate in [model_value.strip(), display_value.strip()]:
        if not candidate:
            continue
        p = model_path(candidate)
        if p.exists():
            return candidate

    display_name = display_value.split(" [", 1)[0].strip()
    if not display_name:
        warnings.append(f"skipping unresolved model row: {display_value or model_value}")
        return ""
    matches = [m for m in find_models() if Path(m).name == display_name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        warnings.append(f"skipping ambiguous model name {display_name!r}: {', '.join(matches)}")
    else:
        warnings.append(f"skipping missing model {display_name!r}")
    return ""


def render_presets(rows: list[PresetRow], fixed_params: list[tuple[str, str]]) -> str:
    lines = [
        "version = 1",
        "",
        "; Model presets for buun llama-server router mode.",
        "; Generated by scripts/update-buun-model-presets.py from bench-results/buun-tui summaries.",
        "; [default] is the best ranked recorded result; named sections are additional ranked models.",
        "; Fixed llama-server args are read from versions/buun/model-preset-fixed-args.txt.",
        "",
    ]
    lines.extend(render_section("default", rows[0], fixed_params, load_on_startup=True))
    for row in rows[1:]:
        lines.append("")
        lines.extend(render_section(row.section, row, fixed_params, load_on_startup=False))
    return "\n".join(lines).rstrip() + "\n"


def render_section(name: str, row: PresetRow, fixed_params: list[tuple[str, str]], load_on_startup: bool) -> list[str]:
    params: list[tuple[str, str]] = [
        ("model", row.model),
        ("ctx-size", str(row.ctx_size)),
    ]
    if row.cache_type_k:
        params.append(("cache-type-k", row.cache_type_k))
    if row.cache_type_v:
        params.append(("cache-type-v", row.cache_type_v))
    if row.threads is not None:
        params.append(("threads", str(row.threads)))
        params.append(("threads-batch", str(row.threads)))
    if row.n_batch is not None:
        params.append(("batch-size", str(row.n_batch)))
    if row.n_ubatch is not None:
        params.append(("ubatch-size", str(row.n_ubatch)))
    if row.n_cpu_moe is not None:
        params.append(("n-cpu-moe", str(row.n_cpu_moe)))
    if row.n_gpu_layers is not None:
        params.append(("n-gpu-layers", str(row.n_gpu_layers)))
    params.extend([
        ("device", "CUDA0"),
        ("parallel", "1"),
        ("flash-attn", "on" if row.flash_attn == "on" else "off"),
        ("cache-ram", "0"),
        ("no-cache-idle-slots", "true"),
        ("no-mmproj", "true"),
    ])
    params = merge_params(params, fixed_params)
    params = merge_params(params, [("load-on-startup", "true" if load_on_startup else "false"), ("stop-timeout", "10")])

    lines = [
        f"[{name}]",
        f"; score={fmt(row.score)} pp_tps={fmt(row.pp_tps)} tg_tps={fmt(row.tg_tps)} samples={row.sample_count or ''}",
    ]
    lines.extend(f"{key} = {value}" for key, value in params)
    return lines


def load_fixed_params(path: Path, snippets: list[str]) -> list[tuple[str, str]]:
    parts: list[str] = []
    if path.exists():
        parts.append(path.read_text())
    parts.extend(snippets)
    return parse_llama_server_args("\n".join(parts)) if parts else []


def parse_llama_server_args(text: str) -> list[tuple[str, str]]:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        cleaned_lines.append(line.rstrip("\\").strip())
    tokens = shlex.split(" ".join(cleaned_lines), comments=True, posix=True)
    params: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"fixed arg must start with --: {token!r}")
        flag = token[2:]
        if not flag:
            raise ValueError("empty fixed arg flag")
        if "=" in flag:
            key, value = flag.split("=", 1)
            params.append((key, value))
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            params.append((flag, tokens[i + 1]))
            i += 2
        else:
            params.append((flag, "true"))
            i += 1
    return merge_params([], params)


def merge_params(base: list[tuple[str, str]], updates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out = list(base)
    for key, value in updates:
        out = [(existing_key, existing_value) for existing_key, existing_value in out if existing_key != key]
        out.append((key, value))
    return out


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "model"


def unique_section_name(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    i = 2
    while f"{base}-{i}" in used:
        i += 1
    return f"{base}-{i}"


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())

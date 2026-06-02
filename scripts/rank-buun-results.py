#!/usr/bin/env python3
"""Rank finalized buun benchmark runs by paired PP/TG throughput."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from bench_buun_tui.models import find_models, model_label
from bench_buun_tui.paths import RESULT_DIR, ROOT

PAIR_TESTS = {"pp512", "tg128"}
KEY_FIELDS = [
    "model_path",
    "cache_strategy",
    "n_batch",
    "n_ubatch",
    "n_threads",
    "n_cpu_moe",
    "n_gpu_layers",
    "flash_attn",
    "type_k",
    "type_v",
]
OPTIONAL_KEY_FIELDS = ["model_layers", "cpu_moe_layers", "remaining_moe_layers"]
SUMMARY_FIELDS = [
    "kind",
    "depth_rank",
    "depth",
    "run_dir",
    "profile",
    "model",
    "model_path",
    "cache_strategy",
    "n_batch",
    "n_ubatch",
    "n_threads",
    "n_cpu_moe",
    "n_gpu_layers",
    "flash_attn",
    "type_k",
    "type_v",
    "model_layers",
    "cpu_moe_layers",
    "remaining_moe_layers",
    "pp_tps",
    "tg_tps",
    "pp_wall_s",
    "tg_wall_s",
    "wall_s",
    "repetitions",
    "normalized_pp",
    "normalized_tg",
    "score",
    "sample_count",
    "config_key",
]


@dataclass(frozen=True)
class GroupKey:
    run_dir: str
    profile: str
    config_key: str
    depth: int


@dataclass
class Sample:
    run_dir: str
    profile: str
    depth: int
    model: str
    model_path: str
    cache_strategy: str
    n_batch: int | None
    n_ubatch: int | None
    n_threads: int | None
    n_cpu_moe: int | None
    n_gpu_layers: int | None
    flash_attn: str
    type_k: str
    type_v: str
    model_layers: int | None
    cpu_moe_layers: int | None
    remaining_moe_layers: int | None
    pp_tps: float | None = None
    tg_tps: float | None = None
    pp_stddev: float | None = None
    tg_stddev: float | None = None
    pp_wall_s: float | None = None
    tg_wall_s: float | None = None
    wall_s: float | None = None
    repetitions: int | None = None
    sample_count: int = 1
    config_key: str = ""
    normalized_pp: float | None = None
    normalized_tg: float | None = None
    score: float | None = None


@dataclass
class ConfigAggregate:
    config_key: str
    model: str
    model_path: str
    cache_strategy: str
    n_batch: int | None
    n_ubatch: int | None
    n_threads: int | None
    n_cpu_moe: int | None
    n_gpu_layers: int | None
    flash_attn: str
    type_k: str
    type_v: str
    model_layers: int | None
    cpu_moe_layers: int | None
    remaining_moe_layers: int | None
    pp_values: list[float]
    tg_values: list[float]
    wall_values: list[float]
    scores: list[float]
    sample_count: int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULT_DIR, help="benchmark results directory")
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "ranking.md", help="markdown report path")
    parser.add_argument("--csv-output", type=Path, default=None, help="csv report path (defaults to sibling ranking.txt)")
    parser.add_argument("--html-output", type=Path, default=None, help="html report path (defaults to input/ranking.html)")
    parser.add_argument("--min-samples", type=int, default=1, help="minimum paired samples required for overall ranking")
    parser.add_argument("--score", choices=["normalized-geomean", "simple-geomean"], default="normalized-geomean")
    parser.add_argument("--include-test-runs", action="store_true", help="include run directories whose name contains 'test'")
    parser.add_argument("--latest-only", action="store_true", help="only rank the latest run directory")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_md = args.output.resolve()
    output_csv = (args.csv_output.resolve() if args.csv_output else output_md.with_suffix(".txt"))
    output_html = (args.html_output.resolve() if args.html_output else input_dir / "ranking.html")

    summaries = discover_summary_files(input_dir, include_test_runs=args.include_test_runs, latest_only=args.latest_only)
    rows = load_rows(summaries)
    if not rows:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("# Buun Benchmark Ranking\n\nNo finalized benchmark rows found.\n")
        write_csv(output_csv, [])
        output_html.write_text(render_html(output_html, input_dir, [], [], [], [], [], [], args, output_csv, output_md, message="No finalized benchmark rows found."))
        print(f"Wrote {output_html}")
        print(f"No ranking data found under {input_dir}")
        return 0

    paired_samples = build_paired_samples(rows)
    if not paired_samples:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("# Buun Benchmark Ranking\n\nNo paired PP/TG rows were found.\n")
        write_csv(output_csv, [])
        output_html.write_text(render_html(output_html, input_dir, [], [], [], [], [], [], args, output_csv, output_md, message="No paired PP/TG rows were found."))
        print(f"Wrote {output_html}")
        print("No paired PP/TG samples found.")
        return 0

    depth_best, pp_best, tg_best = rank_by_depth(paired_samples, score_mode=args.score)
    overall = rank_overall(paired_samples, min_samples=max(1, args.min_samples), score_mode=args.score)
    untested_models = find_untested_models(paired_samples)
    all_candidate_rows = rows_to_csv_rows(paired_samples, args.score)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, all_candidate_rows)
    output_md.write_text(render_markdown(output_md, input_dir, overall, depth_best, pp_best, tg_best, paired_samples, untested_models, args))
    output_html.write_text(render_html(output_html, input_dir, overall, depth_best, pp_best, tg_best, paired_samples, untested_models, args, output_csv, output_md))
    print(f"Wrote {output_md}")
    print(f"Wrote {output_html}")
    print(f"Wrote {output_csv}")
    return 0


def discover_summary_files(input_dir: Path, include_test_runs: bool, latest_only: bool) -> list[Path]:
    files = sorted(input_dir.glob("*/summary.csv"))
    selected: list[Path] = []
    for path in files:
        run_dir = path.parent
        if not include_test_runs and "test" in run_dir.name.lower():
            continue
        selected.append(path)
    if not selected:
        return []
    if latest_only:
        latest = max(selected, key=lambda p: (p.parent.name, p.stat().st_mtime))
        return [latest]
    return selected


def load_rows(summary_files: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in summary_files:
        run_dir = path.parent.name
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["run_dir"] = run_dir
                out.append(row)
    return out


def display_model_label(model: str, model_path: str = "") -> str:
    if model and "[" in model and "]" in model:
        return model
    if model_path:
        return model_label(model_path)
    if model:
        return model_label(model)
    return ""


def build_paired_samples(rows: list[dict[str, Any]]) -> list[Sample]:
    grouped: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        test = (row.get("test") or "").strip()
        if test not in PAIR_TESTS:
            continue
        key = GroupKey(
            run_dir=str(row.get("run_dir") or ""),
            profile=str(row.get("profile") or ""),
            config_key=profile_config_key(row),
            depth=parse_int(row.get("n_depth")) or 0,
        )
        grouped[key].append(row)

    samples: list[Sample] = []
    for key, items in grouped.items():
        pp_rows = [row for row in items if (row.get("test") or "").strip() == "pp512"]
        tg_rows = [row for row in items if (row.get("test") or "").strip() == "tg128"]
        if not pp_rows or not tg_rows:
            continue
        rep = items[0]
        pp_vals = [v for v in (parse_float(r.get("avg_ts")) for r in pp_rows) if v is not None]
        tg_vals = [v for v in (parse_float(r.get("avg_ts")) for r in tg_rows) if v is not None]
        if not pp_vals or not tg_vals:
            continue
        pp_stddev_vals = [v for v in (parse_float(r.get("stddev_ts")) for r in pp_rows) if v is not None]
        tg_stddev_vals = [v for v in (parse_float(r.get("stddev_ts")) for r in tg_rows) if v is not None]
        pp_wall_vals = [infer_row_wall_s(r) for r in pp_rows if infer_row_wall_s(r) is not None]
        tg_wall_vals = [infer_row_wall_s(r) for r in tg_rows if infer_row_wall_s(r) is not None]
        pp_wall_s = median(pp_wall_vals) if pp_wall_vals else None
        tg_wall_s = median(tg_wall_vals) if tg_wall_vals else None
        samples.append(
            Sample(
                run_dir=key.run_dir,
                profile=key.profile,
                depth=key.depth,
                model=str(rep.get("model") or model_label(str(rep.get("model_path") or "")) or Path(str(rep.get("model_path") or "")).name),
                model_path=str(rep.get("model_path") or ""),
                cache_strategy=str(rep.get("cache_strategy") or ""),
                n_batch=parse_int(rep.get("n_batch")),
                n_ubatch=parse_int(rep.get("n_ubatch")),
                n_threads=parse_int(rep.get("n_threads")),
                n_cpu_moe=parse_int(rep.get("n_cpu_moe")),
                n_gpu_layers=parse_int(rep.get("n_gpu_layers")),
                flash_attn=normalize_bool_text(rep.get("flash_attn")),
                type_k=str(rep.get("type_k") or ""),
                type_v=str(rep.get("type_v") or ""),
                model_layers=parse_int(rep.get("model_layers")),
                cpu_moe_layers=parse_int(rep.get("cpu_moe_layers")),
                remaining_moe_layers=parse_int(rep.get("remaining_moe_layers")),
                pp_tps=median(pp_vals),
                tg_tps=median(tg_vals),
                pp_stddev=median(pp_stddev_vals) if pp_stddev_vals else None,
                tg_stddev=median(tg_stddev_vals) if tg_stddev_vals else None,
                pp_wall_s=pp_wall_s,
                tg_wall_s=tg_wall_s,
                wall_s=(pp_wall_s + tg_wall_s) if (pp_wall_s is not None and tg_wall_s is not None) else (pp_wall_s if pp_wall_s is not None else tg_wall_s),
                repetitions=parse_int(rep.get("repetitions")),
                sample_count=min(len(pp_vals), len(tg_vals)),
                config_key=key.config_key,
            )
        )
    return samples


def rank_by_depth(samples: list[Sample], score_mode: str) -> tuple[dict[int, Sample], dict[int, Sample], dict[int, Sample]]:
    by_depth: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_depth[sample.depth].append(sample)

    depth_best: dict[int, Sample] = {}
    pp_best: dict[int, Sample] = {}
    tg_best: dict[int, Sample] = {}
    for depth, items in by_depth.items():
        pp_best[depth] = max(items, key=lambda s: (s.pp_tps or 0.0, s.tg_tps or 0.0, s.model_path))
        tg_best[depth] = max(items, key=lambda s: (s.tg_tps or 0.0, s.pp_tps or 0.0, s.model_path))
        ranked = score_samples(items, score_mode)
        if ranked:
            depth_best[depth] = ranked[0]
    return depth_best, pp_best, tg_best


def score_samples(samples: list[Sample], score_mode: str) -> list[Sample]:
    if score_mode == "simple-geomean":
        ranked = []
        for sample in samples:
            if sample.pp_tps is None or sample.tg_tps is None:
                continue
            sample.normalized_pp = None
            sample.normalized_tg = None
            sample.score = math.sqrt(max(sample.pp_tps, 0.0) * max(sample.tg_tps, 0.0))
            ranked.append(sample)
        return sorted(ranked, key=lambda s: (s.score or 0.0, s.pp_tps or 0.0, s.tg_tps or 0.0), reverse=True)

    by_depth: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.pp_tps is not None and sample.tg_tps is not None:
            by_depth[sample.depth].append(sample)

    ranked: list[Sample] = []
    for depth, items in by_depth.items():
        best_pp = max((s.pp_tps or 0.0) for s in items) or 1.0
        best_tg = max((s.tg_tps or 0.0) for s in items) or 1.0
        best_wall = min((s.wall_s for s in items if s.wall_s is not None and s.wall_s > 0), default=None)
        for sample in items:
            sample.normalized_pp = (sample.pp_tps or 0.0) / best_pp if best_pp else 0.0
            sample.normalized_tg = (sample.tg_tps or 0.0) / best_tg if best_tg else 0.0
            normalized_wall = (best_wall / sample.wall_s) if best_wall and sample.wall_s and sample.wall_s > 0 else 0.0
            sample.score = math.pow(max(sample.normalized_pp, 0.0) * max(sample.normalized_tg, 0.0) * max(normalized_wall, 0.0), 1.0 / 3.0)
            ranked.append(sample)
    return sorted(ranked, key=lambda s: (s.score or 0.0, s.pp_tps or 0.0, s.tg_tps or 0.0), reverse=True)


def rank_overall(samples: list[Sample], min_samples: int, score_mode: str) -> list[ConfigAggregate]:
    aggregates: dict[str, ConfigAggregate] = {}
    for sample in samples:
        agg = aggregates.get(sample.config_key)
        if agg is None:
            agg = ConfigAggregate(
                config_key=sample.config_key,
                model=sample.model,
                model_path=sample.model_path,
                cache_strategy=sample.cache_strategy,
                n_batch=sample.n_batch,
                n_ubatch=sample.n_ubatch,
                n_threads=sample.n_threads,
                n_cpu_moe=sample.n_cpu_moe,
                n_gpu_layers=sample.n_gpu_layers,
                flash_attn=sample.flash_attn,
                type_k=sample.type_k,
                type_v=sample.type_v,
                model_layers=sample.model_layers,
                cpu_moe_layers=sample.cpu_moe_layers,
                remaining_moe_layers=sample.remaining_moe_layers,
                pp_values=[],
                tg_values=[],
                wall_values=[],
                scores=[],
                sample_count=0,
            )
            aggregates[sample.config_key] = agg
        if sample.pp_tps is not None:
            agg.pp_values.append(sample.pp_tps)
        if sample.tg_tps is not None:
            agg.tg_values.append(sample.tg_tps)
        if sample.wall_s is not None:
            agg.wall_values.append(sample.wall_s)
        if sample.score is None:
            # score_samples() mutates the samples used by the depth ranking.
            # Recompute on demand here to keep overall ranking independent.
            if score_mode == "simple-geomean":
                sample.score = math.sqrt(max(sample.pp_tps or 0.0, 0.0) * max(sample.tg_tps or 0.0, 0.0))
            else:
                sample.score = math.pow(max(sample.pp_tps or 0.0, 0.0) * max(sample.tg_tps or 0.0, 0.0), 0.5)
        agg.scores.append(sample.score or 0.0)
        agg.sample_count += 1

    ranked = [
        agg
        for agg in aggregates.values()
        if agg.sample_count >= min_samples and agg.pp_values and agg.tg_values
    ]
    return sorted(
        ranked,
        key=lambda a: (
            median(a.scores) if a.scores else 0.0,
            median(a.tg_values) if a.tg_values else 0.0,
            median(a.pp_values) if a.pp_values else 0.0,
        ),
        reverse=True,
    )


def rows_to_csv_rows(samples: list[Sample], score_mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_depth = {depth: score_samples([s for s in samples if s.depth == depth], score_mode) for depth in {sample.depth for sample in samples}}
    depth_ranks = {
        depth: {sample.config_key: idx + 1 for idx, sample in enumerate(ranked)}
        for depth, ranked in per_depth.items()
    }
    for sample in sorted(samples, key=lambda s: (s.depth, -(s.score or 0.0), s.model_path, s.profile, s.run_dir)):
        rows.append(
            {
                "kind": "paired",
                "depth_rank": depth_ranks.get(sample.depth, {}).get(sample.config_key, ""),
                "depth": sample.depth,
                "run_dir": sample.run_dir,
                "profile": sample.profile,
                "model": sample.model,
                "model_path": sample.model_path,
                "cache_strategy": sample.cache_strategy,
                "n_batch": sample.n_batch,
                "n_ubatch": sample.n_ubatch,
                "n_threads": sample.n_threads,
                "n_cpu_moe": sample.n_cpu_moe,
                "n_gpu_layers": sample.n_gpu_layers,
                "flash_attn": sample.flash_attn,
                "type_k": sample.type_k,
                "type_v": sample.type_v,
                "model_layers": sample.model_layers,
                "cpu_moe_layers": sample.cpu_moe_layers,
                "remaining_moe_layers": sample.remaining_moe_layers,
                "pp_tps": sample.pp_tps,
                "tg_tps": sample.tg_tps,
                "pp_wall_s": sample.pp_wall_s,
                "tg_wall_s": sample.tg_wall_s,
                "wall_s": sample.wall_s,
                "repetitions": sample.repetitions,
                "normalized_pp": sample.normalized_pp,
                "normalized_tg": sample.normalized_tg,
                "score": sample.score,
                "sample_count": sample.sample_count,
                "config_key": sample.config_key,
            }
        )
    return rows


def render_markdown(output_md: Path, input_dir: Path, overall: list[ConfigAggregate], depth_best: dict[int, Sample], pp_best: dict[int, Sample], tg_best: dict[int, Sample], paired_samples: list[Sample], untested_models: list[str], args: argparse.Namespace) -> str:
    lines: list[str] = ["# Buun Benchmark Ranking", ""]
    lines.append(f"Input: `{input_dir}`  ")
    lines.append(f"Paired samples: {len(paired_samples)}  ")
    lines.append(f"Score mode: `{args.score}`  ")
    lines.append("Wall time is part of the normalized score and is measured as elapsed benchmark-process time for each paired PP/TG sample.")
    lines.append("")

    lines.append("## Overall Winner")
    lines.append("")
    if overall:
        winner = overall[0]
        lines.extend([
            f"Model: `{display_model_label(winner.model, winner.model_path)}`",
            f"Config: {format_config_inline(winner)}",
            f"Median PP TPS: `{fmt_num(median(winner.pp_values))}`",
            f"Median TG TPS: `{fmt_num(median(winner.tg_values))}`",
            f"Median wall time: `{fmt_num(median_or_none(winner.wall_values))}`",
            f"Median score: `{fmt_num(median(winner.scores))}`",
            f"Samples: `{winner.sample_count}`",
            "",
            "How to run the same params on llama-server:",
            "",
            "```sh",
            llama_server_command(winner, max_depth_for_config(paired_samples, winner.config_key)),
            "```",
            "",
        ])
    else:
        lines.append("No overall winner found.\n")

    lines.append("## Top Overall Rankings")
    lines.append("")
    lines.append("| Rank | Model | Score | PP TPS | TG TPS | Wall | Samples | Config |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for rank, agg in enumerate(overall[:5], start=1):
        lines.append(
            f"| {rank} | `{display_model_label(agg.model, agg.model_path)}` | {fmt_num(median(agg.scores))} | {fmt_num(median(agg.pp_values))} | {fmt_num(median(agg.tg_values))} | {fmt_num(median_or_none(agg.wall_values))} | `{agg.sample_count}` | {format_config_inline(agg)} |"
        )
    lines.append("")

    lines.append("## Fastest Overall by Wall Time")
    lines.append("")
    lines.append("| Rank | Model | Wall | PP TPS | TG TPS | Score | Samples | Config |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for rank, agg in enumerate(sorted(overall, key=lambda a: (median_or_none(a.wall_values) if a.wall_values else float('inf'), -(median_or_none(a.scores) or 0.0)))[:5], start=1):
        lines.append(
            f"| {rank} | `{display_model_label(agg.model, agg.model_path)}` | {fmt_num(median_or_none(agg.wall_values))} | {fmt_num(median(agg.pp_values))} | {fmt_num(median(agg.tg_values))} | {fmt_num(median(agg.scores))} | `{agg.sample_count}` | {format_config_inline(agg)} |"
        )
    lines.append("")

    lines.append("## Winners by Context Depth")
    lines.append("")
    lines.append("| Depth | Rank | Model | PP TPS | TG TPS | Wall | Score | Run | Profile |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---|---|")
    for depth in sorted(depth_best):
        sample = depth_best[depth]
        lines.append(
            f"| {depth} | 1 | `{display_model_label(sample.model, sample.model_path)}` | {fmt_num(sample.pp_tps)} | {fmt_num(sample.tg_tps)} | {fmt_num(sample.wall_s)} | {fmt_num(sample.score)} | `{sample.run_dir}` | `{sample.profile}` |"
        )
    lines.append("")

    lines.append("## Best PP-only by Depth")
    lines.append("")
    lines.append("| Depth | Model | PP TPS | Wall | Run | Profile |")
    lines.append("|---:|---|---:|---:|---|---|")
    for depth in sorted(pp_best):
        sample = pp_best[depth]
        lines.append(f"| {depth} | `{display_model_label(sample.model, sample.model_path)}` | {fmt_num(sample.pp_tps)} | {fmt_num(sample.pp_wall_s)} | `{sample.run_dir}` | `{sample.profile}` |")
    lines.append("")

    lines.append("## Best TG-only by Depth")
    lines.append("")
    lines.append("| Depth | Model | TG TPS | Wall | Run | Profile |")
    lines.append("|---:|---|---:|---:|---|---|")
    for depth in sorted(tg_best):
        sample = tg_best[depth]
        lines.append(f"| {depth} | `{display_model_label(sample.model, sample.model_path)}` | {fmt_num(sample.tg_tps)} | {fmt_num(sample.tg_wall_s)} | `{sample.run_dir}` | `{sample.profile}` |")
    lines.append("")

    lines.append("## Untested Models")
    lines.append("")
    if untested_models:
        lines.append(f"Untested models under `models/`: `{len(untested_models)}`")
        lines.append("")
        lines.append("| Model |")
        lines.append("|---|")
        for model in untested_models:
            lines.append(f"| `{model_label(model)}` |")
    else:
        lines.append("All models under `models/` have been tested.")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def find_untested_models(samples: list[Sample]) -> list[str]:
    tested = {str(sample.model_path).strip() for sample in samples if str(sample.model_path).strip()}
    return [model for model in find_models() if model not in tested]


def max_depth_for_config(samples: list[Sample], config_key: str) -> int:
    depths = [sample.depth for sample in samples if sample.config_key == config_key]
    return max(depths) if depths else 0


def llama_server_command(winner: ConfigAggregate, ctx_size: int) -> str:
    cmd = [
        "llama-server",
        "--host 0.0.0.0",
        "--port 8080",
        f"--model {quote_shell(winner.model_path)}",
        f"--ctx-size {ctx_size or 262144}",
    ]
    if winner.type_k:
        cmd.append(f"--cache-type-k {quote_shell(winner.type_k)}")
    if winner.type_v:
        cmd.append(f"--cache-type-v {quote_shell(winner.type_v)}")
    if winner.n_threads is not None:
        cmd.append(f"--threads {winner.n_threads}")
        cmd.append(f"--threads-batch {winner.n_threads}")
    if winner.n_batch is not None:
        cmd.append(f"--batch-size {winner.n_batch}")
    if winner.n_ubatch is not None:
        cmd.append(f"--ubatch-size {winner.n_ubatch}")
    if winner.n_cpu_moe is not None:
        cmd.append(f"--n-cpu-moe {winner.n_cpu_moe}")
    if winner.n_gpu_layers is not None:
        cmd.append(f"--n-gpu-layers {winner.n_gpu_layers}")
    if winner.flash_attn:
        cmd.append(f"--flash-attn {'on' if normalize_bool_text(winner.flash_attn) == 'on' else 'off'}")
    cmd.extend(["--parallel 1", "--metrics", "--reasoning off", "--no-mmproj"])
    separator = " \\" + "\n  "
    return separator.join(cmd)


def llama_server_command_for_sample(sample: Sample) -> str:
    cmd = [
        "llama-server",
        "--host 0.0.0.0",
        "--port 8080",
        f"--model {quote_shell(sample.model_path)}",
        f"--ctx-size {sample.depth or 262144}",
    ]
    if sample.type_k:
        cmd.append(f"--cache-type-k {quote_shell(sample.type_k)}")
    if sample.type_v:
        cmd.append(f"--cache-type-v {quote_shell(sample.type_v)}")
    if sample.n_threads is not None:
        cmd.append(f"--threads {sample.n_threads}")
        cmd.append(f"--threads-batch {sample.n_threads}")
    if sample.n_batch is not None:
        cmd.append(f"--batch-size {sample.n_batch}")
    if sample.n_ubatch is not None:
        cmd.append(f"--ubatch-size {sample.n_ubatch}")
    if sample.n_cpu_moe is not None:
        cmd.append(f"--n-cpu-moe {sample.n_cpu_moe}")
    if sample.n_gpu_layers is not None:
        cmd.append(f"--n-gpu-layers {sample.n_gpu_layers}")
    if sample.flash_attn:
        cmd.append(f"--flash-attn {'on' if normalize_bool_text(sample.flash_attn) == 'on' else 'off'}")
    cmd.extend(["--parallel 1", "--metrics", "--reasoning off", "--no-mmproj"])
    separator = " \\" + "\n  "
    return separator.join(cmd)


def format_config_inline(agg: ConfigAggregate) -> str:
    parts = []
    if agg.cache_strategy:
        parts.append(f"cache_strategy={agg.cache_strategy}")
    if agg.n_batch is not None or agg.n_ubatch is not None:
        parts.append(f"batch/ubatch={agg.n_batch}/{agg.n_ubatch}")
    if agg.n_threads is not None:
        parts.append(f"threads={agg.n_threads}")
    if agg.n_cpu_moe is not None:
        parts.append(f"n_cpu_moe={agg.n_cpu_moe}")
    if agg.n_gpu_layers is not None:
        parts.append(f"n_gpu_layers={agg.n_gpu_layers}")
    if agg.flash_attn:
        parts.append(f"flash_attn={agg.flash_attn}")
    if agg.type_k or agg.type_v:
        parts.append(f"type_k/type_v={agg.type_k}/{agg.type_v}")
    return ", ".join(parts)


def profile_config_key(row: dict[str, Any]) -> str:
    values = [str(row.get(field) or "").strip() for field in KEY_FIELDS]
    for field in OPTIONAL_KEY_FIELDS:
        value = str(row.get(field) or "").strip()
        if value:
            values.append(f"{field}={value}")
    return "|".join(values)


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def infer_row_wall_s(row: dict[str, Any]) -> float | None:
    wall_s = parse_float(row.get("wall_s"))
    if wall_s and wall_s > 0:
        return wall_s
    avg_ns = parse_float(row.get("avg_ns"))
    reps = parse_int(row.get("repetitions")) or 1
    if avg_ns is not None and avg_ns > 0 and reps > 0:
        return avg_ns * reps / 1e9
    return None


def normalize_bool_text(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return "on"
    if text in {"0", "false", "no", "off"}:
        return "off"
    return str(value).strip()


def quote_shell(value: str) -> str:
    if value == "":
        return "''"
    if all(ch.isalnum() or ch in "-._/:" for ch in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def fmt_num(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}".rstrip("0").rstrip(".")


def median_or_none(values: list[float]) -> float | None:
    return median(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_html(
    output_html: Path,
    input_dir: Path,
    overall: list[ConfigAggregate],
    depth_best: dict[int, Sample],
    pp_best: dict[int, Sample],
    tg_best: dict[int, Sample],
    paired_samples: list[Sample],
    untested_models: list[str],
    args: argparse.Namespace,
    csv_path: Path,
    md_path: Path,
    message: str | None = None,
) -> str:
    class RawHTML(str):
        pass

    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value), quote=True)

    def row(cells: list[Any]) -> str:
        def render_cell(value: Any) -> str:
            return str(value) if isinstance(value, RawHTML) else esc(value)

        return "<tr>" + "".join(f"<td>{render_cell(cell)}</td>" for cell in cells) + "</tr>"

    def table(headers: list[str], rows: list[list[Any]]) -> str:
        return (
            "<table>"
            + "<thead><tr>"
            + "".join(f"<th>{esc(h)}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + ("".join(row(r) for r in rows) if rows else f"<tr><td colspan='{len(headers)}'>No rows</td></tr>")
            + "</tbody></table>"
        )

    def model_cell(sample: Sample) -> RawHTML:
        label = display_model_label(sample.model, sample.model_path)
        command = llama_server_command_for_sample(sample)
        return RawHTML(
            f"<span>{esc(label)}</span> "
            f"<button type='button' class='copy-cmd' data-cmd='{esc(command)}' onclick='copyCmd(this)'>copy llama-server cmd</button>"
        )

    def aggregate_model_cell(agg: ConfigAggregate) -> RawHTML:
        label = display_model_label(agg.model, agg.model_path)
        command = llama_server_command(agg, max_depth_for_config(paired_samples, agg.config_key))
        return RawHTML(
            f"<span>{esc(label)}</span> "
            f"<button type='button' class='copy-cmd' data-cmd='{esc(command)}' onclick='copyCmd(this)'>copy llama-server cmd</button>"
        )

    links = f"<a href='{esc(csv_path.name)}'>ranking.txt</a> · <a href='{esc(md_path.name)}'>ranking.md</a>"
    overall_block = ""
    if overall:
        winner = overall[0]
        command = llama_server_command(winner, max_depth_for_config(paired_samples, winner.config_key))
        overall_block = (
            f"<p><b>Model:</b> <code>{esc(display_model_label(winner.model, winner.model_path))}</code> "
            f"<button type='button' class='copy-cmd' data-cmd='{esc(command)}' onclick='copyCmd(this)'>copy llama-server cmd</button></p>"
            f"<p><b>Config:</b> <code>{esc(format_config_inline(winner))}</code></p>"
            f"<p><b>Median PP TPS:</b> <code>{esc(fmt_num(median(winner.pp_values)))}</code> "
            f"<b>Median TG TPS:</b> <code>{esc(fmt_num(median(winner.tg_values)))}</code> "
            f"<b>Median wall time:</b> <code>{esc(fmt_num(median_or_none(winner.wall_values)))}</code> "
            f"<b>Median score:</b> <code>{esc(fmt_num(median(winner.scores)))}</code> "
            f"<b>Samples:</b> <code>{winner.sample_count}</code></p>"
            f"<pre>{esc(command)}</pre>"
        )
    else:
        overall_block = "<p>No overall winner found.</p>"

    overall_rows = [[rank, aggregate_model_cell(agg), fmt_num(median(agg.scores)), fmt_num(median(agg.pp_values)), fmt_num(median(agg.tg_values)), fmt_num(median_or_none(agg.wall_values)), agg.sample_count, format_config_inline(agg)] for rank, agg in enumerate(overall[:5], start=1)]
    wall_ranked = sorted(overall, key=lambda a: (median_or_none(a.wall_values) if a.wall_values else float('inf'), -(median_or_none(a.scores) or 0.0)))
    wall_rows = [[rank, aggregate_model_cell(agg), fmt_num(median_or_none(agg.wall_values)), fmt_num(median(agg.pp_values)), fmt_num(median(agg.tg_values)), fmt_num(median(agg.scores)), agg.sample_count, format_config_inline(agg)] for rank, agg in enumerate(wall_ranked[:5], start=1)]
    depth_rows = [[depth, 1, model_cell(sample), fmt_num(sample.pp_tps), fmt_num(sample.tg_tps), fmt_num(sample.wall_s), fmt_num(sample.score), sample.run_dir, sample.profile] for depth, sample in sorted(depth_best.items())]
    pp_rows = [[depth, model_cell(sample), fmt_num(sample.pp_tps), fmt_num(sample.pp_wall_s), sample.run_dir, sample.profile] for depth, sample in sorted(pp_best.items())]
    tg_rows = [[depth, model_cell(sample), fmt_num(sample.tg_tps), fmt_num(sample.tg_wall_s), sample.run_dir, sample.profile] for depth, sample in sorted(tg_best.items())]
    untested_rows = [[model] for model in untested_models]

    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Buun Benchmark Ranking</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;line-height:1.4}",
        "table{border-collapse:collapse;margin:16px 0;width:100%}",
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top;font-size:13px}",
        "th{background:#f4f4f4}",
        "pre{white-space:pre-wrap;background:#fafafa;border:1px solid #ddd;padding:12px;overflow:auto}",
        "code{background:#f5f5f5;padding:0 3px;border-radius:3px}",
        ".muted{color:#666}",
        ".copy-cmd{margin-left:8px;padding:2px 6px;font-size:12px;cursor:pointer}",
        "</style>",
        "<script>",
        "async function copyCmd(button){const cmd=button?.dataset?.cmd||'';const done=()=>{const old=button.textContent;button.textContent='copied';setTimeout(()=>button.textContent=old,1200);};try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(cmd);done();return;}const ta=document.createElement('textarea');ta.value=cmd;ta.style.position='fixed';ta.style.opacity='0';ta.style.left='-9999px';document.body.appendChild(ta);ta.focus();ta.select();const ok=document.execCommand('copy');ta.remove();if(ok){done();return;}throw new Error('execCommand returned false');}catch(err){console.error('clipboard copy failed', err);button.textContent='copy failed';setTimeout(()=>button.textContent='copy llama-server cmd',1200);}}",
        "</script></head><body>",
        "<h1>Buun Benchmark Ranking</h1>",
        f"<p class='muted'>Input: <code>{esc(input_dir)}</code> · Paired samples: <code>{len(paired_samples)}</code> · Score: <code>{esc(args.score)}</code> · {links}</p>",
        "<p class='muted'>Wall time is part of the normalized score and is measured as elapsed benchmark-process time for each paired PP/TG sample.</p>",
    ]
    if message:
        body.append(f"<p>{esc(message)}</p>")
    body.extend([
        "<h2>Overall Winner</h2>",
        overall_block,
        "<h2>Top Overall Rankings</h2>",
        table(["Rank", "Model", "Score", "PP TPS", "TG TPS", "Wall", "Samples", "Config"], overall_rows),
        "<h2>Fastest Overall by Wall Time</h2>",
        table(["Rank", "Model", "Wall", "PP TPS", "TG TPS", "Score", "Samples", "Config"], wall_rows),
        "<h2>Winners by Context Depth</h2>",
        table(["Depth", "Rank", "Model", "PP TPS", "TG TPS", "Wall", "Score", "Run", "Profile"], depth_rows),
        "<h2>Best PP-only by Depth</h2>",
        table(["Depth", "Model", "PP TPS", "Wall", "Run", "Profile"], pp_rows),
        "<h2>Best TG-only by Depth</h2>",
        table(["Depth", "Model", "TG TPS", "Wall", "Run", "Profile"], tg_rows),
        "<h2>Untested Models</h2>",
        (f"<p>Untested models under <code>models/</code>: <code>{len(untested_models)}</code></p>" if untested_models else "<p>All models under <code>models/</code> have been tested.</p>"),
        table(["Model"], untested_rows),
        "</body></html>",
    ])
    return "\n".join(body)


if __name__ == "__main__":
    raise SystemExit(main())

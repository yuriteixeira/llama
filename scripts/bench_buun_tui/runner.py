from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

from .commands import is_set, profile_command
from .config_store import load_config, slugify
from .paths import RESULT_DIR, ROOT
from .reports import SUMMARY_FIELDS, summary_row, write_html_report, write_summary
from .telemetry import capture_telemetry


def run_config(config_path: Path) -> int:
    cfg = load_config(config_path)
    started_at = time.time()
    out_dir = create_result_dir(cfg.run_prefix)
    live_summary = init_live_summary(out_dir)
    all_rows: list[dict] = []
    telemetry_rows: list[dict] = []
    errors: list[dict] = []

    for profile in cfg.profiles:
        error = run_profile(profile, out_dir, live_summary, all_rows, telemetry_rows)
        if error:
            errors.append(error)
            break

    finished_at = time.time()
    run_meta = {
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": finished_at - started_at,
    }
    write_final_outputs(out_dir, live_summary, all_rows, telemetry_rows, errors, run_meta)
    return 1 if errors else 0


def create_result_dir(run_prefix: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = RESULT_DIR / f"{stamp}-{slugify(run_prefix)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def init_live_summary(out_dir: Path) -> Path:
    live_summary = out_dir / "summary.live.csv"
    with live_summary.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore").writeheader()
    return live_summary


def run_profile(profile, out_dir: Path, live_summary: Path, all_rows: list[dict], telemetry_rows: list[dict]) -> dict | None:
    raw_path = out_dir / f"{profile.name}.jsonl"
    err_path = out_dir / f"{profile.name}.stderr.log"
    cmd = profile_command(profile, raw_path)
    print("+ " + " ".join(cmd), flush=True)
    telemetry_rows.append(capture_telemetry(profile.name))
    returncode = stream_profile_rows(profile, cmd, raw_path, err_path, live_summary, all_rows, telemetry_rows)
    telemetry_rows.append(capture_telemetry(profile.name))
    if returncode == 0:
        return None
    return profile_error(profile, err_path, returncode, live_summary)


def stream_profile_rows(profile, cmd: list[str], raw_path: Path, err_path: Path, live_summary: Path, all_rows: list[dict], telemetry_rows: list[dict]) -> int:
    with raw_path.open("w") as raw, err_path.open("w") as err:
        proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=err, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            raw.write(line)
            raw.flush()
            handle_stdout_line(profile, line, live_summary, all_rows, telemetry_rows)
        return proc.wait()


def handle_stdout_line(profile, line: str, live_summary: Path, all_rows: list[dict], telemetry_rows: list[dict]) -> None:
    line = line.strip()
    if not line:
        return
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        print(f"partial non-JSON stdout from {profile.name}: {line[:160]}")
        return
    enrich_row(profile, row)
    all_rows.append(row)
    append_live_summary(row, live_summary)
    telemetry_rows.append(capture_telemetry(profile.name, summary_row(row)))
    print_progress(profile.name, row)


def enrich_row(profile, row: dict) -> None:
    row["profile"] = profile.name
    row["model"] = profile.model
    row["cache_strategy"] = profile.cache_strategy
    row["repetitions"] = profile.repetitions
    row["wall_s"] = infer_wall_s(row, profile.repetitions)
    row.setdefault("flash_attn", profile.flash_attn if is_set(profile.flash_attn) else "")


def infer_wall_s(row: dict, repetitions: str | int | None) -> float | None:
    avg_ns = row.get("avg_ns")
    try:
        avg_ns_f = float(avg_ns)
        reps = int(repetitions) if repetitions not in (None, "", "unset") else 1
        if avg_ns_f > 0 and reps > 0:
            return round(avg_ns_f * reps / 1e9, 3)
    except Exception:
        pass
    return None


def append_live_summary(row: dict, live_summary: Path) -> None:
    with live_summary.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore").writerow(summary_row(row))


def print_progress(profile_name: str, row: dict) -> None:
    out = summary_row(row)
    print(
        f"{profile_name} depth={row.get('n_depth')} {out['test']} "
        f"b={row.get('n_batch')} ub={row.get('n_ubatch')} "
        f"t={row.get('n_threads')} moe={row.get('n_cpu_moe')} "
        f"ngl={row.get('n_gpu_layers')} fa={row.get('flash_attn')} {row.get('avg_ts')} tok/s",
        flush=True,
    )


def format_duration(seconds: float | int) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def profile_error(profile, err_path: Path, returncode: int, live_summary: Path) -> dict:
    stderr_text = err_path.read_text(errors="replace") if err_path.exists() else ""
    print(f"{profile.name} failed; see {err_path}; partial results: {live_summary}")
    return {
        "profile": profile.name,
        "model": profile.model,
        "returncode": returncode,
        "stderr_log": str(err_path),
        "stderr_tail": stderr_text[-8000:],
    }


def write_final_outputs(out_dir: Path, live_summary: Path, all_rows: list[dict], telemetry_rows: list[dict], errors: list[dict], run_meta: dict) -> None:
    summary = out_dir / "summary.csv"
    telemetry = out_dir / "telemetry.json"
    report = out_dir / "index.html"
    write_summary(all_rows, summary)
    telemetry.write_text(json.dumps(telemetry_rows, indent=2))
    write_html_report(all_rows, report, errors, telemetry_rows, run_meta)
    print(f"summary: {summary}")
    print(f"live:    {live_summary}")
    print(f"telemetry: {telemetry}")
    print(f"report:    {report}")
    print(f"total time: {format_duration(run_meta.get('elapsed_seconds', 0))}")
    print(f"report URL path: /{out_dir.name}/")

from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import model_label, model_metadata
from .paths import D3_SCRIPT, HTML_REPORT_TEMPLATE


def parse_rows(text: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_name(row: dict) -> str:
    if int(row.get("n_prompt") or 0) > 0:
        return f"pp{row.get('n_prompt')}"
    if int(row.get("n_gen") or 0) > 0:
        return f"tg{row.get('n_gen')}"
    return "unknown"


def report_rows(rows: list[dict]) -> list[dict]:
    return [_with_model_offload_info(row) for row in rows]


def _with_model_offload_info(row: dict) -> dict:
    out = dict(row)
    model_path = str(out.get("model_path") or "").strip()
    model = str(out.get("model") or out.get("model_filename") or model_path or "").strip()
    if model_path:
        out["model_path"] = model_path
    if model_path or model:
        label_source = model_path or model
        out["model_display"] = model_label(label_source)
        out["model"] = out["model_display"]
    try:
        meta = model_metadata(str(model_path or model))
        if meta.get("block_count"):
            layers = int(meta["block_count"])
            out["model_layers"] = layers
            if "n_cpu_moe" in out and str(out.get("n_cpu_moe", "")).strip():
                cpu_moe = max(0, min(layers, int(out["n_cpu_moe"])))
                out["cpu_moe_layers"] = cpu_moe
                out["remaining_moe_layers"] = max(0, layers - cpu_moe)
    except Exception:
        pass
    return out


def write_html_report(
    rows: list[dict],
    path: Path,
    errors: list[dict] | None = None,
    telemetry: list[dict] | None = None,
    run_meta: dict | None = None,
) -> None:
    html = render_html_report(rows, errors or [], telemetry or [], run_meta or {})
    path.write_text(html)
    copy_report_assets(path.parent)


def copy_report_assets(report_dir: Path) -> None:
    assets = report_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / D3_SCRIPT.name).write_bytes(D3_SCRIPT.read_bytes())


def render_html_report(rows: list[dict], errors: list[dict], telemetry: list[dict] | None = None, run_meta: dict | None = None) -> str:
    template = HTML_REPORT_TEMPLATE.read_text()
    return (
        template
        .replace("__ROWS_JSON__", json.dumps(report_rows(rows)))
        .replace("__ERRORS_JSON__", json.dumps(errors))
        .replace("__TELEMETRY_JSON__", json.dumps(telemetry or []))
        .replace("__RUN_META_JSON__", json.dumps(run_meta or {}))
    )


SUMMARY_FIELDS = [
    "profile", "model", "model_path", "cache_strategy", "test", "n_depth", "n_prompt", "n_gen",
    "n_batch", "n_ubatch", "n_threads", "n_cpu_moe", "n_gpu_layers", "flash_attn",
    "model_layers", "cpu_moe_layers", "remaining_moe_layers",
    "type_k", "type_v", "repetitions", "avg_ts", "stddev_ts", "avg_ns", "wall_s",
]


def summary_row(row: dict) -> dict:
    out = report_rows([row])[0]
    out["test"] = test_name(row)
    return out


def write_summary(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(summary_row(row))

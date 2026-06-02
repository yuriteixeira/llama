from __future__ import annotations

import subprocess
import time
import urllib.request

NVIDIA_SMI_QUERY = [
    "timestamp",
    "name",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
]

GPU_NUMERIC_FIELDS = {
    "temperature_gpu": "gpu_temp_c",
    "power_draw": "gpu_power_w",
    "utilization_gpu": "gpu_util_pct",
    "utilization_memory": "gpu_mem_util_pct",
    "memory_used": "gpu_mem_used_mib",
    "memory_total": "gpu_mem_total_mib",
}

PROMETHEUS_METRICS_URL = "http://127.0.0.1:8080/metrics"


def capture_telemetry(profile_name: str, row: dict | None = None) -> dict:
    sample = {
        "sample_ts": time.time(),
        "profile": profile_name,
    }
    if row:
        sample.update({
            "test": row.get("test"),
            "n_depth": row.get("n_depth"),
            "n_prompt": row.get("n_prompt"),
            "n_gen": row.get("n_gen"),
        })
    sample.update(capture_nvidia_smi())
    sample.update(capture_llama_metrics())
    return sample


def capture_nvidia_smi() -> dict:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(NVIDIA_SMI_QUERY)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=3, check=False)
    except Exception as exc:
        return {"gpu_error": repr(exc)}
    if result.returncode != 0 or not result.stdout.strip():
        return {"gpu_error": result.stderr.strip() or f"nvidia-smi exit {result.returncode}"}
    return parse_nvidia_smi_row(result.stdout.splitlines()[0])


def parse_nvidia_smi_row(line: str) -> dict:
    values = [part.strip() for part in line.split(",")]
    raw = dict(zip(NVIDIA_SMI_QUERY, values))
    out = {"gpu_timestamp": raw.get("timestamp", ""), "gpu_name": raw.get("name", "")}
    for source, dest in GPU_NUMERIC_FIELDS.items():
        out[dest] = parse_float(raw.get(source.replace("_", "."), ""))
    return out


def capture_llama_metrics() -> dict:
    try:
        with urllib.request.urlopen(PROMETHEUS_METRICS_URL, timeout=3) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {"metrics_error": repr(exc)}
    return parse_prometheus_metrics(text)


def parse_prometheus_metrics(text: str) -> dict:
    out: dict[str, float] = {}
    for line in text.splitlines():
        parsed = parse_prometheus_line(line)
        if parsed:
            key, value = parsed
            out[f"metric_{key}"] = value
    return out


def parse_prometheus_line(line: str) -> tuple[str, float] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.rsplit(None, 1)
    if len(parts) != 2:
        return None
    name = parts[0].split("{", 1)[0]
    value = parse_float(parts[1])
    if value is None:
        return None
    return sanitize_metric_name(name), value


def sanitize_metric_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None

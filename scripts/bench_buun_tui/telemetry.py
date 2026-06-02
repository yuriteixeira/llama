from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path

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
_PREV_CPU_SAMPLE: tuple[float, float] | None = None


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
    sample.update(capture_system_telemetry())
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


def capture_system_telemetry() -> dict:
    out: dict[str, float | str | None] = {}
    out.update(capture_cpu_usage())
    out.update(capture_cpu_temperature())
    out.update(capture_ram_usage())
    return out


def capture_cpu_usage() -> dict:
    global _PREV_CPU_SAMPLE
    sample = read_cpu_sample()
    if sample is None:
        return {"cpu_error": "failed to read /proc/stat"}
    total, idle = sample
    if _PREV_CPU_SAMPLE is None:
        _PREV_CPU_SAMPLE = sample
        return {}
    prev_total, prev_idle = _PREV_CPU_SAMPLE
    _PREV_CPU_SAMPLE = sample
    total_delta = total - prev_total
    idle_delta = idle - prev_idle
    if total_delta <= 0:
        return {}
    return {"cpu_util_pct": max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))}


def read_cpu_sample() -> tuple[float, float] | None:
    try:
        line = Path("/proc/stat").read_text().splitlines()[0]
    except Exception:
        return None
    parts = line.split()
    if not parts or parts[0] != "cpu":
        return None
    values = [float(v) for v in parts[1:]]
    if len(values) < 5:
        return None
    idle = values[3] + values[4]
    return sum(values), idle


def capture_cpu_temperature() -> dict:
    temp = read_cpu_temperature()
    return {"cpu_temp_c": temp} if temp is not None else {}


def read_cpu_temperature() -> float | None:
    candidates: list[tuple[int, Path]] = []
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            name = (hwmon / "name").read_text().strip().lower()
        except Exception:
            name = ""
        priority = 1 if name in {"k10temp", "coretemp"} else 5
        for temp_input in hwmon.glob("temp*_input"):
            label_path = temp_input.with_name(temp_input.name.replace("_input", "_label"))
            try:
                label = label_path.read_text().strip().lower()
            except Exception:
                label = ""
            label_priority = 0 if label in {"tctl", "tdie", "package id 0", "cpu"} else priority
            candidates.append((label_priority, temp_input))
    for _, path in sorted(candidates, key=lambda item: item[0]):
        try:
            value = float(path.read_text().strip()) / 1000.0
        except Exception:
            continue
        if 0.0 < value < 130.0:
            return value
    return None


def capture_ram_usage() -> dict:
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except Exception as exc:
        return {"ram_error": repr(exc)}
    values: dict[str, float] = {}
    for line in meminfo.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            values[parts[0].rstrip(":")] = parse_float(parts[1]) or 0.0
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return {}
    used = max(0.0, total - available)
    return {
        "ram_total_mib": total / 1024.0,
        "ram_used_mib": used / 1024.0,
        "ram_used_pct": 100.0 * used / total,
    }


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

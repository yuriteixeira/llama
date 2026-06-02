from __future__ import annotations

import configparser
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import find_models, model_metadata, model_path
from .paths import CONFIG_DIR, ROOT
from .profile import CACHE_STRATEGIES, Profile

BEST_KNOWN_PARAMS = CONFIG_DIR / "best-known-params.ini"
SCHEMA_VERSION = "1"
APPLICABLE_FIELDS = {key for key, _ in Profile("profile").items()} - {"model", "depths"}
BOOLEAN_FIELDS = {"warmup", "flash_attn", "no_mmap"}
NUMERIC_LIST_FIELDS = {
    "depths",
    "batch_size",
    "ubatch_size",
    "threads",
    "n_cpu_moe",
    "gpu_layers",
    "prompt",
    "gen",
    "repetitions",
}
RESEARCH_FIELDS = {
    "model_name",
    "architecture",
    "quantization",
    "size_bytes",
    "is_mtp",
    "is_reap",
    "supports_moe",
    "active_params",
    "research_summary",
    "confidence",
    "references",
    "depths",
}


@dataclass(frozen=True)
class Recommendation:
    section: str
    model: str
    values: dict[str, str]
    metadata: dict[str, str]


@dataclass(frozen=True)
class ApplyPlan:
    applied: dict[str, str]
    skipped: dict[str, str]
    invalid: dict[str, str]


@dataclass(frozen=True)
class ResearchRun:
    path: Path
    command: str
    response: str


class ResearchError(RuntimeError):
    def __init__(self, message: str, command: str, response: str, returncode: int | None = None):
        super().__init__(message)
        self.command = command
        self.response = response
        self.returncode = returncode


def load_research_meta(path: Path = BEST_KNOWN_PARAMS) -> dict[str, str]:
    if not path.exists():
        return {}
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(path)
    return dict(cp["meta"]) if "meta" in cp else {}


def load_recommendations(path: Path = BEST_KNOWN_PARAMS) -> list[Recommendation]:
    if not path.exists():
        return []
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(path)
    recs: list[Recommendation] = []
    for section in cp.sections():
        if not section.startswith("model:"):
            continue
        model = section.removeprefix("model:")
        raw = dict(cp[section])
        values = {k: v for k, v in raw.items() if k in APPLICABLE_FIELDS}
        metadata = {k: v for k, v in raw.items() if k not in APPLICABLE_FIELDS}
        recs.append(Recommendation(section, model, values, metadata))
    return recs


def find_recommendation(model: str, path: Path = BEST_KNOWN_PARAMS) -> Recommendation | None:
    matches = find_recommendations(model, path)
    return matches[0] if matches else None


def find_recommendations(model: str, path: Path = BEST_KNOWN_PARAMS) -> list[Recommendation]:
    recs = load_recommendations(path)
    if not model:
        return []
    meta = model_metadata(model)
    basename = Path(model).name
    stem = Path(model).stem.lower()
    name = str(meta.get("name") or "").lower()

    exact = [r for r in recs if r.model == model]
    if exact:
        return exact
    by_basename = [r for r in recs if Path(r.model).name == basename]
    if by_basename:
        return by_basename
    by_name = [r for r in recs if name and str(r.metadata.get("model_name", "")).lower() == name]
    if by_name:
        return by_name
    fuzzy = [r for r in recs if stem and (stem in Path(r.model).stem.lower() or Path(r.model).stem.lower() in stem)]
    return fuzzy


def build_apply_plan(rec: Recommendation) -> ApplyPlan:
    applied: dict[str, str] = {}
    skipped: dict[str, str] = {}
    invalid: dict[str, str] = {}
    for key, value in rec.values.items():
        reason = validate_field(key, value)
        if reason:
            invalid[key] = f"{value} ({reason})"
        else:
            applied[key] = value
    for key, value in rec.metadata.items():
        if key not in RESEARCH_FIELDS and key != "model":
            skipped[key] = value
    return ApplyPlan(applied, skipped, invalid)


def apply_recommendation(profile: Profile, rec: Recommendation) -> ApplyPlan:
    plan = build_apply_plan(rec)
    for key, value in plan.applied.items():
        setattr(profile, key, value)
    return plan


def validate_field(key: str, value: str) -> str:
    value = value.strip()
    if not value:
        return "empty value"
    if key == "cache_strategy" and value not in CACHE_STRATEGIES:
        return f"unknown cache_strategy; expected one of {', '.join(CACHE_STRATEGIES)}"
    if key in BOOLEAN_FIELDS and value.lower() not in {"0", "1", "true", "false", "yes", "no", "on", "off", "unset", "none", "default"}:
        return "not a boolean/unset value"
    if key in NUMERIC_LIST_FIELDS and value.lower() not in {"unset", "none", "default"}:
        parts = [p.strip() for p in value.split(",")]
        if not parts or any(not p.isdigit() for p in parts):
            return "expected integer or comma-separated integers"
    return ""


def generate_best_known_params(path: Path = BEST_KNOWN_PARAMS) -> Path:
    return run_best_known_research(path).path


def run_best_known_research(path: Path = BEST_KNOWN_PARAMS, on_output: Callable[[str], None] | None = None, stream: bool = False) -> ResearchRun:
    models = find_models()
    prompt = build_research_prompt(models)
    cmd = ["pi", "--mode", "json", prompt] if stream else ["pi", "-p", prompt]
    command = shlex.join(cmd)
    output = run_research_command_json(cmd, command, on_output) if stream else run_research_command(cmd, command, on_output)
    try:
        cp = parse_research_output(output)
    except Exception as exc:
        raise ResearchError(f"Failed to parse pi response as INI: {exc}", command, output) from exc
    cp["meta"] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": command,
        "response": output,
        "schema_version": SCHEMA_VERSION,
    }
    ensure_sections_for_models(cp, models)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        cp.write(f)
    return ResearchRun(path=path, command=command, response=output)


def run_research_command(cmd: list[str], command: str, on_output: Callable[[str], None] | None = None) -> str:
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    except Exception as exc:
        raise ResearchError(f"Failed to start pi research: {exc}", command, "") from exc
    output_parts: list[str] = []
    assert proc.stdout is not None
    for chunk in proc.stdout:
        output_parts.append(chunk)
        if on_output:
            on_output(chunk)
    returncode = proc.wait()
    output = "".join(output_parts)
    if returncode != 0:
        raise ResearchError(f"pi research exited with status {returncode}", command, output, returncode)
    return output


def run_research_command_json(cmd: list[str], command: str, on_output: Callable[[str], None] | None = None) -> str:
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    except Exception as exc:
        raise ResearchError(f"Failed to start pi research: {exc}", command, "") from exc
    output_parts: list[str] = []
    raw_parts: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        raw_parts.append(line)
        delta = extract_json_stream_delta(line)
        if delta:
            output_parts.append(delta)
            if on_output:
                on_output(delta)
        elif on_output and not line.lstrip().startswith("{"):
            on_output(line)
    returncode = proc.wait()
    output = "".join(output_parts)
    raw_output = "".join(raw_parts)
    if returncode != 0:
        raise ResearchError(f"pi research exited with status {returncode}", command, output or raw_output, returncode)
    if not output.strip():
        raise ResearchError("pi JSON stream completed without assistant text output", command, raw_output, returncode)
    return output


def extract_json_stream_delta(line: str) -> str:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if event.get("type") != "message_update":
        return ""
    assistant_event = event.get("assistantMessageEvent") or {}
    if assistant_event.get("type") == "text_delta":
        return str(assistant_event.get("delta") or "")
    return ""


def build_research_prompt(models: list[str]) -> str:
    model_blocks = []
    for model in models:
        p = model_path(model)
        meta = model_metadata(model)
        model_blocks.append(
            "\n".join(
                [
                    f"relative_path={model}",
                    f"filename={Path(model).name}",
                    f"size_bytes={p.stat().st_size if p.exists() else 0}",
                    f"metadata={meta}",
                    f"quantization_hint={quantization_hint(model)}",
                ]
            )
        )
    profile_fields = ", ".join(key for key in APPLICABLE_FIELDS)
    caches = ", ".join(CACHE_STRATEGIES)
    return f"""
Research best-known llama.cpp/buun benchmark and llama-server parameters for these local GGUF models on this hardware: NVIDIA RTX 3070 8GB VRAM, about 12 CPU threads, 16GB+ RAM.

Return ONLY an INI document. Do not include prose outside the INI. Use one section per model named exactly [model:<relative_path>]. Include research metadata and recommendations.

Allowed auto-applied profile fields: {profile_fields}
Allowed cache_strategy values: {caches}
Do not include depths as a recommended parameter; depths is only for benchmark sweeps and must not be changed by best-known params.
Recommend one best-known value per field, not a list of possible values. Use a single integer for numeric fields. Use unset when a parameter should not be passed.
Include research_summary, confidence, references, architecture, quantization, is_mtp, is_reap, supports_moe, active_params when knowable.

Models:

{'\n\n---\n\n'.join(model_blocks)}
""".strip()


def parse_research_output(output: str) -> configparser.ConfigParser:
    text = extract_ini(output)
    cp = configparser.ConfigParser(interpolation=None)
    cp.read_string(text)
    return cp


def extract_ini(output: str) -> str:
    stripped = output.strip()
    fenced = re.search(r"```(?:ini)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    first = stripped.find("[model:")
    if first == -1:
        first = stripped.find("[meta]")
    return stripped[first:].strip() if first >= 0 else stripped


def ensure_sections_for_models(cp: configparser.ConfigParser, models: list[str]) -> None:
    for model in models:
        section = f"model:{model}"
        if section not in cp:
            cp[section] = fallback_recommendation(model)


def fallback_recommendation(model: str) -> dict[str, str]:
    meta = model_metadata(model)
    return {
        "model_name": str(meta.get("name") or Path(model).stem),
        "architecture": str(meta.get("architecture") or "?"),
        "quantization": quantization_hint(model),
        "size_bytes": str(model_path(model).stat().st_size if model_path(model).exists() else 0),
        "research_summary": "No pi research recommendation was returned for this model; using conservative TUI defaults.",
        "confidence": "low",
        "cache_strategy": "turbo",
        "depths": "0,4096,8192,16384,32768",
        "batch_size": "1024",
        "ubatch_size": "256",
        "threads": "12",
        "n_cpu_moe": "33" if is_probably_moe(model, meta) else "unset",
        "gpu_layers": "unset",
        "prompt": "512",
        "gen": "128",
        "repetitions": "1",
        "warmup": "false",
        "device": "CUDA0",
        "flash_attn": "1",
        "no_mmap": "unset",
        "references": "fallback defaults",
    }


def quantization_hint(model: str) -> str:
    name = Path(model).name.upper()
    patterns = [r"UD-IQ\d_[A-Z0-9_]+", r"IQ\d_[A-Z0-9_]+", r"Q\d_K_[A-Z]+", r"Q\d_[0-9A-Z_]+", r"F16", r"BF16"]
    for pattern in patterns:
        m = re.search(pattern, name)
        if m:
            return m.group(0)
    return "unknown"


def is_probably_moe(model: str, meta: dict) -> bool:
    haystack = f"{model} {meta.get('name', '')} {meta.get('architecture', '')}".lower()
    return any(token in haystack for token in ["moe", "a3b", "mixtral", "qwen3moe"])

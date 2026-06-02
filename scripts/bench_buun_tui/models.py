from __future__ import annotations

import re
import sys
from pathlib import Path

from .paths import GGUF_PY, ROOT
from .profile import PRESETS


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def human_size(path: Path) -> str:
    size = path.stat().st_size if path.exists() else 0
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit in {"B", "KB", "MB"} else f"{size:.1f}{unit}"
        size /= 1024
    return "?"


def find_models() -> list[str]:
    models = sorted(rel(p) for p in (ROOT / "models").glob("**/*.gguf"))
    return models or [""]


def model_path(model: str) -> Path:
    p = Path(model)
    return p if p.is_absolute() else ROOT / p


def model_label(model: str) -> str:
    if not model:
        return "No model selected"
    p = model_path(model)
    return f"{Path(model).name} {model_detail_brackets(model)} ({human_size(p)})"


def model_detail_brackets(model: str) -> str:
    parts = model_detail_parts(model)
    return f"[{', '.join(parts)}]" if parts else "[unknown]"


def model_detail_parts(model: str) -> list[str]:
    meta = model_metadata(model) if model else {}
    size_parts = model_size_parts(model, meta)
    type_parts = model_type_parts(model, meta)
    return size_parts + type_parts


def model_size_parts(model: str, meta: dict | None = None) -> list[str]:
    meta = meta or {}
    total = first_total_size_label(model, meta)
    active = first_active_size_label(model, meta)
    parts: list[str] = []
    if total:
        parts.append(f"{total} total" if active else total)
    elif meta.get("parameter_count"):
        parts.append(format_param_count(meta["parameter_count"]))
    if active and active != total:
        parts.append(f"{active} active")
    return parts


def model_type_parts(model: str, meta: dict | None = None) -> list[str]:
    meta = meta or {}
    haystack = model_haystack(model, meta)
    parts: list[str] = []
    if "reap" in haystack:
        parts.append("REAP")
    if re.search(r"\bmtp\b|multi[-_ ]token|speculative", haystack):
        parts.append("MTP")
    if looks_moe(model, meta):
        parts.append("MoE")
    if not parts:
        parts.append("Dense" if model else "Unknown")
    return parts


def model_haystack(model: str, meta: dict | None = None) -> str:
    meta = meta or {}
    return " ".join(str(x) for x in [model, meta.get("name", ""), meta.get("architecture", ""), meta.get("size_label", "")]).lower()


def looks_moe(model: str, meta: dict | None = None) -> bool:
    meta = meta or {}
    if meta.get("expert_count") or meta.get("expert_used_count"):
        return True
    haystack = model_haystack(model, meta)
    return bool(re.search(r"(?<![a-z0-9])a\d+(?:\.\d+)?b\b", haystack) or any(token in haystack for token in ["moe", "mixtral", "qwen3moe", "qwen35moe"]))


def first_total_size_label(model: str, meta: dict | None = None) -> str:
    meta = meta or {}
    for text in [Path(model).stem, str(meta.get("name") or ""), str(meta.get("size_label") or "")]:
        label = extract_total_size_label(text)
        if label:
            return label
    return ""


def first_active_size_label(model: str, meta: dict | None = None) -> str:
    meta = meta or {}
    for text in [Path(model).stem, str(meta.get("name") or ""), str(meta.get("size_label") or "")]:
        label = extract_active_size_label(text)
        if label:
            return label
    return ""


def extract_total_size_label(text: str) -> str:
    match = re.search(r"(?<![A-Za-z])(?P<size>\d+(?:\.\d+)?)\s*[Bb]\b", text)
    return normalize_b_label(match.group("size")) if match else ""


def extract_active_size_label(text: str) -> str:
    match = re.search(r"(?<![A-Za-z0-9])A(?P<size>\d+(?:\.\d+)?)\s*[Bb]\b", text)
    return f"A{normalize_b_label(match.group('size'))}" if match else ""


def normalize_b_label(value: str) -> str:
    number = float(value)
    shown = str(int(number)) if number.is_integer() else value.rstrip("0").rstrip(".")
    return f"{shown}B"


def format_param_count(value: object) -> str:
    try:
        count = float(value)
    except (TypeError, ValueError):
        return str(value)
    for suffix, divisor in [("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000)]:
        if count >= divisor:
            amount = count / divisor
            shown = f"{amount:.1f}".rstrip("0").rstrip(".")
            return f"{shown}{suffix}"
    return str(int(count))


def model_options(models: list[str]) -> tuple[list[str], dict[str, str]]:
    mapping: dict[str, str] = {}
    options: list[str] = []
    for model in models:
        label = model_label(model)
        if label in mapping:
            label = f"{model} {model_detail_brackets(model)} ({human_size(model_path(model))})"
        mapping[label] = model
        options.append(label)
    return options, mapping


def gguf_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        if value and all(isinstance(x, int) and 0 <= x < 256 for x in value):
            try:
                return bytes(value).decode("utf-8")
            except UnicodeDecodeError:
                return value
    return value


META_CACHE: dict[str, dict] = {}


def model_metadata(model: str) -> dict:
    if model in META_CACHE:
        return META_CACHE[model]
    meta = {
        "architecture": "?",
        "name": Path(model).stem,
        "block_count": None,
        "size_label": "",
        "parameter_count": None,
        "expert_count": None,
        "expert_used_count": None,
    }
    try:
        if str(GGUF_PY) not in sys.path:
            sys.path.insert(0, str(GGUF_PY))
        from gguf import GGUFReader  # type: ignore

        reader = GGUFReader(model_path(model), "r")
        for key, field in reader.fields.items():
            val = gguf_value(field.parts[-1])
            if key == "general.architecture":
                meta["architecture"] = val
            elif key == "general.name":
                meta["name"] = val
            elif key == "general.size_label":
                meta["size_label"] = val
            elif key == "general.parameter_count":
                meta["parameter_count"] = int(val)
            elif key.endswith(".block_count"):
                meta["block_count"] = int(val)
            elif key.endswith(".expert_count"):
                meta["expert_count"] = int(val)
            elif key.endswith(".expert_used_count"):
                meta["expert_used_count"] = int(val)
    except Exception as exc:
        meta["error"] = repr(exc)
    META_CACHE[model] = meta
    return meta


def model_description(model: str) -> str:
    meta = model_metadata(model)
    parts = [f"Selected: {model_label(model)}"]
    if meta.get("name"):
        parts.append(f"name={meta['name']}")
    if meta.get("architecture"):
        parts.append(f"arch={meta['architecture']}")
    if meta.get("block_count"):
        parts.append(f"layers/block_count={meta['block_count']}")
    if meta.get("error"):
        parts.append(f"metadata read failed: {meta['error']}")
    return "; ".join(parts)


def gpu_layer_options(model: str) -> list[str]:
    n = model_metadata(model).get("block_count")
    if not isinstance(n, int) or n <= 0:
        return PRESETS["gpu_layers"]
    vals = []
    for ratio in [1.0, 0.8, 0.6, 0.4]:
        v = max(1, round(n * ratio))
        if v not in vals:
            vals.append(v)
    return [str(v) for v in vals] + [",".join(str(v) for v in vals[:3]), "custom", "unset"]


def choose_default_model(models: list[str]) -> str:
    for needle in ["IQ4_NL", "Q4", "IQ2_M"]:
        for model in models:
            if needle in model:
                return model
    return models[0]

from __future__ import annotations

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
    p = model_path(model)
    return f"{Path(model).name} ({human_size(p)})"


def model_options(models: list[str]) -> tuple[list[str], dict[str, str]]:
    mapping: dict[str, str] = {}
    options: list[str] = []
    for model in models:
        label = model_label(model)
        if label in mapping:
            label = f"{model} ({human_size(model_path(model))})"
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
    meta = {"architecture": "?", "name": Path(model).stem, "block_count": None}
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
            elif key.endswith(".block_count"):
                meta["block_count"] = int(val)
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

from __future__ import annotations

from dataclasses import dataclass

from .profile import Profile


_LOAD_ERROR_TOKENS = [
    "failed to load model",
    "load model failed",
    "error loading model",
    "unknown model architecture",
    "unsupported architecture",
    "not supported",
    "invalid model",
    "invalid gguf",
    "mmap failed",
    "failed to initialize tokenizer",
    "tokenizer error",
    "model load failed",
    "unable to load",
]


@dataclass(frozen=True)
class RepairStep:
    gpu_layers: str
    n_cpu_moe: str
    label: str


def diagnose_failure(output: str) -> str:
    text = output.lower()
    if any(token in text for token in [
        "timed out",
    ]):
        return "timeout"
    if any(token in text for token in [
        "cudamalloc failed: out of memory",
        "failed to allocate cuda",
        "failed to allocate compute",
        "failed to create context",
        "failed to initialize the context",
        "graph_reserve: failed",
        "out of memory",
        "cuda error",
    ]):
        return "gpu_oom"
    if any(token in text for token in [
        "cannot allocate memory",
        "std::bad_alloc",
        "bad_alloc",
        "malloc failed",
        "memory allocation failed",
    ]):
        return "host_oom"
    if any(token in text for token in _LOAD_ERROR_TOKENS):
        return "load_error"
    return "unknown"


def next_repair_step(
    profile: Profile,
    block_count: int,
    supports_moe: bool | None,
    diagnosis: str,
    tried: set[tuple[str, str]],
) -> RepairStep | None:
    ngl = int_or_default(profile.gpu_layers, block_count + 1)
    moe_current = profile.n_cpu_moe
    moe = None if moe_current == "unset" else int_or_default(moe_current, 0)

    def candidate(gpu_layers: int, moe_value: int | None, label: str) -> RepairStep | None:
        key = (str(gpu_layers), "unset" if moe_value is None else str(moe_value))
        if key in tried:
            return None
        return RepairStep(key[0], key[1], label)

    ordered: list[RepairStep | None] = []
    if diagnosis in {"gpu_oom", "timeout"}:
        if supports_moe is not False and moe is not None and moe < block_count:
            ordered.append(candidate(ngl, moe + 1, "gpu OOM/timeout: increase n_cpu_moe by 1"))
        if ngl > 0:
            ordered.append(candidate(max(0, ngl - 1), moe if moe is not None else None, "gpu OOM/timeout: lower gpu_layers by 1"))
    elif diagnosis == "host_oom":
        if supports_moe is not False and moe is not None and moe > 0:
            ordered.append(candidate(ngl, moe - 1, "RAM OOM: decrease n_cpu_moe by 1"))
        if ngl < block_count + 1:
            ordered.append(candidate(ngl + 1, moe if moe is not None else None, "RAM OOM: increase gpu_layers by 1"))
    elif diagnosis == "load_error":
        return None
    else:
        if supports_moe is not False and moe is not None and moe < block_count:
            ordered.append(candidate(ngl, moe + 1, "unknown failure: increase n_cpu_moe by 1"))
        if ngl > 0:
            ordered.append(candidate(max(0, ngl - 1), moe if moe is not None else None, "unknown failure: lower gpu_layers by 1"))

    for step in ordered:
        if step is not None:
            return step
    return None


def int_or_default(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

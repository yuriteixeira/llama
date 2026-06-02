from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .models import gguf_value, model_path
from .paths import GGUF_PY
from .profile import Profile

MIB = 1024 * 1024
GIB = 1024 * MIB

# Approximate bytes per KV element. Turbo values are bit-width based; q8_0 is
# rounded up for quantization block metadata.
KV_BYTES_BY_STRATEGY = {
    "turbo": (0.50, 0.375),
    "q8": (1.125, 1.125),
    "f16": (2.0, 2.0),
    "unset": (2.0, 2.0),
}

EXPERT_TENSOR_MARKERS = ("_exps", "_shexp", ".expert", "expert_")


class OffloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class LayerSize:
    total_bytes: int = 0
    expert_bytes: int = 0


@dataclass(frozen=True)
class ModelSizing:
    model: str
    name: str
    architecture: str
    block_count: int
    context_length: int | None
    embedding_length: int | None
    key_length: int | None
    value_length: int | None
    supports_moe: bool | None
    expert_count: int | None
    expert_used_count: int | None
    file_size_bytes: int
    non_layer_bytes: int
    output_bytes: int
    layers: list[LayerSize]
    exact_tensor_sizes: bool
    notes: list[str]


@dataclass(frozen=True)
class RuntimeBudget:
    desired_context: int
    vram_total_bytes: int
    vram_free_bytes: int | None
    ram_available_bytes: int
    gpu_workspace_bytes: int
    cpu_runtime_bytes: int
    kv_gpu_bytes_per_layer: int
    kv_cpu_bytes_per_layer: int


@dataclass(frozen=True)
class Candidate:
    gpu_layers: int
    n_cpu_moe: int
    gpu_weight_bytes: int
    cpu_weight_bytes: int
    gpu_kv_bytes: int
    cpu_kv_bytes: int
    gpu_total_bytes: int
    cpu_total_bytes: int
    gpu_over_bytes: int
    cpu_over_bytes: int

    @property
    def fits(self) -> bool:
        return self.gpu_over_bytes <= 0 and self.cpu_over_bytes <= 0


@dataclass(frozen=True)
class OffloadRecommendation:
    gpu_layers: str
    n_cpu_moe: str
    confidence: str
    model: ModelSizing
    budget: RuntimeBudget
    candidate: Candidate
    explanations: list[str]


def calculate_offload(profile: Profile, desired_context: int | None = None) -> OffloadRecommendation:
    if not profile.model:
        raise OffloadError("Select a model before auto-calculating offload.")
    sizing = read_model_sizing(profile.model)
    if sizing.block_count <= 0:
        raise OffloadError("Could not read a valid GGUF block/layer count for this model.")
    budget = runtime_budget(profile, sizing, desired_context)
    candidates = build_candidates(sizing, budget)
    if not candidates:
        raise OffloadError("Could not build any offload candidates for this model.")
    best = choose_candidate(candidates)
    confidence = recommendation_confidence(sizing, best)
    n_cpu_moe = "unset" if sizing.supports_moe is False else str(best.n_cpu_moe)
    return OffloadRecommendation(
        gpu_layers=str(best.gpu_layers),
        n_cpu_moe=n_cpu_moe,
        confidence=confidence,
        model=sizing,
        budget=budget,
        candidate=best,
        explanations=recommendation_explanations(sizing, budget, best),
    )


def read_model_sizing(model: str) -> ModelSizing:
    p = model_path(model)
    if not p.exists():
        raise OffloadError(f"Model file not found: {model}")
    try:
        reader = open_gguf_reader(p)
        fields = {key: gguf_value(field.parts[-1]) for key, field in reader.fields.items()}
        arch = str(fields.get("general.architecture") or "?")
        block_count = int(first_field_value(fields, "block_count") or 0)
        if block_count <= 0:
            raise ValueError("missing block_count")
        layers = [LayerSize() for _ in range(block_count)]
        totals = [0 for _ in range(block_count)]
        experts = [0 for _ in range(block_count)]
        non_layer_bytes = 0
        output_bytes = 0
        for tensor in reader.tensors:
            name = str(tensor.name)
            n_bytes = int(tensor.n_bytes)
            match = re.match(r"blk\.(\d+)\.", name)
            if match:
                idx = int(match.group(1))
                if 0 <= idx < block_count:
                    totals[idx] += n_bytes
                    if is_expert_tensor_name(name):
                        experts[idx] += n_bytes
                else:
                    non_layer_bytes += n_bytes
            else:
                non_layer_bytes += n_bytes
                if name == "output.weight":
                    output_bytes += n_bytes
        layers = [LayerSize(total, expert) for total, expert in zip(totals, experts)]
        return ModelSizing(
            model=model,
            name=str(fields.get("general.name") or Path(model).stem),
            architecture=arch,
            block_count=block_count,
            context_length=optional_int(first_field_value(fields, "context_length")),
            embedding_length=optional_int(first_field_value(fields, "embedding_length")),
            key_length=optional_int(first_field_value(fields, "attention.key_length")) or infer_kv_length(fields),
            value_length=optional_int(first_field_value(fields, "attention.value_length")) or infer_kv_length(fields),
            supports_moe=detect_moe(model, fields, any(x.expert_bytes for x in layers)),
            expert_count=optional_int(first_field_value(fields, "expert_count")),
            expert_used_count=optional_int(first_field_value(fields, "expert_used_count")),
            file_size_bytes=p.stat().st_size,
            non_layer_bytes=non_layer_bytes,
            output_bytes=output_bytes,
            layers=layers,
            exact_tensor_sizes=True,
            notes=[],
        )
    except Exception as exc:
        return fallback_model_sizing(model, p, exc)


def open_gguf_reader(path: Path):
    if str(GGUF_PY) not in sys.path:
        sys.path.insert(0, str(GGUF_PY))
    from gguf import GGUFReader  # type: ignore

    return GGUFReader(path, "r")


def fallback_model_sizing(model: str, path: Path, exc: Exception) -> ModelSizing:
    from .models import model_metadata

    meta = model_metadata(model)
    block_count = int(meta.get("block_count") or 0)
    if block_count <= 0:
        raise OffloadError(f"Failed to read GGUF sizing metadata: {exc}") from exc
    file_size = path.stat().st_size
    non_layer = int(file_size * 0.05)
    layer_total = max(1, (file_size - non_layer) // block_count)
    is_moe = detect_moe(model, {"general.architecture": meta.get("architecture"), "general.name": meta.get("name")}, False)
    expert_total = int(layer_total * 0.85) if is_moe else 0
    return ModelSizing(
        model=model,
        name=str(meta.get("name") or Path(model).stem),
        architecture=str(meta.get("architecture") or "?"),
        block_count=block_count,
        context_length=None,
        embedding_length=None,
        key_length=None,
        value_length=None,
        supports_moe=is_moe,
        expert_count=None,
        expert_used_count=None,
        file_size_bytes=file_size,
        non_layer_bytes=non_layer,
        output_bytes=0,
        layers=[LayerSize(layer_total, expert_total) for _ in range(block_count)],
        exact_tensor_sizes=False,
        notes=[f"Tensor sizing fallback used after GGUF read failed: {exc!r}"],
    )


def first_field_value(fields: dict[str, object], suffix: str) -> object | None:
    for key, value in fields.items():
        if key.endswith("." + suffix) or key == suffix:
            return value
    return None


def optional_int(value: object | None) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_kv_length(fields: dict[str, object]) -> int | None:
    embd = optional_int(first_field_value(fields, "embedding_length"))
    heads = optional_int(first_field_value(fields, "attention.head_count"))
    heads_kv = optional_int(first_field_value(fields, "attention.head_count_kv"))
    if embd and heads and heads_kv:
        return max(1, embd // heads * heads_kv)
    return None


def detect_moe(model: str, fields: dict[str, object], has_expert_tensors: bool) -> bool | None:
    if has_expert_tensors:
        return True
    expert_count = optional_int(first_field_value(fields, "expert_count"))
    expert_used = optional_int(first_field_value(fields, "expert_used_count"))
    if expert_count or expert_used:
        return True
    haystack = " ".join(str(x) for x in [model, fields.get("general.name", ""), fields.get("general.architecture", "")]).lower()
    if any(token in haystack for token in ["moe", "mixtral", "qwen3moe", "qwen35moe", "a3b"]):
        return True
    if fields:
        return False
    return None


def is_expert_tensor_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in EXPERT_TENSOR_MARKERS)


def runtime_budget(profile: Profile, sizing: ModelSizing, desired_context_override: int | None = None) -> RuntimeBudget:
    desired_context = desired_context_override if desired_context_override and desired_context_override > 0 else desired_context_size(profile, sizing)
    k_bytes, v_bytes = KV_BYTES_BY_STRATEGY.get(profile.cache_strategy, KV_BYTES_BY_STRATEGY["unset"])
    key_len = sizing.key_length or infer_length_from_embedding(sizing)
    value_len = sizing.value_length or infer_length_from_embedding(sizing)
    kv_per_layer = int(desired_context * (key_len * k_bytes + value_len * v_bytes))
    gpu_workspace = estimate_gpu_workspace(profile, sizing)
    cpu_runtime = estimate_cpu_runtime(profile, sizing)
    vram_total, vram_free = query_vram_bytes()
    ram_available = query_mem_available_bytes()
    return RuntimeBudget(
        desired_context=desired_context,
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        ram_available_bytes=ram_available,
        gpu_workspace_bytes=gpu_workspace,
        cpu_runtime_bytes=cpu_runtime,
        kv_gpu_bytes_per_layer=kv_per_layer,
        kv_cpu_bytes_per_layer=kv_per_layer,
    )


def desired_context_size(profile: Profile, sizing: ModelSizing) -> int:
    depth_values = parse_int_list(profile.depths)
    if depth_values:
        return max(depth_values)
    env_ctx = optional_int(os.environ.get("CTX_SIZE"))
    if env_ctx and env_ctx > 0:
        return env_ctx
    if sizing.context_length and sizing.context_length > 0:
        return sizing.context_length
    return 262144


def infer_length_from_embedding(sizing: ModelSizing) -> int:
    # Conservative fallback for GQA metadata gaps.
    if sizing.embedding_length:
        return max(1, sizing.embedding_length // 4)
    return 1024


def parse_int_list(value: str) -> list[int]:
    if not value or value.strip().lower() in {"unset", "none", "default"}:
        return []
    vals: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            vals.append(int(part))
    return vals


def first_int(value: str, default: int) -> int:
    vals = parse_int_list(value)
    return vals[0] if vals else default


def estimate_gpu_workspace(profile: Profile, sizing: ModelSizing) -> int:
    embd = sizing.embedding_length or 4096
    batch = first_int(profile.batch_size, 1024)
    ubatch = first_int(profile.ubatch_size, 256)
    activation_scratch = max(batch, ubatch) * embd * 2 * 8
    graph_workspace = 256 * MIB
    cuda_workspace = 256 * MIB
    return int(graph_workspace + cuda_workspace + activation_scratch)


def estimate_cpu_runtime(profile: Profile, sizing: ModelSizing) -> int:
    embd = sizing.embedding_length or 4096
    ubatch = first_int(profile.ubatch_size, 256)
    scratch = ubatch * embd * 4 * 4
    server_runtime = 128 * MIB
    return int(server_runtime + scratch)


def query_vram_bytes() -> tuple[int, int | None]:
    fallback = 8192 * MIB
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except Exception:
        return fallback, None
    if proc.returncode != 0 or not proc.stdout.strip():
        return fallback, None
    first = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    try:
        total = int(parts[0]) * MIB
        free = int(parts[1]) * MIB if len(parts) > 1 else None
        return total, free
    except (TypeError, ValueError):
        return fallback, None


def query_mem_available_bytes() -> int:
    fallback = 16 * GIB
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except OSError:
        return fallback
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
    if not match:
        return fallback
    return int(match.group(1)) * 1024


def build_candidates(sizing: ModelSizing, budget: RuntimeBudget) -> list[Candidate]:
    max_ngl = sizing.block_count + 1
    if sizing.supports_moe is False:
        moe_values = [0]
    else:
        moe_values = list(range(0, sizing.block_count + 1))
    candidates: list[Candidate] = []
    for gpu_layers in range(max_ngl, -1, -1):
        for n_cpu_moe in moe_values:
            candidates.append(evaluate_candidate(sizing, budget, gpu_layers, n_cpu_moe))
    return candidates


def evaluate_candidate(sizing: ModelSizing, budget: RuntimeBudget, gpu_layers: int, n_cpu_moe: int) -> Candidate:
    offloaded_layers = min(max(gpu_layers, 0), sizing.block_count)
    cpu_moe_layers = min(max(n_cpu_moe, 0), offloaded_layers)
    output_on_gpu = gpu_layers > sizing.block_count

    layer_totals = [layer.total_bytes for layer in sizing.layers]
    expert_totals = [layer.expert_bytes for layer in sizing.layers]

    gpu_layer_bytes = sum(layer_totals[:offloaded_layers])
    cpu_layer_bytes = sum(layer_totals[offloaded_layers:])
    moved_expert_bytes = sum(expert_totals[:cpu_moe_layers]) if sizing.supports_moe is not False else 0

    output_bytes = sizing.output_bytes if output_on_gpu else 0
    gpu_weight = max(0, gpu_layer_bytes - moved_expert_bytes + output_bytes)
    cpu_weight = max(0, cpu_layer_bytes + moved_expert_bytes + sizing.non_layer_bytes - output_bytes)

    gpu_kv_layers = offloaded_layers
    cpu_kv_layers = sizing.block_count - offloaded_layers
    gpu_kv = gpu_kv_layers * budget.kv_gpu_bytes_per_layer
    cpu_kv = cpu_kv_layers * budget.kv_cpu_bytes_per_layer

    gpu_total = gpu_weight + gpu_kv + budget.gpu_workspace_bytes
    cpu_total = cpu_weight + cpu_kv + budget.cpu_runtime_bytes
    return Candidate(
        gpu_layers=gpu_layers,
        n_cpu_moe=n_cpu_moe,
        gpu_weight_bytes=gpu_weight,
        cpu_weight_bytes=cpu_weight,
        gpu_kv_bytes=gpu_kv,
        cpu_kv_bytes=cpu_kv,
        gpu_total_bytes=gpu_total,
        cpu_total_bytes=cpu_total,
        gpu_over_bytes=max(0, gpu_total - budget.vram_total_bytes),
        cpu_over_bytes=max(0, cpu_total - budget.ram_available_bytes),
    )


def choose_candidate(candidates: list[Candidate]) -> Candidate:
    fitting = [candidate for candidate in candidates if candidate.fits]
    if fitting:
        return max(
            fitting,
            key=lambda c: (
                c.gpu_layers,
                -c.n_cpu_moe,
                budget_spare_score(c),
            ),
        )
    return min(
        candidates,
        key=lambda c: (
            c.gpu_over_bytes + c.cpu_over_bytes,
            c.gpu_over_bytes,
            c.cpu_over_bytes,
            -c.gpu_layers,
            c.n_cpu_moe,
        ),
    )


def budget_spare_score(candidate: Candidate) -> int:
    return -(candidate.gpu_total_bytes + candidate.cpu_total_bytes)


def recommendation_confidence(sizing: ModelSizing, candidate: Candidate) -> str:
    if not candidate.fits:
        return "low"
    if not sizing.exact_tensor_sizes or sizing.supports_moe is None:
        return "medium"
    return "high"


def recommendation_explanations(sizing: ModelSizing, budget: RuntimeBudget, candidate: Candidate) -> list[str]:
    lines: list[str] = []
    if candidate.fits:
        lines.append("Selected the highest GPU layer count that fits VRAM and host RAM for the target context, then minimized CPU MoE offload.")
    else:
        limiting = []
        if candidate.gpu_over_bytes > 0:
            limiting.append(f"VRAM over by {format_bytes(candidate.gpu_over_bytes)}")
        if candidate.cpu_over_bytes > 0:
            limiting.append(f"RAM over by {format_bytes(candidate.cpu_over_bytes)}")
        lines.append("No clean CPU/GPU split fit the current budgets; selected the closest candidate (" + ", ".join(limiting) + ").")
    if sizing.supports_moe is False:
        lines.append("Model is detected as dense/non-MoE, so n_cpu_moe is set to unset.")
    elif sizing.supports_moe is True:
        lines.append("Model is detected as MoE; n_cpu_moe moves expert tensors for the first N layers to CPU/RAM while keeping other offloaded tensors on GPU.")
    else:
        lines.append("MoE support could not be proven from metadata; recommendation is conservative.")
    if not sizing.exact_tensor_sizes:
        lines.extend(sizing.notes)
    lines.append(f"Target context: {budget.desired_context} tokens.")
    lines.append(f"Estimated GPU total: {format_bytes(candidate.gpu_total_bytes)} / {format_bytes(budget.vram_total_bytes)}.")
    lines.append(f"Estimated RAM total: {format_bytes(candidate.cpu_total_bytes)} / {format_bytes(budget.ram_available_bytes)} MemAvailable.")
    return lines


def format_bytes(value: int | float) -> str:
    value = float(value)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(value) < 1024 or unit == "TiB":
            if unit == "B":
                return f"{value:.0f} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"

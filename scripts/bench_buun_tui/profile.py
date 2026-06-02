from __future__ import annotations

from dataclasses import dataclass, field

CACHE_STRATEGIES = {
    "turbo": ("turbo4", "turbo3"),
    "q8": ("q8_0", "q8_0"),
    "f16": ("f16", "f16"),
    "unset": ("", ""),
}

CACHE_TYPE_OPTIONS = ["f32", "f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl", "turbo2", "turbo3", "turbo4", "turbo3_tcq", "turbo2_tcq", "unset"]


def normalize_cache_type(value: str) -> str:
    value = value.strip()
    if value.lower() in {"", "unset", "none", "default"}:
        return ""
    return value


def resolve_cache_types(profile: "Profile") -> tuple[str, str]:
    strategy_k, strategy_v = CACHE_STRATEGIES.get(profile.cache_strategy, CACHE_STRATEGIES["unset"])
    cache_type_k = normalize_cache_type(profile.cache_type_k) or strategy_k
    cache_type_v = normalize_cache_type(profile.cache_type_v) or strategy_v
    return cache_type_k, cache_type_v


def cache_types_text(profile: "Profile") -> str:
    k, v = resolve_cache_types(profile)
    return f"{k or 'unset'}/{v or 'unset'}"


def apply_cache_strategy(profile: "Profile", cache_strategy: str) -> None:
    profile.cache_strategy = cache_strategy
    profile.cache_type_k, profile.cache_type_v = CACHE_STRATEGIES.get(cache_strategy, CACHE_STRATEGIES["unset"])

FIELD_DESCRIPTIONS = {
    "model": "llama-bench -m/--model. GGUF model file discovered under ./models; bench source default is models/7B/ggml-model-q4_0.gguf but this TUI selects a local GGUF. Variant notes: choose a GGUF matching the scenario (quantized, MoE, MTP/REAP); no parameter conflict found in help/source. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "cache_strategy": "Preset that maps to llama-bench -ctk/--cache-type-k and -ctv/--cache-type-v. The individual cache_type_k/cache_type_v fields override it when set. Bench default is f16/f16; TUI default turbo uses buun turbo4/turbo3. Variant notes: quantized KV (turbo/q8) can reduce memory but may affect quality/compat; use f16 when debugging MTP/REAP or unsupported cache types. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "cache_type_k": "llama-bench -ctk/--cache-type-k. Overrides the preset cache_strategy for K when set. Choose a type supported by the model's head dimension; turbo4 on some models (like GLM A3B with n_embd_head_k=576) can fail if block size does not divide the head size. Refs: llama-bench --help; llama-context.cpp; buun README recommendations.",
    "cache_type_v": "llama-bench -ctv/--cache-type-v. Overrides the preset cache_strategy for V when set. Use this to mix K/V types independently for compatibility or memory tuning. Refs: llama-bench --help; llama-context.cpp; buun README recommendations.",
    "depths": "llama-bench -d/--n-depth. Default is 0. Prefills this many tokens before measuring; high depths stress KV memory. Variant notes: bad for low-memory quantized/MoE/MTP/REAP runs when combined with large batch/ubatch or f16 KV; reduce depth first on OOM. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "batch_size": "llama-bench -b/--batch-size. Default is 2048. Logical prompt-processing batch size. Larger improves prefill but increases memory/graph pressure. Variant notes: bad for long-context quantized/MoE/MTP/REAP if paired with high n-depth, high ubatch, f16 KV, or many GPU layers. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "ubatch_size": "llama-bench -ub/--ubatch-size. Default is 512. Physical micro-batch size. Lower reduces memory pressure; higher can improve throughput. Variant notes: for quantized/MoE/MTP/REAP, lower this before lowering batch_size when VRAM pressure or CUDA graph failures appear. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "threads": "llama-bench -t/--threads. Default is common_cpu_get_num_math() from source (6 on this binary's --help). CPU eval/decode threads; oversubscription can hurt latency. Variant notes: MoE CPU-offload needs CPU headroom, so avoid setting this too high together with high n_cpu_moe. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "n_cpu_moe": "buun llama-bench -ncmoe/--n-cpu-moe. Default is 0. Keeps first N MoE expert layers on CPU. Variant notes: MoE-specific; useless for dense/non-MoE. Higher saves VRAM but can bottleneck CPU/RAM; conflicts with very high threads if CPU is saturated. Refs: llama-bench --help; common/arg.cpp; docs/speculative.md.",
    "gpu_layers": "llama-bench -ngl/--n-gpu-layers. Default is 99. Offloads layers to GPU. Variant notes: high values can OOM quantized/MoE/MTP/REAP with high depth/batch/f16 KV; lower it or raise n_cpu_moe for MoE. 99 means as much as possible in bench source. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "prompt": "llama-bench -p/--n-prompt. Default is 512. Prompt/prefill tokens measured after depth; use 0 for generation-only depth tests. Variant notes: large prompt plus high depth/batch stresses KV and prompt kernels for quantized/MoE/MTP/REAP. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "gen": "llama-bench -n/--n-gen. Default is 128. Generated tokens to measure; smaller is faster but noisier. Variant notes: use nonzero generation when testing MTP/speculative behavior; prompt-only tests will not reflect draft-token acceptance. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp; docs/speculative.md.",
    "repetitions": "llama-bench -r/--repetitions. Default is 5; TUI default is 1 for faster exploration. Higher stabilizes numbers but multiplies runtime. Variant notes: use more reps for noisy MoE CPU-offload, quantized KV, MTP/REAP comparisons. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "warmup": "llama-bench --no-warmup toggle. Source default is warmup enabled; false appends --no-warmup. Variant notes: disabling warmup is bad for final quantized/MoE/MTP/REAP comparisons because first-run allocation/cache effects skew results. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "device": "llama-bench -dev/--device. Default is auto. CUDA0 targets the RTX GPU explicitly. Variant notes: for single-GPU quantized/MoE/MTP/REAP tests, CUDA0 avoids accidental CPU/other-device placement; use auto only when testing device selection. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "flash_attn": "llama-bench -fa/--flash-attn. Default is 0. 1 enables Flash Attention. Variant notes: usually useful for long context, but if a quantized/MTP/REAP model or KV type misbehaves, compare with 0; interacts with cache_strategy and high depth. Refs: llama-bench --help; tools/llama-bench/llama-bench.cpp.",
    "no_mmap": "llama-server --no-mmap. Disable memory mapping; server and llama-bench source/help default to mmap enabled. In llama-bench this is emitted as --mmap 0 when enabled. Variant notes: may reduce pageouts when not using mlock but slows load; no quantized/MTP/MoE/REAP-specific conflict found. Refs: llama-bench --help; common/arg.cpp; llama-model.cpp.",
}

PRESETS = {
    "depths": ["0,4096,8192,16384,32768", "0,8192,32768,65536", "0,32768,65536,131072", "custom", "unset"],
    "batch_size": ["1024", "512", "2048", "512,1024", "1024,2048", "custom", "unset"],
    "ubatch_size": ["256", "128", "512", "128,256", "256,512", "custom", "unset"],
    "threads": ["12", "8", "10", "8,10,12", "custom", "unset"],
    "n_cpu_moe": ["33", "40", "36", "32", "24", "33,36,40", "custom", "unset"],
    "gpu_layers": ["99", "80", "60", "99,80,60", "custom", "unset"],
    "prompt": ["0", "512", "0,512", "custom", "unset"],
    "gen": ["128", "32", "32,128", "custom", "unset"],
    "cache_type_k": CACHE_TYPE_OPTIONS,
    "cache_type_v": CACHE_TYPE_OPTIONS,
    "repetitions": ["1", "2", "3", "unset"],
    "warmup": ["false", "true", "unset"],
    "device": ["CUDA0", "auto", "unset"],
    "flash_attn": ["1", "0", "unset"],
    "no_mmap": ["unset", "false", "true"],
}

@dataclass
class Profile:
    name: str
    model: str = ""
    cache_strategy: str = "turbo"
    cache_type_k: str = ""
    cache_type_v: str = ""
    depths: str = "0,4096,8192,16384,32768"
    batch_size: str = "1024"
    ubatch_size: str = "256"
    threads: str = "12"
    n_cpu_moe: str = "33"
    gpu_layers: str = "unset"
    prompt: str = "512"
    gen: str = "128"
    repetitions: str = "1"
    warmup: str = "false"
    device: str = "CUDA0"
    flash_attn: str = "1"
    no_mmap: str = "unset"

    def items(self) -> list[tuple[str, str]]:
        return [(key, getattr(self, key)) for key in [
            "model", "cache_strategy", "cache_type_k", "cache_type_v", "depths", "batch_size", "ubatch_size",
            "threads", "n_cpu_moe", "gpu_layers", "prompt", "gen", "repetitions",
            "warmup", "device", "flash_attn", "no_mmap",
        ]]

@dataclass
class BenchConfig:
    profiles: list[Profile] = field(default_factory=lambda: [Profile("profile_a"), Profile("profile_b")])
    output_format: str = "jsonl"
    run_prefix: str = "bench"

# Plan 004 — Bigger Context on Large Models via Aggressive `n-cpu-moe`

**Date:** 2026-06-28
**Author:** Automated plan based on current presets and hardware profile
**Status:** In Progress — turbo3/turbo4 defaults changed the VRAM calculus; upstream fork unaffected

---

## Fork-specific notes

This plan applies to **buun** and **TurboQuant+** forks only, both of which support the `turbo3`/`turbo4` asymmetric KV cache types. The **upstream** vanilla llama.cpp fork does **not** support these cache types — it uses standard `q8_0`, `f16`, etc.

| Fork | Global KV defaults (2026-07) | Plan applicability |
|---|---|---|
| buun (`model-presets.buun.ini`) | `cache-type-k = turbo3`, `cache-type-v = turbo4` | ✅ Full — turbo3/turbo4 VRAM savings are the primary delta |
| TurboQuant+ (`model-presets.turboquant.ini`) | `cache-type-k = turbo3`, `cache-type-v = turbo4` | ✅ Full — identical KV semantics to buun |
| Upstream (`model-presets.upstream.ini`) | `cache-type-k = q8_0`, `cache-type-v = q8_0` | ⚠️ Partial — only `n-cpu-moe` and ctx-size changes apply; KV cache sizes in this plan are irrelevant for upstream |

> **Key delta (2026-07):** The buun and TurboQuant+ presets moved from `q8_0/turbo3` to `turbo3/turbo4` globally. This roughly **halved the KV cache footprint** for all models, freeing ~0.5–1.0 GB of VRAM that was previously reserved for KV at 64K context. The upstream preset still uses `q8_0/q8_0` and is unaffected by this change.

---

## Executive summary

The largest models in this setup (Qwen3.6-35B-A3B MoE, GLM-4.7-Flash, GPT-OSS-20B) are currently pinned to a **16 GB VRAM RTX 5060 Ti** with `ctx-size = 65536` (global default) or `32768` (explicit per-preset overrides). The goal of this plan is to **increase context sizes toward 128K** for these large models by **offloading more MoE layers to the CPU** via `n-cpu-moe`, which frees VRAM for the larger KV cache that 128K context demands.

The core trade-off is:

| Lever | Effect | Cost |
|---|---|---|
| ↑ `n-cpu-moe` | Frees VRAM from MoE weights → bigger KV cache | MoE layers run on CPU → slower inference per token |
| ↑ `ctx-size` | Bigger context window | Larger KV cache → more VRAM needed |
| ↓ `cache-type-k/v` | Smaller KV cache per token | Lower precision → potentially lower quality |
| ↑ `cache-reuse` | Reduces effective KV cache pressure via prompt caching | Only helps on repeated/overlapping prompts |

This plan focuses on **`n-cpu-moe`** as the primary lever because it directly frees VRAM for the KV cache without sacrificing KV precision or quality.

---

## Hardware Profile

| Resource | Value | Notes |
|---|---|---|
| GPU 0 | RTX 3070 — 8 GB VRAM | Small models only |
| GPU 1 | RTX 5060 Ti — 16.3 GB VRAM | Large models only; ~13.5 GB currently in use |
| RAM | 16 GB system RAM | MoE CPU offload, system services |
| CPU | AMD Ryzen 5 5600X — 12 threads | MoE CPU computation |

**GPU usage policy:** Both GPUs are dedicated to inference only — no desktop compositing, no display server, no other GPU-accelerated workloads. Every MiB of VRAM not used by model weights or KV cache is available for context.

**VRAM constraint:** The RTX 5060 Ti has 16.3 GB VRAM. With no competing workloads, the only consumers are:
1. Model weights loaded on GPU (via `n-gpu-layers`)
2. KV cache for the active context
3. Minimal CUDA runtime / driver overhead (~50-100 MiB on a headless server)

…so the usable VRAM budget is ~16.0 GB. There is **no desktop compositor or display overhead** to account for.

---

## Current State: Large Model Presets

> All snippets below are from `model-presets.buun.ini` (buun fork). The TurboQuant+ preset (`model-presets.turboquant.ini`) is nearly identical — same turbo3/turbo4 defaults, same `n-cpu-moe` values, same 128K context targets. The upstream preset (`model-presets.upstream.ini`) uses `q8_0/q8_0` KV cache and different effective VRAM constraints.

### `1-smart-qwen3-6-35b-a3b` (Qwen3.6-35B-A3B, IQ4_NL, ~17 GB file)

```ini
model = .../Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
device = CUDA1
n-gpu-layers = 99
n-cpu-moe = 16
ctx-size = 131072
cache-type-k = turbo3
cache-type-v = turbo4
cache-reuse = 256
```

### `5-compare-qwen3-6-35b-a3b-xs` (Qwen3.6-35B-A3B, IQ4_XS, ~17 GB file)

```ini
model = .../Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
device = CUDA1
n-gpu-layers = 99
n-cpu-moe = 14
ctx-size = 131072
cache-type-k = turbo3
cache-type-v = turbo4
```

### `2-glm-flash` (GLM-4.7-Flash, Q4_K_M, ~18 GB file)

```ini
model = .../GLM-4.7-Flash-Q4_K_M.gguf
device = CUDA1
n-gpu-layers = 99
n-cpu-moe = 20
ctx-size = 131072
cache-type-k = turbo3
cache-type-v = turbo4
```

### `6-compare-glm-flash-reap-23b-a3b-xs` (GLM-4.7-Flash REAP, IQ4_XS, ~12 GB file)

```ini
model = .../GLM-4.7-Flash-REAP-23B-A3B-IQ4_XS.gguf
device = CUDA1
n-gpu-layers = 99
n-cpu-moe = 12
ctx-size = 131072
cache-type-k = turbo3
cache-type-v = turbo4
```

### `10-qwen3.6-27b` (Qwen3.6-27B, IQ4_XS, dense — buun/TurboQuant+)

```ini
model = .../Qwen3.6-27B-IQ4_XS.gguf
device = CUDA1
n-gpu-layers = 59
ctx-size = 32768
cache-type-k = turbo3
cache-type-v = turbo4
```

> **Note:** This dense model is currently capped at 32K because it was evaluated under the old `q8_0` K-cache regime. With turbo3 on K now, 65K may be feasible without changing `n-gpu-layers`.

### Other large models (buun/TurboQuant+)

| Preset | Model | VRAM | Status |
|---|---|---|---|
| `7-gemma-4-12b-it-qat` | gemma-4-12B-it-qat (Q4_K_XL) | ~7-8 GB | Inherits global 65K ctx; no explicit offload |
| `8-gemma-4-26b` | gemma-4-26B-A4B (Q4_K_XL) | ~12-14 GB | Dense, not MoE; inherits global 65K ctx |
| `9-gpt-oss-20b` | gpt-oss-20b (Q4_K_M) | ~11-13 GB | Dense, not MoE; inherits global 65K ctx |

### Upstream preset differences (for reference)

| Preset | Buun `n-cpu-moe` | Upstream `n-cpu-moe` | Buun KV | Upstream KV |
|---|---|---|---|---|
| `1-smart-qwen3-6-35b-a3b` | 16 | 16 | turbo3/turbo4 | q8_0/q8_0 |
| `2-glm-flash` | 20 | 20 | turbo3/turbo4 | q8_0/q8_0 |
| `5-compare-qwen3-6-35b-a3b-xs` | 14 | 14 | turbo3/turbo4 | q8_0/q8_0 |
| `6-compare-glm-flash-reap-23b-xs` | 12 | 12 | turbo3/turbo4 | q8_0/q8_0 |
| `10-qwen3.6-27b` | 59 (n-gpu-layers) | 59 (n-gpu-layers) | turbo3/turbo4 | q8_0/q8_0 |

> The upstream preset uses the same `n-cpu-moe` values but `q8_0/q8_0` KV cache, meaning **~2–3× larger KV cache** than buun/TurboQuant+. This is why upstream dense models are more constrained on context size.

---

## Understanding `n-cpu-moe`

From the buun llama.cpp source (`common/arg.cpp`):

> `--n-cpu-moe N` — keep the Mixture of Experts (MoE) weights of the first **N** layers in the CPU

The implementation iteratively applies CPU buffer type overrides to the MoE exps blocks of the first N FFN layers:

```cpp
for (int i = 0; i < value; ++i) {
    buft_overrides.push_back(llm_ffn_exps_block_regex(i));
    params.tensor_buft_overrides.push_back({buft_overrides.back().c_str(), ggml_backend_cpu_buffer_type()});
}
```

**Key insight:** `n-cpu-moe` only affects the **MoE expert weights** within the first N FFN layers. It does NOT move attention layers, embeddings, or output layers to CPU. This means:

- **VRAM freed per MoE layer:** Proportional to the expert weight matrix sizes (depends on model architecture and expert count)
- **Performance cost:** Each MoE layer now requires CPU→GPU data transfer for expert weights during inference
- **Best for:** MoE models where expert weights are the dominant VRAM consumer after attention layers

For the Qwen3.6-35B-A3B model (~35B total params, ~3B active), the MoE layers are a significant VRAM consumer. Moving more of them to CPU directly frees VRAM for the KV cache.

---

## VRAM Budget Calculation

### KV Cache Size Formula

For a transformer model:

```
KV cache size ≈ 2 × num_layers × hidden_size × ctx_size × cache_type_bytes
```

Where:
- `num_layers` = model depth
- `hidden_size` = model embedding dimension
- `ctx_size` = context length
- `cache_type_bytes` = bytes per KV element (varies by cache type)

### Approximate KV Cache Sizes for 128K Context

> Sizes below are for **buun/TurboQuant+** with `turbo3/turbo4` defaults. For **upstream** (`q8_0/q8_0`), multiply by ~3–4×.

| Model | Layers | Hidden | Cache Type | 32K KV | 64K KV | 128K KV |
|---|---:|---:|---|---:|---:|---:|
| Qwen3.6-35B-A3B | ~40 | 4096 | turbo3/turbo4 | ~400 MB | ~800 MB | ~1.6 GB |
| GLM-4.7-Flash | ~40 | 5120 | turbo3/turbo4 | ~500 MB | ~1.0 GB | ~2.0 GB |
| GPT-OSS-20B | ~44 | 4096 | turbo3/turbo4 | ~350 MB | ~700 MB | ~1.4 GB |
| Qwen3.6-27B (dense) | ~48 | 3584 | turbo3/turbo4 | ~300 MB | ~600 MB | ~1.2 GB |
| Gemma-4-26B (dense) | ~44 | 5120 | turbo3/turbo4 | ~500 MB | ~1.0 GB | ~2.0 GB |

| Model | q8_0/q8_0 (upstream) 64K KV | q8_0/q8_0 (upstream) 128K KV |
|---|---:|---:|
| Qwen3.6-35B-A3B | ~2.5 GB | ~5.0 GB |
| GLM-4.7-Flash | ~3.2 GB | ~6.4 GB |
| Qwen3.6-27B (dense) | ~2.0 GB | ~4.0 GB |

> **Turbo3/turbo4 delta:** Compared to the old `q8_0/turbo3` default (K=8-bit, V≈3-bit), switching K to turbo3 (~3-bit) roughly **halved** the KV cache for all models. This freed ~0.5–1.0 GB of VRAM at 64K context that was previously consumed by key caches.

### Current VRAM Usage (from `nvidia-smi`)

```
RTX 5060 Ti: 13518 MiB / 16311 MiB (82.9% used)
```

This means approximately **2.8 GB free** for KV cache. Since no other workloads compete for VRAM, this free space is **entirely available** for context expansion — no additional headroom buffer needed for competing processes.

### VRAM Budget for 128K Context (buun/TurboQuant+ with turbo3/turbo4)

Starting with ~2.8 GB free:

| Component | 64K Context | 128K Context | Δ |
|---|---:|---:|---:|
| KV cache (Qwen A3B) | ~800 MB | ~1.6 GB | +800 MB |
| KV cache (GLM Flash) | ~1.0 GB | ~2.0 GB | +1.0 GB |
| KV cache (Qwen 27B dense) | ~600 MB | ~1.2 GB | +600 MB |
| Model weights on GPU | varies | same | 0 |
| **Headroom needed** | baseline | **+600 MB to +1.0 GB** | |

To absorb this, we need to free **~0.6–1.0 GB of VRAM** by moving more MoE layers to CPU.

> **Turbo3/turbo4 delta:** Under the old `q8_0/turbo3` default, the headroom needed was ~1.0–1.5 GB because the K-cache was 8-bit. The turbo3/turbo4 shift reduced this by ~0.5 GB, meaning the current `n-cpu-moe` values (16, 20, 14, 12) are already closer to optimal than the original plan assumed.

### VRAM Budget for 128K Context (upstream with q8_0/q8_0)

| Component | 64K Context | 128K Context | Δ |
|---|---:|---:|---:|
| KV cache (Qwen A3B) | ~2.5 GB | ~5.0 GB | +2.5 GB |
| KV cache (GLM Flash) | ~3.2 GB | ~6.4 GB | +3.2 GB |
| Model weights on GPU | varies | same | 0 |
| **Headroom needed** | baseline | **+2.5–3.2 GB** | |

> Upstream cannot reach 128K on these models with `q8_0/q8_0` KV cache on a 16 GB GPU — the KV cache alone exceeds available VRAM even with max MoE offload. Upstream is realistically capped at 32–64K for large models.

---

## Proposed `n-cpu-moe` Values

### Status: Already Applied (buun/TurboQuant+)

The presets have already been updated from the original plan's proposed values. The turbo3/turbo4 shift reduced the required `n-cpu-moe` bumps, so the presets landed close to optimal without needing the full proposed increases.

| Preset | Original Plan `n-cpu-moe` | Current Preset `n-cpu-moe` | Status |
|---|---|---|---|
| `1-smart-qwen3-6-35b-a3b` | 20 | **16** | ✅ Applied — turbo3/turbo4 saved ~0.5 GB, so 16 is sufficient |
| `5-compare-qwen3-6-35b-a3b-xs` | 18 | **14** | ✅ Applied — XS quant + turbo3/turbo4 = sufficient |
| `2-glm-flash` | 24 | **20** | ✅ Applied — turbo3/turbo4 delta covers the gap |
| `6-compare-glm-flash-reap-23b-xs` | 16 | **12** | ✅ Applied — XS quant + turbo3/turbo4 = sufficient |

### Dense Models: 32K → 65K (new opportunity)

With turbo3/turbo4, dense models that were previously capped at 32K may now fit 65K:

| Preset | Model | Current ctx | Proposed ctx | Reason |
|---|---|---|---|---|
| `10-qwen3.6-27b` | Qwen3.6-27B (IQ4_XS) | 32K | **65K** | turbo3 on K halves the KV footprint; 65K KV ≈ 600 MB fits alongside ~9 GB model weights |
| `8-gemma-4-26b` | Gemma-4-26B (Q4_K_XL) | 65K (inherited) | **65K** | Already at 65K; test 128K if VRAM allows |
| `9-gpt-oss-20b` | GPT-OSS-20B (Q4_K_M) | 65K (inherited) | **65K** | Already at 65K; test 128K if VRAM allows |

> **Note:** GPT-OSS-20B and Gemma-4-26B don't have explicit `ctx-size` overrides in the presets — they inherit the global 65K default. They should be tested at 65K first, then 128K if headroom allows. Neither is MoE, so `n-cpu-moe` won't help them.

---

## Alternative Levers (Complementary)

If `n-cpu-moe` alone doesn't free enough VRAM for 128K, combine with:

### 1. Reduce KV Cache Precision (buun/TurboQuant+ only)

| Current | Alternative | Δ per token | Applicable to |
|---|---|---|---|
| turbo3/turbo4 (~0.125 B/tok/layer) | turbo2/turbo2 (~0.063 B/tok/layer) | −50% KV cache | buun, TurboQuant+ |
| turbo3/turbo4 | turbo3_tcq/turbo2_tcq | ~−30-40% KV cache | buun, TurboQuant+ |

**Trade-off:** Lower KV precision may reduce long-context quality. Benchmark carefully.

> **Upstream note:** Upstream can't use turbo types. Its options are `f16/f16` (4× larger than q8_0/q8_0) or `q8_0/q8_0` (current). No further reduction is possible without a fork-specific extension.

### 2. Increase `cache-reuse`

Current: `cache-reuse = 256` (default for prompt caching)

Increasing to `512` or `1024` reduces effective KV cache pressure when prompts overlap significantly. This only helps if the client supports prompt caching with overlapping prefixes.

### 3. Use `--fit on` for automatic VRAM management

```ini
; In preset or service args
fit = on
```

This lets llama.cpp automatically reduce context or offload more layers to avoid OOM. Useful as a safety net but not precise.

### 4. KV Cache Offloading

```ini
; If supported in buun fork
kv-cache-offload = on
```

Offloads idle KV cache slots to CPU RAM. Reduces VRAM pressure but adds latency on context switches.

> **Upstream note:** KV cache offloading support depends on the fork. Test availability per fork before relying on it.

---

## Implementation Plan

### Phase 1: Benchmark Current Baseline

> **Update:** The turbo3/turbo4 shift means the original baseline (q8_0/turbo3) is no longer relevant. Benchmark against the **current presets** (turbo3/turbo4) as the new baseline.

For each large model preset, record:

| Metric | Method |
|---|---|
| VRAM usage at load | `nvidia-smi` immediately after load |
| VRAM usage at 64K context | Send 60K-token prompt, check `nvidia-smi` |
| VRAM usage at 128K context | Send 120K-token prompt, check `nvidia-smi` |
| tok/s at 64K context | Measure generation speed |
| tok/s at 128K context | Measure generation speed |
| OOM threshold | Max context before OOM |

**Additional benchmarks needed for turbo3/turbo4 delta:**

| Metric | Method |
|---|---|
| Dense model 32K → 65K | Test `10-qwen3.6-27b` at 65K; check for OOM or CPU offload |
| Upstream vs buun KV delta | Run same model on upstream (q8_0/q8_0) and buun (turbo3/turbo4); compare VRAM at 64K |
| Gemma-4-26B at 128K | Test if Gemma-4-26B fits 128K with turbo3/turbo4 and `n-gpu-layers = 99` |

### Phase 2: Incremental `n-cpu-moe` Testing

For each large model, test in increasing steps:

```
n-cpu-moe = 12 → 16 → 20 → 24 → 28 (stop before model fails to load)
```

At each step:
1. Reload the model
2. Check VRAM usage
3. Test with 64K context
4. Test with 96K context (intermediate target)
5. Test with 128K context
6. Record tok/s for each

### Phase 3: Select Optimal Values

For each model, select the `n-cpu-moe` value that achieves:
- **Primary:** 128K context loads without OOM
- **Secondary:** Acceptable tok/s (>5 tok/s for chat, >3 tok/s for long context)
- **Tertiary:** Output quality preserved (subjective check)

If 128K still doesn't fit even at max `n-cpu-moe`, fall back to:
- 96K context (practical sweet spot for most use cases)
- Lower KV cache precision (turbo2/turbo2 for buun/TurboQuant+)

### Phase 4: Apply Preset Changes

Update `model-presets.buun.ini` with the validated values.

### Phase 5: Update Other Preset Files

Mirror changes to:
- `model-presets.turboquant.ini` (if applicable)
- `model-presets.upstream.ini` (if applicable, with upstream-compatible cache types)

### Phase 6: Service Restart & Validation

```bash
./manage-server.sh restart
./manage-server.sh status
./manage-server.sh logs
```

Test each preset through the pi extension.

---

## Already Applied (turbo3/turbo4 delta) ✅

The following changes from the original draft have already been applied to `model-presets.buun.ini` and `model-presets.turboquant.ini`:

- ✅ `cache-type-k = turbo3`, `cache-type-v = turbo4` as global defaults (buun/TurboQuant+)
- ✅ `n-cpu-moe` bumped to 16/20/14/12 for MoE models
- ✅ `ctx-size = 131072` for MoE models (1, 2, 4, 5, 6)
- ✅ `n-gpu-layers = 99` for MoE models (previously 99 or `all`)

## Still Needed (pending validation)

### `10-qwen3.6-27b` — Try 65K context

```diff
 [10-qwen3.6-27b]
 model = /home/yuriteixeira/Workspaces/local/services/llama/models/unsloth/Qwen3.6-27B-IQ4_XS.gguf
 device = CUDA1
-; Dense 27B cannot fully fit in 16GB with q8 KV; 32k reduces CPU-offloaded layers vs the global 65k ctx.
-n-gpu-layers = 59
-ctx-size = 32768
+; turbo3/turbo4 halves KV footprint vs q8_0; 65K KV ≈ 600 MB should fit.
+n-gpu-layers = 59
+ctx-size = 65536
```

### `8-gemma-4-26b` — Verify 65K, test 128K

```diff
 [8-gemma-4-26b]
 model = /home/yuriteixeira/Workspaces/local/services/llama/models/unsloth/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
 device = CUDA1
 ; Gemma 4 26B is dense (not MoE), so n-cpu-moe won't help.
 ; Use n-gpu-layers offload instead, or rely on turbo KV cache for compactness.
+n-gpu-layers = 99
+ctx-size = 65536
+; Test: can 128K fit with turbo3/turbo4? KV ≈ 2 GB at 128K.
```

### `9-gpt-oss-20b` — Verify 65K, test 128K

```diff
 [9-gpt-oss-20b]
 model = /home/yuriteixeira/Workspaces/local/services/llama/models/unsloth/gpt-oss-20b-Q4_K_M.gguf
 device = CUDA1
 ; GPT-OSS 20B is dense (not MoE), so n-cpu-moe won't help.
+n-gpu-layers = 99
+; Test: can 128K fit with turbo3/turbo4? KV ≈ 1.4 GB at 128K.
```

### Upstream preset — Document 128K infeasibility

For `model-presets.upstream.ini`, add a comment noting that 128K context is not feasible with `q8_0/q8_0` KV cache on a 16 GB GPU for large models:

```diff
 ; NOTE: Upstream uses q8_0/q8_0 KV cache (no turbo support).
 ; 128K context is not feasible for models >15B on 16 GB VRAM with q8_0/q8_0.
 ; Recommended max: 32K for dense models, 64K for MoE models with heavy n-cpu-moe.
```

## Fork Comparison Summary

| Aspect | Buun / TurboQuant+ | Upstream |
|---|---|---|
| KV cache type | turbo3/turbo4 (global) | q8_0/q8_0 (global) |
| 64K KV (Qwen A3B) | ~800 MB | ~2.5 GB |
| 128K KV (Qwen A3B) | ~1.6 GB | ~5.0 GB |
| 128K feasible? | ✅ Yes (with n-cpu-moe) | ❌ No (>16 GB needed) |
| 65K feasible for dense? | ✅ Yes (new with turbo3/turbo4) | ⚠️ Marginal (depends on model) |
| `n-cpu-moe` values | Same across forks | Same across forks |

---

## Benchmark Checklist

Record for every model + configuration tested:

| Field | Description |
|---|---|
| `model` | Model name + quant |
| `preset` | Preset name from ini |
| `n-gpu-layers` | Value used |
| `n-cpu-moe` | Value used |
| `ctx-size` | Value used |
| `cache-type-k` | Value used |
| `cache-type-v` | Value used |
| `cache-reuse` | Value used |
| `load_success` | true/false |
| `peak_vram_mib` | Peak VRAM usage in MiB |
| `peak_ram_mib` | Peak RAM usage in MiB |
| `prompt_tok` | Number of prompt tokens sent |
| `prompt_tok_s` | Prompt processing speed |
| `gen_tok_s` | Generation speed (tokens/sec) |
| `max_ctx_no_oom` | Maximum context before OOM |
| `quality_check` | Short output sanity (pass/fail) |

---

## Risks & Watchpoints

| Risk | Impact | Mitigation |
|---|---|---|
| High `n-cpu-moe` → slow inference | MoE layers on CPU are much slower than GPU | Benchmark tok/s at each step; set a minimum acceptable speed |
| `n-cpu-moe` too high → model fails to load | Too much offloaded, CPU RAM fills up | Monitor RAM usage; cap at model-specific max |
| 128K context still doesn't fit | VRAM budget exhausted even at max `n-cpu-moe` | Fall back to 96K context + lower KV precision |
| GLM + turbo3/turbo4 quality degradation | Asymmetric turbo may hurt GLM specifically | Test GLM with `q8_0/q8_0` as fallback if turbo is problematic |
| Prompt caching interactions | `cache-reuse` may behave differently with large contexts | Test prompt caching explicitly with 128K contexts |
| Service restart failures | Large context changes may cause cold-start timeouts | Increase `TimeoutStartSec` if needed (already 30 min) |
| Upstream preset confusion | Mixing turbo3/turbo4 configs with upstream | Clearly label fork-specific changes; use separate preset files |

---

## Success Criteria

- [ ] At least one large model (Qwen3.6-35B-A3B or GLM-4.7-Flash) loads and generates with **128K context** without OOM
- [ ] All large model presets have documented optimal `n-cpu-moe` values
- [ ] Generation speed remains **>5 tok/s** for chat use at 64K context
- [ ] Generation speed remains **>2 tok/s** for chat use at 128K context
- [ ] Output quality preserved (subjective check against baseline)
- [ ] Presets documented in `model-presets.buun.ini` with comments explaining rationale
- [ ] Changes mirrored to `model-presets.turboquant.ini` where applicable
- [ ] **New:** Dense models (Qwen3.6-27B, Gemma-4-26B) load with **65K context** using turbo3/turbo4 without `n-cpu-moe`
- [ ] **New:** Upstream preset documents 128K infeasibility and recommended max context

---

## Rollback

If any change causes issues:

```bash
# Revert to known-good presets (git history)
git checkout HEAD -- model-presets.buun.ini model-presets.turboquant.ini model-presets.upstream.ini

# Or manually set n-cpu-moe back to original values
# 1-smart-qwen3-6-35b-a3b: n-cpu-moe = 16 (was 12 in original plan)
# 5-compare-qwen3-6-35b-a3b-xs: n-cpu-moe = 14 (was 10 in original plan)
# 2-glm-flash: n-cpu-moe = 20 (was 16 in original plan)
# 6-compare-glm-flash-reap-23b-xs: n-cpu-moe = 12 (was n-gpu-layers = all in original plan)
# 10-qwen3.6-27b: ctx-size = 32768 (revert from 65536)
```

Then:

```bash
./manage-server.sh restart
```

---

## Appendix: Related Parameters Reference

### `n-cpu-moe` vs `cpu-moe`

| Parameter | Effect | Scope |
|---|---|---|
| `n-cpu-moe N` | First N MoE layers on CPU | Granular, layer-by-layer |
| `cpu-moe` (no value) | All MoE layers on CPU | All-or-nothing |

### KV Cache Types (buun turbo)

| Type | K bits | V bits | Relative size | Quality | Applicable to |
|---|---:|---:|---|---|---|
| `f32` | 32 | 32 | 1.0x | Reference | All forks |
| `f16` | 16 | 16 | 0.5x | Near-reference | All forks |
| `q8_0` | 8 | 8 | 0.25x | Good | All forks |
| `turbo3` (K) | ~3 | — | ~0.09x | Good | buun, TurboQuant+ only |
| `turbo4` (V) | — | ~4 | ~0.125x | Good | buun, TurboQuant+ only |
| `turbo2` (K) | ~2 | — | ~0.06x | Acceptable | buun, TurboQuant+ only |
| `turbo2_tcq` (V) | — | ~2 | ~0.06x | Lower | buun, TurboQuant+ only |
| `turbo3_tcq` (K) | ~3 | — | ~0.09x | Good | buun, TurboQuant+ only |
| `turbo4_tcq` (V) | — | ~4 | ~0.125x | Good | buun, TurboQuant+ only |

> **Global defaults (buun/TurboQuant+):** `cache-type-k = turbo3`, `cache-type-v = turbo4`
> **Global defaults (upstream):** `cache-type-k = q8_0`, `cache-type-v = q8_0`

### `cache-reuse`

| Value | Effect |
|---|---|
| 0 | Disabled |
| 256 | Default — chunks ≥256 tokens can be reused |
| 512 | Fewer, larger chunks reused (less cache pressure) |
| 1024 | Very aggressive reuse (best for repeated long prompts) |

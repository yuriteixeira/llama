# Plan 002 — Preset Optimization for RTX 3070 8GB

**Date:** 2026-06-06
**Author:** Automated audit reconciled against the working `model-presets.ini`
**Status:** Reconciled with current presets; some earlier suggestions were not kept because they did not work reliably

---

## Executive summary

This plan documents the **current working direction** for `model-presets.ini` on an RTX 3070 8GB machine using the buun llama.cpp fork.

The previous plan over-recommended `turbo4` KV cache for every large/MoE model. After comparing with the actual `model-presets.ini`, the working configuration is more conservative:

- Global defaults remain:
  - `ctx-size = 65536`
  - `cache-type-k = q8_0`
  - `cache-type-v = q8_0`
  - `cache-reuse = 256`
  - `logits-all = false`
  - `reasoning = off`
- Large Qwen MoE presets use explicit GPU/CPU offload plus `turbo4` KV:
  - `n-gpu-layers = 99`
  - `n-cpu-moe = 35`
  - `ctx-size = 32768`
  - `cache-type-k/v = turbo4`
- GLM presets use explicit GPU/CPU offload and reduced context, but **do not override cache type**:
  - they inherit global `q8_0` KV cache
  - the previous `turbo4` recommendation for GLM should be treated as unproven / not currently working
- Small models run fully on GPU and inherit global `q8_0` KV:
  - Qwen3.5-9B: `n-gpu-layers = all`, `ctx-size = 65536`
  - Gemma-4-E4B: `n-gpu-layers = all`, `ctx-size = 131072`

Overall: keep the actual working split — `turbo4` for Qwen3.6 A3B presets, `q8_0` inherited for GLM and small models — unless future benchmarks prove a better setting.

---

## Current `model-presets.ini` baseline

```ini
version = 1

[*]
ctx-size = 65536
cache-type-k = q8_0
cache-type-v = q8_0
cache-reuse = 256
logits-all = false
load-on-startup = false
reasoning = off
chat-template-kwargs = {"enable_thinking":true}
```

### Important reconciliation notes

| Earlier suggestion | Current reality | Plan update |
|---|---|---|
| Use `turbo4` for all large MoE models | Only Qwen3.6 A3B presets use `turbo4`; GLM presets inherit `q8_0` | Do not recommend `turbo4` for GLM as a default anymore |
| Remove `logits-all = false` | It is still present in the working config | Keep it documented as current behavior; only remove if the server rejects it |
| Validate `reasoning = off` | It is still present in the working config | Keep it documented as current behavior; only change if logs show it is ignored/rejected |
| Add explicit per-model context/offload | Implemented for all active presets | Keep |
| Small models use full GPU | Implemented with `n-gpu-layers = all` | Keep |

---

## Active preset recommendations

### 1. `1-smart-qwen3-6-35b-a3b`

Current preset:

```ini
n-gpu-layers = 99
n-cpu-moe = 35
ctx-size = 32768
cache-type-k = turbo4
cache-type-v = turbo4
```

Recommendation: **keep as-is**.

Rationale: this is the main large Qwen MoE preset. Explicit `n-gpu-layers`, `n-cpu-moe`, reduced context, and `turbo4` KV are aligned with the RTX 3070 8GB constraint.

---

### 2. `2-glm-flash`

Current preset:

```ini
n-gpu-layers = 99
n-cpu-moe = 35
ctx-size = 32768
# cache-type-k/v inherited from global q8_0
```

Recommendation: **keep inherited `q8_0`; do not re-add `turbo4` by default**.

Rationale: the previous plan recommended `turbo4`, but the actual working preset does not use it. Treat GLM + `turbo4` as a benchmark experiment only, not the baseline.

---

### 3. `3-fast-qwen3-5-9b`

Current preset:

```ini
n-gpu-layers = all
ctx-size = 65536
# cache-type-k/v inherited from global q8_0
```

Recommendation: **keep as-is**.

Rationale: this model fits on GPU and does not need MoE offload. `q8_0` KV remains the stable default.

---

### 4. `4-lightweight-gemma`

Current preset:

```ini
n-gpu-layers = all
ctx-size = 131072
# cache-type-k/v inherited from global q8_0
```

Recommendation: **keep as-is**.

Rationale: Gemma-4-E4B is treated as dense, not MoE. The current full-GPU + large-context preset matches the intended lightweight profile.

---

### 5. `5-compare-qwen3-6-35b-a3b-xs`

Current preset:

```ini
n-gpu-layers = 99
n-cpu-moe = 35
ctx-size = 32768
cache-type-k = turbo4
cache-type-v = turbo4
```

Recommendation: **keep as-is**.

Rationale: this comparison preset mirrors the main Qwen3.6 A3B setup while testing the IQ4_XS quant.

---

### 6. `6-compare-glm-flash-reap-23b-a3b-xs`

Current preset:

```ini
n-gpu-layers = 99
n-cpu-moe = 30
ctx-size = 32768
# cache-type-k/v inherited from global q8_0
```

Recommendation: **keep inherited `q8_0`; do not re-add `turbo4` by default**.

Rationale: the current working config uses `q8_0` for this GLM REAP preset. `n-cpu-moe = 30` is more conservative than the larger GLM model and should remain the baseline.

---

## Benchmark / future-change criteria

Only change the current presets when a benchmark records:

- exact model and quant
- `ctx-size`
- `n-gpu-layers`
- `n-cpu-moe`, if applicable
- `cache-type-k` / `cache-type-v`
- load success/failure
- peak VRAM
- peak RAM
- prompt tok/s
- generation tok/s
- short output sanity check

### Experiments worth trying later

| Experiment | Scope | Baseline to compare against |
|---|---|---|
| GLM with `turbo4` KV | `2-glm-flash`, `6-compare-glm-flash-reap-23b-a3b-xs` | Current inherited `q8_0` |
| Qwen3.6 A3B `n-cpu-moe = 30` or `25` | Qwen MoE presets | Current `n-cpu-moe = 35` |
| Qwen3.6 A3B `turbo3_tcq` | Qwen MoE presets | Current `turbo4` |
| Small models with `turbo4` | Qwen3.5-9B, Gemma-4-E4B | Current inherited `q8_0` |

Do not promote an experiment into the default config unless it beats the current baseline on this RTX 3070 8GB machine.

---

## Final target state

The plan now matches the active `model-presets.ini`:

- Global stable default: `q8_0` KV, 64K context.
- Qwen3.6 A3B presets: explicit MoE offload + 32K context + `turbo4` KV.
- GLM presets: explicit MoE offload + 32K context + inherited `q8_0` KV.
- Small models: full GPU + inherited `q8_0` KV.

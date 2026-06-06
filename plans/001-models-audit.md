# Plan 001 — Models Audit

**Date:** 2026-06-06
**Author:** Automated audit based on current local inventory
**Status:** Presets updated — model deletion/download intentionally skipped

---

## Hardware Profile

| Resource | Value |
|---|---|
| GPU | NVIDIA RTX 3070 — 8 GB VRAM (~7.65 GB usable) |
| RAM | 16 GB system RAM |
| CPU | AMD Ryzen 5 5600X (6-core / 12-thread) |

**Implication:** Prefer Q4 quantized ≤9B dense models, or MoE models with ~3B active params. Larger files can run only with CPU offload and reduced context/headroom.

---

## Current Model Inventory

Local `models/**/*.gguf` as of this update:

| Model | Size | Repo | Verdict |
|---|---:|---|---|
| `Qwen3.5-9B-Q4_K_M.gguf` | 5.3 GB | unsloth | ✅ Keep — fast general/coding fallback |
| `Qwen3-Coder-Next-Q4_K_M.gguf` | 6.3 GB | unsloth | ✅ Keep — primary fast coding candidate |
| `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` | 17 GB | unsloth | ✅ Keep — main high-quality MoE candidate |
| `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | 17 GB | unsloth | ✅ Keep for now — lighter quant alternative to compare against NL |
| `GLM-4.7-Flash-Q4_K_M.gguf` | 18 GB | unsloth | ✅ Keep — main GLM Flash candidate |
| `GLM-4.7-Flash-REAP-23B-A3B-IQ4_XS.gguf` | 12 GB | unsloth | ✅ Keep for now — lighter GLM fallback |
| `gemma-4-E4B-it-Q4_K_M.gguf` | 4.7 GB | unsloth | ✅ Keep — ultra-light model |
| `Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf` | 17 GB | barozp | ❌ Remove — redundant vs Qwen3.6-35B-A3B / GLM Flash |
| `Qwen3.6-28B-REAP20-A3B-IQ3_XXS.gguf` | 11 GB | barozp | ❌ Remove — low-quality quant and redundant |

**Current total:** ~106 GB  
**Still removable:** ~28 GB from the two remaining `barozp` REAP20 models.

---

## Already Deleted / No Longer Present

These models from the original audit are no longer present locally:

- `Qwen3.6-35B-A3B-IQ2_M.gguf`
- `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` from `bartowski` duplicate repo
- `Qwen3.6-27B-Q4_K_S.gguf`
- `Qwopus3.5-9B-Coder-MTP-Q4_K_M.gguf`
- `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
- `Phi-3.5-mini-instruct-Q4_K_M.gguf`

---

## Download Status

| Model | Status |
|---|---|
| `Qwen3-Coder-Next-Q4_K_M.gguf` | ✅ Downloaded |
| `GLM-4.7-Flash-Q4_K_M.gguf` | ✅ Downloaded |
| `GPT-OSS-20B-Q4_K_M` | ⏭️ Optional / not downloaded |

**GLM-5.1** remains **not practical** on 8 GB VRAM. Skip.

---

## model-presets.ini Issues Resolved

The plan's preset changes have been applied:

- Removed presets that referenced the `barozp` deletion candidates.
- Removed experimental TurboQuant overrides.
- Removed redundant `ctkd` / `ctvd` cache overrides.
- Raised coding `top-k` values into the recommended range.
- Enabled template thinking with `chat-template-kwargs = {"enable_thinking":true}` while keeping `reasoning = off`.
- Added comparison presets for `Qwen3.6-35B-A3B-UD-IQ4_XS` and `GLM-4.7-Flash-REAP-23B-A3B-IQ4_XS`.
- Added a commented optional preset for `GPT-OSS-20B-Q4_K_M`, which is not downloaded.

---

## Recommended Presets Target

```ini
version = 1

[*]
ctx-size = 65536
cache-type-k = q8_0
cache-type-v = q8_0
cache-reuse = 256
load-on-startup = false
reasoning = off
chat-template-kwargs = {"enable_thinking":true}

[1-main-coder-qwen3-coder-next]
model = /home/yuriteixeira/Workspaces/local/services/llama/models/unsloth/Qwen3-Coder-Next-Q4_K_M.gguf
temp = 0.6
top-p = 0.9
top-k = 80

[2-smart-qwen3-6-35b-a3b]
model = /home/yuriteixeira/Workspaces/local/services/llama/models/unsloth/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
temp = 0.6
top-p = 0.95
top-k = 80

[3-glm-flash]
model = /home/yuriteixeira/Workspaces/local/services/llama/models/unsloth/GLM-4.7-Flash-Q4_K_M.gguf
temp = 0.6
top-p = 0.92
top-k = 80

[4-fast-qwen3-5-9b]
model = /home/yuriteixeira/Workspaces/local/services/llama/models/unsloth/Qwen3.5-9B-Q4_K_M.gguf
temp = 0.6
top-p = 0.9
top-k = 60

[5-lightweight-gemma]
model = /home/yuriteixeira/Workspaces/local/services/llama/models/unsloth/gemma-4-E4B-it-Q4_K_M.gguf
temp = 0.5
top-p = 0.95
top-k = 40
```

---

## VRAM Reality Check for RTX 3070 (8 GB)

| Model | File Size | Expected Fit |
|---|---:|---|
| Gemma-4-E4B | 4.7 GB | Good full-GPU + useful KV cache |
| Qwen3.5-9B | 5.3 GB | Good, but context may need tuning |
| Qwen3-Coder-Next | 6.3 GB | Good primary coding candidate |
| GLM-4.7-Flash | 18 GB | CPU offload likely; slower |
| Qwen3.6-35B-A3B | 17 GB | CPU offload likely; slower |

---

## Action Items

1. **Skipped by request:** Delete remaining obsolete models (~28 GB):
   - `models/barozp/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf`
   - `models/barozp/Qwen3.6-28B-REAP20-A3B-IQ3_XXS.gguf`
2. **Done:** Update `model-presets.ini` to remove `barozp` presets and add presets for:
   - `Qwen3-Coder-Next-Q4_K_M.gguf`
   - `GLM-4.7-Flash-Q4_K_M.gguf`
3. **Ready to test:** NL vs XS for `Qwen3.6-35B-A3B`; comparison presets are available.
4. **Ready to test:** GLM main vs REAP XS; comparison presets are available.
5. **Skipped by request:** Optional `GPT-OSS-20B-Q4_K_M` download; a commented preset was added instead.

---

## Reference URLs

- `https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf?download=true`
- `https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf?download=true`
- `https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-Q4_K_M.gguf?download=true`
- `https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf?download=true`
- `https://huggingface.co/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF/resolve/main/GLM-4.7-Flash-REAP-23B-A3B-IQ4_XS.gguf?download=true`

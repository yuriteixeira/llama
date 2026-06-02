# PLAN.md — Auto-calculate `ngl` and `n-cpu-moe` in Profile View

## Goal

Add a profile editor action that calculates safe, model-aware values for:

- `gpu_layers` / `--n-gpu-layers` (`ngl`)
- `n_cpu_moe` / `--n-cpu-moe`

The calculation should use the selected GGUF model's size, layer count, tensor metadata when available, and whether the model actually supports MoE.

## Current touchpoints

- Profile action UI: `scripts/bench_buun_tui/tui.py`
  - `PROFILE_ACTIONS`
  - `profile_action_lines()`
  - `handle_profile_editor_selection()`
- Model metadata: `scripts/bench_buun_tui/models.py`
  - `model_metadata()` already reads `general.architecture`, `general.name`, and `*.block_count`.
- Profile fields: `scripts/bench_buun_tui/profile.py`
  - `gpu_layers`, `n_cpu_moe`
- Command rendering: `scripts/bench_buun_tui/commands.py`
  - already maps profile values to `--n-gpu-layers` and `--n-cpu-moe`.

## Proposed files/changes

- Add `scripts/bench_buun_tui/offload.py`
  - model capability detection
  - tensor/layer size collection
  - VRAM budget detection
  - `gpu_layers` + `n_cpu_moe` calculation
  - preview/explanation data model
- Update `scripts/bench_buun_tui/tui.py`
  - add profile actions: **Auto-calculate GPU/MoE offload** and **Smoke-test current profile load**
  - show a preview before applying
  - optionally offer a smoke test immediately after applying auto-offload values
- Optionally extend `scripts/bench_buun_tui/models.py`
  - expose richer GGUF metadata helpers if reused elsewhere.

## Calculation inputs

For the selected `profile.model`:

1. GGUF metadata:
   - `general.name`
   - `general.architecture`
   - `*.block_count`
   - any keys indicating experts/MoE, for example `*.expert_count`, `*.expert_used_count`, or architecture/name containing `moe`, `mixtral`, `qwen3moe`, `a3b`.
2. GGUF tensor sizes when available:
   - group tensors by block/layer name, e.g. `blk.<n>.*`.
   - identify expert tensors by names containing expert markers such as `expert`, `exps`, or MoE FFN expert tensor patterns.
3. File size fallback:
   - if tensor-level sizing fails, estimate per-layer weight bytes as `model_size_bytes / block_count` with a conservative overhead reserve.
4. GPU budget:
   - prefer `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`.
   - fallback to project target usable CUDA memory: RTX 3070, about 7836 MiB as reported by llama.cpp.
   - use currently free VRAM as the practical budget when available, not nominal board memory.
   - budget only for model-runtime needs, not a generic desktop/OS reserve because this is a headless server.
   - subtract estimated runtime allocations: KV cache for the desired max context, CUDA/compute workspace, graph overhead, late CUDA allocation slack, and batch/ubatch scratch buffers.
5. CPU/RAM budget:
   - read `/proc/meminfo` for `MemAvailable`; do not subtract a fixed OS reserve beyond memory that is already unavailable.
   - estimate CPU-resident bytes as all non-GPU layers plus MoE expert tensors moved by `n_cpu_moe`.
   - subtract model-runtime RAM allocations: CPU KV/cache portions if any, mmap/page-cache pressure, non-GPU compute buffers, and server/runtime scratch.
   - account for mmap behavior: with mmap enabled, CPU-resident weights are backed by the GGUF file but still need page cache/RSS headroom; with `no_mmap=true`, treat them as stronger RAM pressure.
6. Desired max context size:
   - use the current profile's largest `depths` value as the benchmark context target.
   - for server-oriented calculation, allow a future explicit target such as `CTX_SIZE`/`--ctx-size`; default to `262144` for buun if no profile depth is set.

## Calculation behavior

### Non-MoE / dense models

- Set `n_cpu_moe = unset` because `--n-cpu-moe` is not useful for dense models.
- Choose the largest `gpu_layers` that fits the estimated GPU weight budget; all remaining layers are intentionally left on CPU/RAM.
- Verify the CPU/RAM side of the split also fits `MemAvailable` after model-runtime allocations for the desired max context.
- If all layers fit, use a full-offload value:
  - preferably `block_count + 1` to include the output layer where llama.cpp supports that convention, or keep existing `99` convention if the UI wants to preserve it.
- If metadata is incomplete, show an error/low-confidence preview instead of silently applying.

### MoE models

- Prefer keeping `gpu_layers` high and offloading only MoE expert layers to CPU/RAM via `n_cpu_moe`.
- Search candidate pairs:
  - `gpu_layers` from full offload down to partial offload.
  - `n_cpu_moe` from `0` up to `block_count`.
- Estimate GPU weight use as:
  - all offloaded layer tensors
  - minus expert tensor bytes for the first `n_cpu_moe` MoE layers when those layers would otherwise be GPU-resident.
- Estimate CPU/RAM weight use as:
  - all layers above `gpu_layers` that remain CPU-resident.
  - plus expert tensors moved out of GPU by `n_cpu_moe`.
  - plus estimated runtime/page-cache allocations needed for the desired max context.
- Pick the best fitting pair by priority:
  1. fits inside VRAM budget,
  2. also fits available CPU/RAM budget,
  3. maximizes `gpu_layers`,
  4. minimizes `n_cpu_moe`,
  5. leaves enough headroom for estimated model-runtime allocations; avoid arbitrary fixed reserves unrelated to the run.
- If no pair fits safely, choose the safest partial CPU/GPU split and set `n_cpu_moe` to the maximum useful value, then mark confidence low and explain whether the limiting resource is VRAM or RAM.

## TUI workflow

Add a new action in the profile editor near the existing actions:

```text
▶ Auto-calculate GPU/MoE offload
```

Behavior:

1. Require `profile.model` to be selected.
2. Prompt for the desired max context used for memory sizing, defaulting to the largest profile `depths` value.
3. Read model metadata and calculate a recommendation.
4. Show a confirmation preview:
   - model name/path
   - architecture
   - detected layer count
   - detected MoE support: yes/no/unknown
   - model size, VRAM budget, RAM budget, and estimated GPU/CPU split
   - current `gpu_layers` → recommended `gpu_layers`
   - current `n_cpu_moe` → recommended `n_cpu_moe`
   - confidence and explanation
5. Apply only after Enter confirmation; `q`/Esc cancels.
6. Keep all unrelated profile fields unchanged.

## Validation and edge cases

- If `block_count` is missing or invalid, do not apply automatically.
- If MoE support is false, always recommend `n_cpu_moe = unset`.
- If MoE support is unknown, recommend conservative `gpu_layers` and leave `n_cpu_moe` unchanged or `unset`; clearly show low confidence.
- Clamp values to valid integer ranges:
  - `gpu_layers >= 0`
  - `n_cpu_moe >= 0`
  - `n_cpu_moe <= block_count`
- Prefer exact tensor sizing; use file-size/layer-count only as fallback and label confidence accordingly.
- If the CPU side of the split exceeds available `MemAvailable` after context/runtime allocations, lower `gpu_layers` only if it helps VRAM but mark the recommendation unsafe/low-confidence; do not present it as a clean fit.

## Implementation phases

1. **Offload calculator module**
   - Add `offload.py` with pure functions for capability detection, tensor sizing, VRAM budget, and recommendation selection.
2. **TUI integration**
   - Add the profile actions, help text, preview screen, confirmation, apply step, and smoke-test flow.
3. **Manual verification**
   - Test current Qwen3.6 MoE GGUF on RTX 3070 target.
   - Verify dense-model behavior with any non-MoE GGUF if available.
   - Confirm rendered bench/server commands include the calculated values correctly.
4. **Docs**
   - Briefly document the action and note that it is a safe estimate, not a replacement for benchmarking.

## Open questions

1. Should full offload be written as `block_count + 1`, exact `block_count`, or existing convention `99`?
2. Should the prompted desired max context also be saved as a profile field in the future?
3. What exact CUDA/workspace/KV overhead formula should be used for buun turbo KV types?
4. Should `no_mmap` alter the recommendation automatically when RAM pressure is high, or should it only affect the estimate?

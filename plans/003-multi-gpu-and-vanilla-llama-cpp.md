# Plan 003: Multi-GPU Strategy + Migration to Vanilla llama.cpp

## Goal

Migrate from the `buun` llama.cpp fork to upstream `ggml-org/llama.cpp` while preparing a two-GPU strategy:

- **Current effective setup**: only the new GPU is active, so all models must continue to use `CUDA0`.
- **Future setup**: when the old GPU is back online, larger models should stay on the new GPU and smaller models may move to the old GPU.
- Keep the single OpenAI-compatible router endpoint using `llama-server --models-preset`.

## Current Findings

### buun vs upstream

Current local buun checkout:

```txt
87c351d28 2026-06-03 cuda: add fused asymmetric turbo8-K/turbo4-V flash-attn kernel
```

Current upstream checked during analysis:

```txt
be4a6a63e 2026-06-23 server : check draft context creation error (#24922)
```

Divergence from common ancestor:

```txt
buun-only commits:      17
upstream-only commits:  362
```

Upstream already has the router workflow:

- `--models-preset`
- `--models-max`
- per-model preset loading
- router server mode

The main buun-only feature currently used by this repo is the extra Turbo/TCQ KV cache types:

```txt
turbo2, turbo3, turbo4, turbo8, turbo3_tcq, turbo2_tcq
```

Upstream supports only:

```txt
f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
```

### Arch Linux package status

No official Arch repo `llama.cpp` package was found in configured repos.

AUR package `llama.cpp-cuda-essentials-only` does compile with CUDA:

```bash
-DGGML_CUDA=ON
-DGGML_CUDA_FA=ON
-DGGML_CUDA_FA_ALL_QUANTS=ON
-DGGML_CUDA_GRAPHS=ON
-DGGML_NATIVE=ON
```

However, it does not explicitly set:

```bash
-DCMAKE_CUDA_ARCHITECTURES="86;120"
```

Therefore, prefer a local source build for explicit support of both GPUs.

## Target Architecture

Keep one router service:

```txt
client/pi extension -> http://host:8080 -> llama-server router -> per-model child server
```

Use per-model device assignment in `model-presets.ini`, but keep all effective assignments on `CUDA0` until the old GPU is back online:

```ini
[small-model]
; Future: this may move to CUDA1 when the old 3070 is back online.
device = CUDA0
model = /path/to/small.gguf
n-gpu-layers = all

[big-model]
device = CUDA0
model = /path/to/big.gguf
n-gpu-layers = all
```

Current expected assignment:

```txt
CUDA0 = new 5060 Ti 16GB  -> all models
```

Future expected assignment after both GPUs are installed and verified:

```txt
CUDA0 = new 5060 Ti 16GB  -> larger models
CUDA1 = old 3070 8GB      -> smaller models that fit comfortably
```

Verify actual ordering with:

```bash
versions/upstream/src/build/bin/llama-server --list-devices
```

## Implementation Steps

### 1. Add upstream llama.cpp source tree

Keep buun as a fallback during migration.

```bash
mkdir -p versions/upstream
git submodule add https://github.com/ggml-org/llama.cpp.git versions/upstream/src
```

If already registered or after cloning the parent repo:

```bash
git submodule update --init --recursive versions/upstream/src
```

Optional: pin the submodule to a known-good commit after testing.

### 2. Add an upstream build script

Create:

```txt
versions/upstream/build-upstream-llama-cpp.sh
```

Build options should include:

```bash
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES="86;120"
-DGGML_CUDA=ON
-DGGML_CUDA_FA=ON
-DGGML_CUDA_FA_ALL_QUANTS=ON
-DGGML_CUDA_GRAPHS=ON
-DGGML_NATIVE=ON
-DLLAMA_BUILD_TESTS=OFF
```

Rationale:

- `86` supports RTX 3070.
- `120` supports the new RTX 5060 Ti class GPU.
- `GGML_CUDA_FA_ALL_QUANTS=ON` keeps quantized KV cache support broad.

### 3. Build upstream

```bash
./versions/upstream/build-upstream-llama-cpp.sh
```

Verify binary:

```bash
./versions/upstream/src/build/bin/llama-server --version
./versions/upstream/src/build/bin/llama-server --list-devices
```

### 4. Split presets by backend instead of mutating buun-specific config

Keep the buun-specific preset file intact and create a separate upstream-compatible preset file.

Recommended files:

```txt
model-presets.buun.ini       # existing buun/turbo/TCQ-oriented presets
model-presets.upstream.ini   # vanilla upstream llama.cpp presets
model-presets.ini            # optional symlink or active default, if desired
```

Migration approach:

1. Copy the current presets to the buun-specific file:

   ```bash
   cp model-presets.ini model-presets.buun.ini
   ```

2. Create the upstream file from the buun file:

   ```bash
   cp model-presets.buun.ini model-presets.upstream.ini
   ```

3. Edit only `model-presets.upstream.ini` to remove buun-only cache types.

Current buun presets use `turbo4` for some models:

```ini
cache-type-k = turbo4
cache-type-v = turbo4
```

Upstream-compatible replacement, conservative first pass:

```ini
cache-type-k = q8_0
cache-type-v = q8_0
```

If VRAM is tight, benchmark alternatives in `model-presets.upstream.ini` only:

```ini
cache-type-k = q4_0
cache-type-v = q4_0
```

or:

```ini
cache-type-k = iq4_nl
cache-type-v = iq4_nl
```

Do not remove buun `turbo*` settings from `model-presets.buun.ini`. Prefer per-model tuning in the upstream file based on VRAM usage and quality.

### 5. Add per-model GPU placement, but keep the active placement on CUDA0

For now, only the new GPU is active. Run:

```bash
./versions/upstream/src/build/bin/llama-server --list-devices
```

Expected current result:

```txt
CUDA0 = new 5060 Ti 16GB
```

Then update `model-presets.upstream.ini` so every model explicitly uses `CUDA0`. Add comments on small models that may later move to `CUDA1`, but do not make that split effective yet.

Optionally mirror the `device = CUDA0` comments into `model-presets.buun.ini`, but do not alter its buun-only cache settings.

Current mapping:

```ini
[1-smart-qwen3-6-35b-a3b]
device = CUDA0

[2-glm-flash]
device = CUDA0

[3-fast-qwen3-5-9b]
; Future: may move to CUDA1 when the old 3070 is back online.
device = CUDA0

[4-lightweight-gemma]
; Future: may move to CUDA1 when the old 3070 is back online.
device = CUDA0
```

Future mapping after both GPUs are installed and verified:

```ini
[1-smart-qwen3-6-35b-a3b]
device = CUDA0

[2-glm-flash]
device = CUDA0

[3-fast-qwen3-5-9b]
device = CUDA1

[4-lightweight-gemma]
device = CUDA1
```

Adjust based on actual memory needs and `--list-devices` ordering.

### 6. Remove global device forcing from the service

The current service passes a global device:

```bash
--device "$LLAMA_DEVICE"
```

In router mode, CLI options are merged into child model presets. This can override per-model `device = ...` entries.

Update `llama-server@.service` so router startup does not globally force a device.

Target `ExecStart` shape:

```bash
ExecStart=/bin/sh -lc 'cd "$LLAMA_WORKDIR" && exec "$LLAMA_BIN" --host "$LLAMA_HOST" --port "$LLAMA_PORT" --models-preset "$LLAMA_MODELS_PRESET" --models-max "$LLAMA_MODELS_MAX" --parallel "$LLAMA_PARALLEL" --metrics'
```

Keep CUDA readiness probing, but consider changing the probe from one `LLAMA_DEVICE` to a generic CUDA-device check or an expected-device-list check.

Also make sure `LLAMA_MODELS_PRESET` is configurable during install so the same service template can run either:

```txt
LLAMA_MODELS_PRESET=./model-presets.upstream.ini
```

or rollback to:

```txt
LLAMA_MODELS_PRESET=./model-presets.buun.ini
```

### 7. Make installer support upstream binary

Update install flow so it can install the service using:

```txt
LLAMA_BIN=/home/yuriteixeira/Workspaces/local/services/llama/versions/upstream/src/build/bin/llama-server
LLAMA_MODELS_PRESET=./model-presets.upstream.ini
```

Options:

1. Keep `scripts/install-system-service.sh --bin <path>` and pass the upstream binary path.
2. Add a new installer option such as `--models-preset <path>` so the env file does not hard-code `./model-presets.ini`.
3. Add a convenience option or variable for selecting `buun` vs `upstream`.

Keep buun binary and `model-presets.buun.ini` usable for rollback.

### 8. Choose `models-max`

For one loaded model at a time:

```txt
LLAMA_MODELS_MAX=1
```

Future option, once both GPUs are active, for one model on each GPU simultaneously:

```txt
LLAMA_MODELS_MAX=2
```

Recommended current setup while only the new GPU is active:

```txt
LLAMA_MODELS_MAX=1
LLAMA_PARALLEL=1
```

Future multi-GPU setup after the old GPU is back online:

```txt
LLAMA_MODELS_MAX=2
LLAMA_PARALLEL=1
```

That future setting allows one big model on the new GPU and one small model on the old GPU to stay loaded concurrently.

### 9. Manual validation before service cutover

Stop current service:

```bash
./manage-server.sh stop
```

Run upstream manually with the current single-active-GPU configuration:

```bash
./versions/upstream/src/build/bin/llama-server \
  --host 0.0.0.0 \
  --port 8080 \
  --models-preset ./model-presets.upstream.ini \
  --models-max 1 \
  --parallel 1 \
  --metrics
```

In another terminal:

```bash
watch -n 1 nvidia-smi
```

Test each configured model from the client/pi extension and verify:

- All models load on `CUDA0`, the currently active 16GB GPU.
- Small-model presets only contain comments about future `CUDA1` placement; they should not actually use `CUDA1` yet.
- Router unload/load behavior works.
- Responses are correct with upstream KV cache types.

After the old GPU is back online, repeat validation with `--models-max 2` and verify:

- Big models load on the 16GB GPU.
- Small models selected for migration load on the 8GB GPU.
- No unexpected cross-GPU memory usage.

### 10. Cut over systemd service

After manual validation:

```bash
./manage-server.sh install \
  --bin "$PWD/versions/upstream/src/build/bin/llama-server" \
  --models-preset ./model-presets.upstream.ini \
  --models-max 1
./manage-server.sh restart
./manage-server.sh status
```

Check logs:

```bash
./manage-server.sh logs
```

### 11. Rollback plan

Keep buun checkout and binary intact.

Rollback by reinstalling service with buun binary and buun presets:

```bash
./manage-server.sh install \
  --bin "$PWD/versions/buun/src/build/bin/llama-server" \
  --models-preset ./model-presets.buun.ini \
  --models-max 1
./manage-server.sh restart
```

No preset conversion should be needed during rollback because `model-presets.buun.ini` keeps the buun-specific cache types:

```ini
cache-type-k = turbo4
cache-type-v = turbo4
```

## Required script updates

### `scripts/audit-models.sh`

Update the audit script so it understands multiple preset files instead of assuming only `./model-presets.ini`.

Current behavior:

```txt
PRESETS="$ROOT_DIR/model-presets.ini"
```

Target behavior:

- Audit all `model-presets.*.ini` files by default, including:

  ```txt
  model-presets.buun.ini
  model-presets.upstream.ini
  ```

  Skip missing files gracefully.

- Add optional CLI support for explicit preset files, for example:

  ```bash
  scripts/audit-models.sh --preset model-presets.upstream.ini
  scripts/audit-models.sh --preset model-presets.buun.ini --preset model-presets.upstream.ini
  ```

- Track preset presence per file, not as one generic `in ./model-presets.ini` column.

Suggested output columns:

```txt
name | in ./models | in ./models.txt | in buun presets | in upstream presets | in default presets
```

or, if dynamic columns are simpler:

```txt
name | in ./models | in ./models.txt | model-presets.buun.ini | model-presets.upstream.ini | model-presets.ini
```

- Keep extracting active `model = ...` lines from each preset file.
- Continue ignoring commented-out model entries.
- Include models referenced by any preset file in the union of audited models.

This prevents the migration from making `audit-models.sh` blind to either the rollback buun presets or the new upstream presets.


## Execution Checklist

This checklist is the execution-focused companion to the strategy above.

### Phase 1: Preserve buun setup

```bash
cp model-presets.ini model-presets.buun.ini
cp model-presets.buun.ini model-presets.upstream.ini
```

Do not remove buun-specific settings such as:

```ini
cache-type-k = turbo4
cache-type-v = turbo4
```

Only `model-presets.upstream.ini` should be edited for upstream compatibility.

### Phase 2: Add upstream llama.cpp as a submodule

```bash
mkdir -p versions/upstream
git submodule add https://github.com/ggml-org/llama.cpp.git versions/upstream/src
```

If already registered:

```bash
git submodule update --init --recursive versions/upstream/src
git -C versions/upstream/src pull --ff-only
```

Create `versions/upstream/build-upstream-llama-cpp.sh` with CUDA support for both current and future GPUs:

```bash
-DCMAKE_CUDA_ARCHITECTURES="86;120"
-DGGML_CUDA=ON
-DGGML_CUDA_FA=ON
-DGGML_CUDA_FA_ALL_QUANTS=ON
-DGGML_CUDA_GRAPHS=ON
-DGGML_NATIVE=ON
-DLLAMA_BUILD_TESTS=OFF
```

Build and verify:

```bash
./versions/upstream/build-upstream-llama-cpp.sh
./versions/upstream/src/build/bin/llama-server --version
./versions/upstream/src/build/bin/llama-server --list-devices
```

Expected for current phase:

```txt
CUDA0 = NVIDIA GeForce RTX 5060 Ti
```

### Phase 3: Convert upstream presets

Edit only `model-presets.upstream.ini`.

Replace upstream-incompatible values:

```ini
cache-type-k = turbo4
cache-type-v = turbo4
```

with conservative upstream-compatible defaults:

```ini
cache-type-k = q8_0
cache-type-v = q8_0
```

Remove or avoid buun-only preset options that upstream rejects. During validation, upstream rejected `logits-all` in router presets.

For now, every active model should have:

```ini
device = CUDA0
```

For small models that may later move to the old GPU, add comments only:

```ini
; Future: may move to CUDA1 when the old RTX 3070 is back online.
device = CUDA0
```

Do not set any active model to `CUDA1` yet.

### Phase 4: Update service and install scripts

Update `scripts/install-system-service.sh` to support:

```bash
--models-preset <path>
```

The env file should write:

```txt
LLAMA_MODELS_PRESET=<chosen preset path>
```

instead of hard-coding:

```txt
LLAMA_MODELS_PRESET=./model-presets.ini
```

Update `llama-server@.service` so `ExecStart` does not pass global `--device "$LLAMA_DEVICE"`.

Target shape:

```bash
ExecStart=/bin/sh -lc 'cd "$LLAMA_WORKDIR" && exec "$LLAMA_BIN" --host "$LLAMA_HOST" --port "$LLAMA_PORT" --models-preset "$LLAMA_MODELS_PRESET" --models-max "$LLAMA_MODELS_MAX" --parallel "$LLAMA_PARALLEL" --metrics'
```

Reason: per-model `device = CUDA0` / future `CUDA1` should come from presets, not router-level CLI args.

For now, readiness can still check `CUDA0`. Future improvement: support expected device lists, for example:

```txt
LLAMA_EXPECTED_DEVICES=CUDA0,CUDA1
```

Do not require `CUDA1` until the old GPU is back online.

### Phase 5: Update model audit script

Update `scripts/audit-models.sh` to:

- support multiple preset files
- default to all `model-presets.*.ini`
- skip missing files gracefully
- support repeated explicit arguments:

  ```bash
  scripts/audit-models.sh --preset model-presets.upstream.ini
  scripts/audit-models.sh --preset model-presets.buun.ini --preset model-presets.upstream.ini
  ```

- report preset presence per file:

  ```txt
  name | in ./models | in ./models.txt | model-presets.buun.ini | model-presets.upstream.ini
  ```

Keep behavior:

- include models from any preset file in the union
- ignore commented-out `model = ...` entries
- extract basename from active `model = ...` paths

### Phase 6: Manual upstream validation

Stop current service:

```bash
./manage-server.sh stop
```

Run upstream manually:

```bash
./versions/upstream/src/build/bin/llama-server \
  --host 0.0.0.0 \
  --port 8080 \
  --models-preset ./model-presets.upstream.ini \
  --models-max 1 \
  --parallel 1 \
  --metrics
```

Monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

Validate:

- all models load on `CUDA0`
- no active preset references `CUDA1`
- upstream accepts all preset options
- router load/unload works
- output quality is acceptable with upstream KV cache types

### Phase 7: Systemd cutover

Install service using upstream binary and upstream presets:

```bash
./manage-server.sh install \
  --bin "$PWD/versions/upstream/src/build/bin/llama-server" \
  --models-preset ./model-presets.upstream.ini \
  --models-max 1
```

Restart and inspect:

```bash
./manage-server.sh restart
./manage-server.sh status
./manage-server.sh logs
```

### Phase 8: Rollback path

Rollback to buun binary and buun presets:

```bash
./manage-server.sh install \
  --bin "$PWD/versions/buun/src/build/bin/llama-server" \
  --models-preset ./model-presets.buun.ini \
  --models-max 1
./manage-server.sh restart
```

Because `model-presets.buun.ini` is preserved, no cache-type edits should be needed for rollback.

### Phase 9: Future old-GPU reactivation

Only after the old GPU is physically back online:

1. Verify device ordering:

   ```bash
   ./versions/upstream/src/build/bin/llama-server --list-devices
   ```

   Expected future shape:

   ```txt
   CUDA0 = new RTX 5060 Ti 16GB
   CUDA1 = old RTX 3070 8GB
   ```

2. Move selected small models to `CUDA1` in `model-presets.upstream.ini` only if they fit comfortably.

3. Allow two loaded models:

   ```bash
   ./manage-server.sh install \
     --bin "$PWD/versions/upstream/src/build/bin/llama-server" \
     --models-preset ./model-presets.upstream.ini \
     --models-max 2
   ./manage-server.sh restart
   ```

4. Validate split with `watch -n 1 nvidia-smi`.

## Benchmark Checklist

For each major model, record:

- GPU device used
- context size
- KV cache type
- load time
- prompt processing tok/s
- generation tok/s
- peak VRAM
- subjective output quality

Suggested cache-type comparison for large models:

1. `q8_0/q8_0`
2. `q4_0/q4_0`
3. `iq4_nl/iq4_nl`
4. buun fallback: `turbo4/turbo4`

## Risks / Watchpoints

- CUDA device ordering may change after adding/removing GPUs or driver changes.
- Global `--device` in router service can override per-model device placement.
- Upstream does not support buun `turbo*` KV cache types.
- Lower-bit upstream KV cache may affect output quality more than buun TCQ.
- Do not enable effective `CUDA1` placement until the old GPU is physically back online and visible in `--list-devices`.
- `models-max = 2` can increase total RAM usage even if VRAM is split across GPUs.
- Mixed GPU generations may not improve single-model performance; this plan uses separate GPUs for separate models, not split-mode multi-GPU inference.

## Success Criteria

- Upstream `llama-server` builds locally with CUDA support for both GPUs.
- Current phase: `--list-devices` detects the active new GPU as `CUDA0`.
- Current phase: all presets explicitly use `CUDA0`.
- Current phase: small-model presets document future `CUDA1` eligibility only as comments.
- Future phase: `--list-devices` detects both GPUs.
- Future phase: `model-presets.ini` can pin selected models to specific CUDA devices.
- One router endpoint remains available on port `8080`.
- Large models reliably load on the new 16GB GPU.
- Future phase: selected small models reliably load on the old 8GB GPU.
- Service starts at boot without requiring buun.
- buun remains available as rollback only.

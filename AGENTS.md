# Local LLM Inference Service

## Overview

This repository contains a minimal local LLM inference setup using the **llama.cpp**. Both upstream and forks (that implement the turbo cache types, which is explored in the tutorials found in https://www.youtube.com/@Codacus) are provided as submodules, which will be the engine that serve large language models as an API server on local hardware. 

**IMPORTANT**: Fork it right away because it will likely not work for you out of the box. Use it as an inspiration. All I hope is to save you some research time to curate a solution that fits your needs.


## Assumptions & Requirements

- Arch Linux with systemd support (but easily adaptable to any linux distro, just change the package manager calls)
- Git with submodule support
- NVIDIA GPU with CUDA support (tested on RTX 5060 Ti)

Tuned for:
- **Headless server**: Resources dedicated to only run `llama-server` with 1 model loaded at a time (even though models are loaded dinamically by clients)
- **GPU:** NVIDIA RTX 5060 Ti (16GB VRAM)
- **VRAM:** ~16.3 GB device memory
- **RAM:** 16 GB system RAM
- **CPU:** 12 threads available


## Building & Using

This project supports three llama.cpp forks:

- **upstream fork**: `./versions/upstream/src/` - vanilla ggml-org/llama.cpp with router workflow and standard quantization types
- **TurboQuant+ fork**: `./versions/turboquant/src/` - TheTom/llama-cpp-turboquant with TurboQuant+ KV cache types (turbo2, turbo3, turbo4) and TQ3_1S/TQ4_1S weight quantization
- **buun fork**: `./versions/buun/src/` - includes Turbo KV cache types (turbo2, turbo3, turbo4, turbo8, turbo3_tcq, turbo2_tcq)

The installer is pre-configured to use the upstream fork by default, but you can build and use buun or TurboQuant+ instead.

**Cloning:**

```bash
git clone --recurse-submodules git@github.com:yuriteixeira/llama.git
cd llama
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

**Updating a fork:**

```bash
git -C versions/buun/src pull origin main  # or master

# or upstream
git -C versions/upstream/src pull origin master

# or TurboQuant+
git -C versions/turboquant/src pull origin feature/turboquant-kv-cache
```

- buun fork: https://github.com/spiritbuun/buun-llama-cpp
- upstream fork: https://github.com/ggml-org/llama.cpp
- TurboQuant+ fork: https://github.com/TheTom/llama-cpp-turboquant

**Building:**

```bash
# Build upstream (pre-configured default)
./versions/upstream/build-upstream-llama-cpp.sh

# Build buun fork
./versions/buun/build-buun-llama-cpp.sh

# Build TurboQuant+ fork
./versions/turboquant/build-turboquant-llama-cpp.sh
```

**Starting the service:**

`llama-server.service` is a system-level systemd service intended to start at boot on a headless machine.
It still runs the `llama-server` process as `yuriteixeira`, not as root.

To make managing it easier, use `./manage-server.sh`:

```bash
./manage-server.sh install
./manage-server.sh enable
./manage-server.sh status
```

The install/enable actions use `sudo` to:

- enable and start `nvidia-persistenced.service`
- copy `./llama-server.service` to `/etc/systemd/system/llama-server.service`
- reload systemd
- enable/start `llama-server.service`

Check logs with:

```bash
./manage-server.sh logs
```

The service uses these boot-safety choices:

- `After=network.target`, not `network-online.target`, because the server binds `0.0.0.0` and should not block boot waiting for every network interface.
- `Wants=nvidia-persistenced.service` and `After=nvidia-persistenced.service` so CUDA driver state is kept initialized.
- A bounded CUDA readiness `ExecStartPre` check using `nvidia-smi`, `llama-server --list-devices`, and `/usr/bin/timeout`.
- `TimeoutStartSec=30min` so cold GPU initialization can take time without systemd killing the startup.

To simulate the boot-relevant service path without rebooting:

```bash
./manage-server.sh test-start
```

This stops `llama-server` and `nvidia-persistenced`, starts only `llama-server`, verifies that NVIDIA persistence is pulled in, and waits for HTTP readiness.


**Using the models:**

My recommendation is https://pi.dev + its [pi-llama-cpp](https://pi.dev/packages/pi-llama-cpp) extension, which will allow to load any models configured in the selected preset file (`./model-presets.buun.ini`, `./model-presets.upstream.ini`, or `./model-presets.turboquant.ini`).

**Model Presets:**

- `model-presets.upstream.ini` - upstream-compatible presets (uses standard quantization, requires `--models-preset` flag)
- `model-presets.buun.ini` - buun-specific presets (uses Turbo KV cache types)
- `model-presets.turboquant.ini` - TurboQuant+-compatible presets (uses `q8_0` K cache with `turbo3` V cache by default)

The systemd installer defaults to `./model-presets.upstream.ini` (upstream fork). You can override it via `--models-preset`. To use buun presets, run:

```bash
./manage-server.sh install --models-preset ./model-presets.buun.ini --bin ./versions/buun/src/build/bin/llama-server
```

To use TurboQuant+ presets, run:

```bash
./manage-server.sh install --models-preset ./model-presets.turboquant.ini --bin ./versions/turboquant/src/build/bin/llama-server
```

## Managing Models & Presets

- Models are downloaded using `./scripts/download-recommended-models.sh` based in the list of curated models in `./models.txt`
- Model presets for `llama-server` are defined in `./model-presets.upstream.ini` (upstream), `./model-presets.buun.ini` (buun), or `./model-presets.turboquant.ini` (TurboQuant+)
- The project uses a single model at a time for optimal resource utilization on dedicated headless servers
- Presets can be updated based on the models downloaded by running `./scripts/update-presets.sh`
- Before updating `./models.txt` with a new model, make sure they are not deny-listed in `./DO-NOT-DOWNLOAD-THESE-MODELS.md`. Make sure this machine's NVRAM + VRAM can fit all the necessary data to run the model correctly (the model itself, context, engine, etc).


## Pro-tips

- `./manage-server.sh logs` & `nvtop` are your best friends to understand if GPU + NVRAM is doing most of the job or if work was offloaded to the CPU+RAM. 9 out of 10, that explains slowness, specially on bigger models.

# Local LLM Inference Service

## Overview

This repository contains a minimal local LLM inference setup using the **buun llama.cpp fork** (inspired by the tutorials found in https://www.youtube.com/@Codacus) to serve large language models as an API server on local hardware. 

**IMPORTANT**: Fork it right away because it will likely not work for you out of the box. Use it as an inspiration. All I hope is to save you some research time to curate a solution that fits your needs.


## Assumptions & Requirements

- Arch Linux with systemd support (but easily adaptable to any linux distro, just change the package manager calls)
- Git with submodule support
- NVIDIA GPU with CUDA support (tested on RTX 3070)

Tuned for:
- **Headless server**: Resources dedicated to only run `llama-server` with 1 model loaded at a time (even though models are loaded dinamically by clients)
- **GPU:** NVIDIA RTX 3070 (8GB VRAM, compute capability 8.6)
- **VRAM:** ~7.7 GB device memory
- **RAM:** 16 GB system RAM
- **CPU:** 12 threads available


## Building & Using

Since this project started This project started inspired by the **buun llama.cpp fork** as a git submodule at `./versions/buun/src/`.

**Cloning:**

```bash
git clone --recurse-submodules git@github.com:yuriteixeira/llama.git
cd llama
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

**Updating the fork:**

```bash
cd versions/buun/src
git pull origin main  # or master
```

The submodule points to https://github.com/spiritbuun/buun-llama-cpp.

**Building:**

```bash
./versions/buun/build-buun-llama-cpp.sh
```

**Starting the service:**

`llama-server.service` is a user-scoped systemd service for auto-start on login.

To make managing it a bit easier, I created `./manage-server.sh`.

Run `./manage-server.sh install && ./manage-server.sh enable && ./manage-server.sh start`.

Check the logs with `./manage-server.sh logs`

The service is installed to `~/.config/systemd/user/` on first use. It auto-restarts on crash (10 s delay).


**Using the models:**

My recommendation is https://pi.dev + its [pi-llama-cpp](https://pi.dev/packages/pi-llama-cpp) extension, which will allow to load any models configured in `./model-presets.ini`

## Managing Models & Presets

- Models are downloaded using `./scripts/download-recommended-models.sh` based in the list of curated models in `./models.txt`
- Model presets for `llama-server` are defined in `./model-presets.ini`. The project uses a single model at a time for optimal resource utilization on dedicated headless servers. They can be updated based on the models downloaded if you run `./scripts/update-presets.sh`
- Before updating `./models.txt` with a new model, make sure they are not deny-listed in `./DO-NOT-DOWNLOAD-THESE-MODELS.md`. Make sure this machine's NVRAM + VRAM can fit all the necessary data to run the model correctly (the model itself, context, engine, etc).


## Pro-tips

- `./manage-server.sh logs` & `nvtop` are your best friends to understand if GPU + NVRAM is doing most of the job or if work was offloaded to the CPU+RAM. 9 out of 10, that explains slowness, specially on bigger models.

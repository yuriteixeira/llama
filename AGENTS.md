# AGENTS.md — Local LLM Inference Service

## Overview

This repository contains a minimal local LLM inference setup using the **buun llama.cpp fork** to serve large language models as an API server on local hardware. The project has been simplified to have minimal code, focusing on running LLMs to help with tasks rather than maintaining complex benchmarking tools.

**OS Assumptions:**
- Arch Linux with systemd support
- Git with submodule support
- NVIDIA GPU with CUDA support (tested on RTX 3070)


## Buun Llama CP

This project uses the **buun llama.cpp fork** as a git submodule at `./versions/buun/src/`.

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


## Systemd Services

`llama-server.service` is a user-scoped systemd service for auto-start on login through `./manage-server.sh`.

The service is installed to `~/.config/systemd/user/` on first use. It auto-restarts on crash (10 s delay).


## Models & Presets

- Models are downloaded using `./scripts/download-recommended-models.sh` based in the list of curated models in `./models.txt`
- Model presets for `llama-server` are defined in `./model-presets.ini`. The project uses a single model at a time for optimal resource utilization on dedicated headless servers. They can be updated based on the models downloaded if you run `./scripts/update-presets.sh`
- Before updating `./models.txt` with a new model, make sure they are not deny-listed in `./DO-NOT-DOWNLOAD-THESE-MODELS.md`. Make sure this machine's NVRAM + VRAM can fit all the necessary data to run the model correctly (the model itself, context, engine, etc).

## Hardware

Tuned for:
- **Headless server**: Resources dedicated to only run `llama-server` with 1 model loaded at a time (even though models are loaded dinamically by clients)
- **GPU:** NVIDIA RTX 3070 (8GB VRAM, compute capability 8.6)
- **VRAM:** ~7.7 GB device memory
- **RAM:** 16 GB system RAM
- **CPU:** 12 threads available

**Note:** Hardware specifications are specific to the development environment. Adjust model presets based on your own hardware constraints.


## Development Guidelines

- Keep it simple: the goal is to have minimal code to make the LLMs help you out


## Related Resources

- **Buun fork:** https://github.com/spiritbuun/buun-llama-cpp

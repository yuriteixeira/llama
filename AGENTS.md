# AGENTS.md — Local LLM Inference Service

## Overview

This repository contains a minimal local LLM inference setup using the **buun llama.cpp fork** to serve large language models as an API server on local hardware. The project has been simplified to have minimal code, focusing on running LLMs to help with tasks rather than maintaining complex benchmarking tools.


## Buun Llama CP

A working copy of https://github.com/spiritbuun/buun-llama-cpp live in `./versions/buun/src/`


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


## Development Guidelines

- Keep it simple: the goal is to have minimal code to make the LLMs help you out


## Related Resources

- **Buun fork:** https://github.com/spiritbuun/buun-llama-cpp

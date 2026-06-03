# AGENTS.md — Local LLM Inference Service

## Overview

This repository contains a local LLM inference setup using the **buun llama.cpp fork** to serve large language models as an API server on local hardware. It includes benchmarking tools.


## Systemd Services

`llama-server.service` is a user-scoped systemd service for auto-start on login through `./manage-server.sh`.

The service is installed to `~/.config/systemd/user/` on first use. It auto-restarts on crash (10 s delay).


## Benchmarking

### Benchmark Scripts
- `./scripts/bench-buun-tui.py` - Runs `llama-bench` after the user sets comparison params through a TUI
- `./scripts/rank-buun-results.py` — Ranks finalized buun TUI benchmark runs
- `./scripts/update-buun-model-presets.py` — Updates `versions/buun/model-presets.ini` from the best recorded buun benchmark results; also available from the TUI and run automatically after ranking updates. Fixed llama-server args can be pasted into `versions/buun/model-preset-fixed-args.txt`.

## Models

See `./models/` folder.

## Hardware Target

Tuned for:
- **Headless server**: Resources dedicated to only run `llama-server` with 1 model loaded at a time (even though models are loaded dinamically by clients)
- **GPU:** NVIDIA RTX 3070 (8GB VRAM, compute capability 8.6)
- **VRAM:** ~7.7 GB device memory
- **RAM:** 16 GB system RAM
- **CPU:** 12 threads available

## Development Guidelines

- Use Conventional Commits for commit messages, matching the existing history (for example, `feat(scope): summary`, `fix(scope): summary`). Include a concise commit body explaining what motivated the change and add links to external references when applicable.
- Clean-code & SOLID principles should be applied

## Related Resources

- **Buun fork:** https://github.com/spiritbuun/buun-llama-cpp

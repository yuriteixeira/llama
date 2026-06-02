# AGENTS.md — Local LLM Inference Service

## Overview

This repository contains a local LLM inference setup using **llama.cpp** to serve large language models as API servers on local hardware. It supports two server variants and includes benchmarking tools.


## Systemd Services

Both variants (vanilla & buun) ship as user-scoped systemd services for auto-start on login through `./manage-server.sh`

> **Note:** Both variants default to port 8080 — only run one at a time, or change the `--port` in the respective `.service` file.

Services are installed to `~/.config/systemd/user/` on first use. They auto-restart on crash (10 s delay).


## Benchmarking

### Benchmark Scripts
- `./scripts/bench-buun-tui.py` - Runs `llama-bench` after the user sets comparison params through a TUI
- `./scripts/bench-llama-suite.py` — Full llama.cpp benchmark suite
- `./scripts/run-llama-bench-matrix.py` — Matrix benchmark across configs

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

# AGENTS.md — Local LLM Inference Service

## Overview

This repository contains a local LLM inference setup using **llama.cpp** to serve large language models as API servers on local hardware. It supports two server variants and includes benchmarking tools.

## Architecture

```
llama/
├── serve.sh           # Vanilla llama.cpp launcher
├── serve-buun.sh      # Spiritbuun fork launcher (optimized)
├── models/            # GGUF model files
│   └── unsloth/       # Default: Qwen3.6-35B-A3B (IQ4_NL quantized)
├── versions/          # Server binary builds
│   ├── vanilla/       # Standard llama.cpp build
│   └── buun/          # spiritbuun/buun-llama-cpp fork
├── scripts/           # Benchmarking scripts
│   ├── bench-llama-suite.py
│   └── run-llama-bench-matrix.py
└── docs/              # Reference notes
    └── clues.md       # Moises' setup notes & community tips
```

## Server Launchers

### Vanilla (`serve.sh`)
Standard llama.cpp with default settings.

```bash
./serve.sh
```

| Option | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | HTTP port |
| `--model` | `models/unsloth/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` | GGUF model path |
| `--ctx-size` | `32768` | Context window |
| `--cache-type-k` | `q8_0` | KV cache key type |
| `--cache-type-v` | `q8_0` | KV cache value type |
| `--threads` | `12` | CPU threads for decoding |
| `--threads-batch` | `12` | CPU threads for batching |
| `--threads-http` | `4` | HTTP worker threads |
| `--batch-size` | `1024` | Prompt processing batch |
| `--ubatch-size` | `256` | Micro-batch size |
| `--fit-target` | `512` | Memory fit target (MiB) |
| `--device` | `CUDA0` | GPU device |
| `--parallel` | `1` | Request parallelism |
| `--flash-attn` | `on` | Flash attention |
| `--metrics` | — | Enable metrics endpoint |

### Buun Fork (`serve-buun.sh`)
Custom fork from [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) with MoE optimizations.

```bash
./serve-buun.sh
```

Key differences from vanilla:
| Option | Default | Description |
|---|---|---|
| `--ctx-size` | `262144` | 8× larger context window |
| `--cache-type-k` | `turbo4` | Turbo KV cache (optimized) |
| `--cache-type-v` | `turbo3` | Turbo KV cache (optimized) |
| `--n-cpu-moe` | `33` | CPU threads for MoE routing |
| `--cache-ram` | `0` | Disable RAM cache offload |
| `--no-cache-idle-slots` | — | Disable idle slot caching |

## Environment Variables

Both scripts respect these overrides:

- `SERVER_BIN` — path to custom `llama-server` binary
- `HOST` / `PORT` — network configuration
- `MODEL_FILE` — custom model path
- `CTX_SIZE` / `CTX_SIZE_BATCH` — context window
- `THREADS` / `THREADS_BATCH` / `THREADS_HTTP` — threading
- `CACHE_TYPE_K` / `CACHE_TYPE_V` — KV cache types
- `N_CPU_MOE` — MoE CPU thread count (buun only)
- `REASONING` — toggle reasoning mode (`on`/`on`)
- `MODEL` — alias for the served model

## Benchmarking

### Single Benchmark
```bash
# Using buun fork
./build/bin/llama-bench \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL \
  --n-cpu-moe 33 \
  --cache-type-k turbo4 \
  --cache-type-v turbo3

# Expected (RTX 3070, 8GB VRAM):
#  pp512:  ~530 tokens/s
#  tg128:  ~51 tokens/s
```

### Benchmark Scripts
- `scripts/bench-llama-suite.py` — Full llama.cpp benchmark suite
- `scripts/run-llama-bench-matrix.py` — Matrix benchmark across configs

## Model

- **Family:** Qwen3.6-35B-A3B (MoE — 35B total params, ~3B active)
- **Quantization:** IQ4_NL (4-bit, ~11-14 GB disk)
- **Location:** `models/unsloth/`
- **Source:** Unsloth/Qwen3.6-35B-A3B-GGUF on HuggingFace

## Hardware Target

Tuned for:
- **GPU:** NVIDIA RTX 3070 (8GB VRAM, compute capability 8.6)
- **VRAM:** ~7.7 GB device memory
- **RAM:** 16+ GB system RAM recommended
- **CPU:** 12+ threads available

## Related Resources

- **Clues / Moises' notes:** `docs/clues.md` — includes Moises' setup configuration and benchmark results from an identical GPU/RAM configuration
- **Community tips:** Reddit discussions on running Qwen3.6-35B-A3B on RTX 2070/3070 cards
- **Buun fork:** https://github.com/spiritbuun/buun-llama-cpp

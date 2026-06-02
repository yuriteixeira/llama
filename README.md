# Local LLM Inference Server

Serve large language models locally on your GPU using llama.cpp.

## Quick Start

### Vanilla Server

```bash
./serve.sh
```

### Optimized Server (Buun Fork)

```bash
./serve-buun.sh
```

Both launchers start a local API server with the default model:

> **Qwen3.6-35B-A3B** — 35B total params, 3B active (MoE), IQ4_NL quantized (~14 GB)

The server exposes an OpenAI-compatible API endpoint on **port 8080**.

## Server Modes

| Mode | Script | Context | Cache | Best For |
|---|---|---|---|---|
| **Vanilla** | `./serve.sh` | 32,768 | q8_0 | Shorter contexts, standard behavior |
| **Buun** | `./serve-buun.sh` | 262,144 | turbo4/turbo3 | Long contexts, MoE-optimized |

## Running as a Systemd Service

Run the server automatically on login using systemd user services:

```bash
# Enable + start the vanilla variant
./manage-server.sh vanilla enable
./manage-server.sh vanilla start

# Or the buun variant
./manage-server.sh buun enable
./manage-server.sh buun start

# View live logs
./manage-server.sh vanilla logs

# Stop / restart
./manage-server.sh buun stop
./manage-server.sh buun restart
```

Only one variant can run at a time (both default to port **8080**). Modify the port in the `.service` file to run both simultaneously.

For startup immediately after reboot, before you log in, enable user lingering once:

```bash
sudo loginctl enable-linger "$USER"
```

## Customization

Override defaults via environment variables:

```bash
# Use a different model
MODEL_FILE=./models/other-model.gguf ./serve-buun.sh

# Change port and host
PORT=3000 HOST=127.0.0.1 ./serve.sh

# More threads
THREADS=16 BATCH_SIZE=2048 ./serve-buun.sh

# Enable reasoning mode
REASONING=on ./serve-buun.sh
```

Or pass extra flags directly (appended to the command):

```bash
./serve.sh --jinja --webui-mcp-proxy
```

## Benchmarks

Expected performance on **RTX 3070 (8 GB VRAM)**:

| Benchmark | Tokens/sec |
|---|---|
| Prompt processing (512 tokens) | ~530 tok/s |
| Token generation (128 tokens) | ~51 tok/s |

Run benchmarks:

```bash
# Quick single run
./build/bin/llama-bench -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL \
  --n-cpu-moe 33 --cache-type-k turbo4 --cache-type-v turbo3

# Interactive buun benchmark TUI
python3 scripts/bench-buun-tui.py

# Research best-known params for models under ./models for the TUI
python3 scripts/research-best-known-params.py

# Matrix across configs
python3 scripts/run-llama-bench-matrix.py

# Full benchmark suite
python3 scripts/bench-llama-suite.py
```

The TUI can scan `./models`, invoke `pi -p`, and write researched heuristics to
`bench-configs/buun/best-known-params.ini`. While editing a profile, use
`Load best-known params for this model` to preview/apply them, or
`Show equivalent llama-server command` to inspect the serving command implied by
that profile. Treat these recommendations as starting points and benchmark them
on your hardware.

## Directory Structure

```
llama/
├── serve.sh                      # Vanilla llama.cpp launcher
├── serve-buun.sh                 # Optimized buun-llama-cpp launcher
├── manage-server.sh              # Systemd service helper
├── llama-server.service          # Systemd unit (vanilla)
├── llama-server-buun.service     # Systemd unit (buun)
├── models/                       # GGUF model storage
│   └── unsloth/
├── versions/                     # Server builds
│   ├── vanilla/
│   └── buun/
├── scripts/                      # Benchmark tools
└── docs/                         # Notes & references
    └── clues.md                  # Moises' setup notes
```

## Hardware

- **GPU:** NVIDIA RTX 3070+ (8 GB VRAM minimum)
- **RAM:** 16 GB+ system memory
- **CUDA:** Compute capability 5.0+

## License

See the llama.cpp license for licensing terms. GGUF model files are subject to their respective model licenses.

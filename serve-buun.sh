#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SERVER_BIN:-$ROOT_DIR/versions/buun/src/buun-llama-cpp/build/bin/llama-server}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8080}" \
  --model "${MODEL_FILE:-$ROOT_DIR/models/unsloth/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf}" \
  --alias "${ALIAS:-selected}" \
  --ctx-size "${CTX_SIZE:-262144}" \
  --cache-type-k "${CACHE_TYPE_K:-turbo4}" \
  --cache-type-v "${CACHE_TYPE_V:-turbo3}" \
  --threads "${THREADS:-12}" \
  --threads-batch "${THREADS_BATCH:-12}" \
  --threads-http "${THREADS_HTTP:-4}" \
  --n-cpu-moe "${N_CPU_MOE:-33}" \
  --device CUDA0 \
  --parallel 1 \
  --flash-attn on \
  --metrics \
  --cache-ram 0 \
  --reasoning off \
  --no-cache-idle-slots \
  --no-mmproj \
  "$@"

# To enable web-ui: 
# --jinja \
# --webui-mcp-proxy \

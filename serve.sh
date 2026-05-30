#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SERVER_BIN:-$ROOT_DIR/versions/vanilla/install/bin/llama-server}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8080}" \
  --model "${MODEL_FILE:-$ROOT_DIR/models/unsloth/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf}" \
  --alias "${ALIAS:-selected}" \
  --ctx-size "${CTX_SIZE:-32768}" \
  --cache-type-k "${CACHE_TYPE_K:-q8_0}" \
  --cache-type-v "${CACHE_TYPE_V:-q8_0}" \
  --threads "${THREADS:-12}" \
  --threads-batch "${THREADS_BATCH:-12}" \
  --threads-http "${THREADS_HTTP:-4}" \
  --batch-size "${BATCH_SIZE:-1024}" \
  --ubatch-size "${UBATCH_SIZE:-256}" \
  --fit-target "${FIT_TARGET_MIB:-512}" \
  --device CUDA0 \
  --parallel "${PARALLEL:-1}" \
  --flash-attn "${FLASH_ATTN:-on}" \
  --metrics \
  --reasoning "${REASONING:-off}" \
  --no-mmproj \
  "$@"

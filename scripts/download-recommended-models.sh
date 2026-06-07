#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"

MAX_JOBS="${MAX_JOBS:-3}"
DOWNLOADS=()

fetch() {
  local dest="$1"
  local url="$2"
  DOWNLOADS+=("$dest"$'\t'"$url")
}

download_one() {
  local spec="$1"
  local dest="${spec%%$'\t'*}"
  local url="${spec#*$'\t'}"

  mkdir -p "$(dirname "$dest")"
  echo "==> $(basename "$dest")"
  wget --continue --show-progress --output-document="$dest" "$url"
}

run_downloads() {
  echo "Queued ${#DOWNLOADS[@]} model downloads (MAX_JOBS=$MAX_JOBS)."
  printf '%s\0' "${DOWNLOADS[@]}" \
    | xargs -0 -n 1 -P "$MAX_JOBS" bash -c 'download_one "$1"' _
}

export -f download_one

# Qwen3.6-35B-A3B MoE — best all-around agentic coding model
fetch \
  "$MODELS_DIR/unsloth/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf" \
  "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf?download=true"

# Qwen3.6-35B-A3B MoE — lighter quant, more KV-cache headroom
fetch \
  "$MODELS_DIR/unsloth/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
  "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf?download=true"

# GLM-4.7-Flash — SWE-Bench leader, 30B MoE / 3B active params
fetch \
  "$MODELS_DIR/unsloth/GLM-4.7-Flash-Q4_K_M.gguf" \
  "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf?download=true"

# GLM-4.7-Flash-REAP-IQ4_XS — lighter REAP variant for slower hardware
fetch \
  "$MODELS_DIR/unsloth/GLM-4.7-Flash-REAP-23B-A3B-IQ4_XS.gguf" \
  "https://huggingface.co/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF/resolve/main/GLM-4.7-Flash-REAP-23B-A3B-IQ4_XS.gguf?download=true"

run_downloads 

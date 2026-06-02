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

# Category 1: best coding models that fit fully in 8GB VRAM (or very close, depending on ctx/KV cache)
fetch \
  "$MODELS_DIR/jackrong/Qwopus3.5-9B-Coder-MTP-Q4_K_M.gguf" \
  "https://huggingface.co/Jackrong/Qwopus3.5-9B-Coder-MTP-GGUF/resolve/main/Qwopus3.5-9B-Coder-MTP-Q4_K_M.gguf?download=1"

fetch \
  "$MODELS_DIR/unsloth/Qwen3.5-9B-Q4_K_M.gguf" \
  "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=1"

# Category 2: best partially offloaded models (MoE / A3B / REAP priority)
fetch \
  "$MODELS_DIR/barozp/Qwen3.6-28B-REAP20-A3B-IQ3_XXS.gguf" \
  "https://huggingface.co/barozp/Qwen3.6-28B-REAP20-A3B-GGUF/resolve/main/Qwen3.6-28B-REAP20-A3B-IQ3_XXS.gguf?download=1"

fetch \
  "$MODELS_DIR/barozp/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf" \
  "https://huggingface.co/barozp/Qwen3.6-28B-REAP20-A3B-GGUF/resolve/main/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf?download=1"

fetch \
  "$MODELS_DIR/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf" \
  "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ2_M.gguf?download=1"

fetch \
  "$MODELS_DIR/unsloth/GLM-4.7-Flash-Q4_K_M.gguf" \
  "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf?download=1"

# Category 3: sensible baselines / comparison models for this RTX 3070 target
# Llama 3.1 8B Q4 is a good general baseline that can fit near/within 8GB depending on ctx/KV cache.
fetch \
  "$MODELS_DIR/bartowski/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
  "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=1"

# Phi 3.5 mini is a lightweight, fast fallback/baseline; it should fit comfortably in 8GB VRAM.
fetch \
  "$MODELS_DIR/bartowski/Phi-3.5-mini-instruct-Q4_K_M.gguf" \
  "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf?download=1"

# Gemma 4 E4B is small enough to try, but requires a recent llama.cpp with gemma4 architecture support.
fetch \
  "$MODELS_DIR/unsloth/gemma-4-E4B-it-Q4_K_M.gguf" \
  "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf?download=1"

run_downloads

echo "Done."

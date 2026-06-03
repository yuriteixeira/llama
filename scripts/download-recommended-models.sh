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

fetch \
  "$MODELS_DIR/unsloth/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
  "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf?download=true"

run_downloads 

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
MANIFEST="$ROOT_DIR/models.txt"

MAX_JOBS="${MAX_JOBS:-3}"
DOWNLOADS=()

# Parse models.txt — one URL per line, skip blanks and comments
while IFS= read -r url; do
  [[ "$url" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${url// /}" ]] && continue
  url="${url%%[[:space:]]}"  # trim trailing whitespace

  # Extract filename from URL
  filename="${url##*/}"
  filename="${filename%%\?*}"  # strip query string

  # Derive dest path: models/<org>/<filename>
  # e.g. https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/<file>
  #      → models/unsloth/<file>
  path_without_file="${url%%/resolve/*}"
  # path_without_file = https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
  # Take the last two segments as the repo dir
  repo_dir="${path_without_file##*/}"       # Qwen3.6-35B-A3B-GGUF
  org="${path_without_file%/*}"             # https://huggingface.co/unsloth
  org="${org##*/}"                          # unsloth

  dest="$MODELS_DIR/$org/$filename"
  DOWNLOADS+=("$dest"$'\t'"$url")
done < "$MANIFEST"

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

run_downloads

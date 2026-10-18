#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${PRESET:-$ROOT_DIR/model-presets.upstream.ini}"

MAX_JOBS="${MAX_JOBS:-3}"
DOWNLOADS=()
declare -A SEEN=()

if [[ ! -f "$PRESET" ]]; then
  echo "error: missing preset file: $PRESET" >&2
  exit 1
fi

if [[ ! "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: MAX_JOBS must be a positive integer" >&2
  exit 1
fi

repository_name() {
  local name="${1%.gguf}"
  local suffix prefix

  # Remove the quantization suffix to get the model name.
  for suffix in '-UD-*' '-IQ[0-9]*' '-Q[0-9]*' '-BF16*' '-F16*' '-F32*'; do
    prefix="${name%%$suffix}"
    if [[ "$prefix" != "$name" ]]; then
      printf '%s-GGUF\n' "$prefix"
      return
    fi
  done

  echo "error: cannot infer repository for $1" >&2
  return 1
}

remote_checksum() {
  local url="$1"
  local headers checksum

  echo "Checking server checksum: $url" >&2
  if ! headers="$(wget --spider --server-response "$url" 2>&1)"; then
    echo "error: cannot check server checksum: $url" >&2
    printf '%s\n' "$headers" >&2
    return 1
  fi
  # Hugging Face reports the SHA256 of large files in X-Linked-Etag.
  checksum="$(awk '
    tolower($1) == "x-linked-etag:" { linked = $2 }
    tolower($1) == "etag:" { etag = $2 }
    END {
      value = linked != "" ? linked : etag
      gsub(/["\r]/, "", value)
      print tolower(value)
    }
  ' <<< "$headers")"
  if [[ ! "$checksum" =~ ^[0-9a-f]{64}$ ]]; then
    echo "error: server provides no usable SHA256 checksum: $url" >&2
    return 1
  fi
  printf '%s\n' "$checksum"
}

file_signature() {
  # Include precise timestamps to detect changes even when the size stays the same.
  stat -L --printf='%d:%i:%s:%y:%z' -- "$1"
}

save_checksum_cache() {
  local dest="$1" signature="$2" checksum="$3"
  local cache="$dest.sha256-cache" temporary

  # A cache write failure must not prevent verification or downloads.
  if temporary="$(mktemp -- "$cache.XXXXXX")"; then
    if ! { printf '%s\n%s\n' "$signature" "$checksum" > "$temporary" &&
      mv -fT -- "$temporary" "$cache"; }; then
      rm -f -- "$temporary"
    fi
  fi
  return 0
}

matches_checksum() {
  local dest="$1" expected="$2"
  local signature actual cached_signature cached_checksum
  signature="$(file_signature "$dest")" || return
  if [[ -f "$dest.sha256-cache" && -r "$dest.sha256-cache" ]]; then
    {
      IFS= read -r cached_signature &&
        IFS= read -r cached_checksum
    } < "$dest.sha256-cache" || cached_checksum=""
    if [[ "$cached_signature" = "$signature" && "$cached_checksum" =~ ^[0-9a-f]{64}$ ]]; then
      echo "Using cached local SHA256 checksum: $dest" >&2
      [[ "$cached_checksum" = "$expected" ]]
      return
    fi
  fi

  echo "Checking local SHA256 checksum: $dest" >&2
  actual="$(sha256sum -- "$dest")" || return
  actual="${actual%% *}"
  if [[ "$(file_signature "$dest")" != "$signature" ]]; then
    echo "error: file changed during checksum calculation: $dest" >&2
    return 1
  fi
  save_checksum_cache "$dest" "$signature" "$actual"
  [[ "$actual" = "$expected" ]]
}

confirm_download() {
  local dest="$1"
  local answer

  echo "The local file differs from the server: $dest"
  echo "The server file may have changed, or the local file may be incomplete or damaged."
  printf 'Download it again? [y/N] ' >&2
  if [[ ! -r /dev/tty || ! -w /dev/tty ]] ||
    ! { read -r answer < /dev/tty; } 2>/dev/null; then
    echo "error: confirmation needs an interactive terminal; keeping $dest" >&2
    return 1
  fi
  [[ "$answer" = [yY] || "$answer" = [yY][eE][sS] ]]
}

# Read model paths and ignore comments and other preset settings.
while IFS= read -r dest; do
  [[ "$dest" = /* ]] || dest="$ROOT_DIR/$dest"
  [[ -n "${SEEN[$dest]:-}" ]] && continue
  SEEN["$dest"]=1

  filename="${dest##*/}"
  parent="${dest%/*}"
  org="${parent##*/}"
  repo_dir="$(repository_name "$filename")"
  url="https://huggingface.co/$org/$repo_dir/resolve/main/$filename"
  checksum="$(remote_checksum "$url")"
  if [[ -f "$dest" ]]; then
    if matches_checksum "$dest" "$checksum"; then
      echo "Verified: $dest"
      continue
    fi
    if ! confirm_download "$dest"; then
      echo "Keeping existing file: $dest"
      continue
    fi
  fi
  DOWNLOADS+=("$dest"$'\t'"$url"$'\t'"$checksum")
done < <(awk '
  /^[[:space:]]*model[[:space:]]*=/ {
    sub(/^[^=]*=[[:space:]]*/, "")
    sub(/[[:space:]]+[;#].*$/, "")
    sub(/[[:space:]]*$/, "")
    if (length) print
  }
' "$PRESET")

download_one() {
  local spec="$1"
  local dest="${spec%%$'\t'*}"
  local remainder="${spec#*$'\t'}"
  local url="${remainder%%$'\t'*}"
  local checksum="${remainder#*$'\t'}"
  # Keep partial downloads separate for each server version.
  local partial="$dest.$checksum.part"

  mkdir -p "$(dirname "$dest")" || return
  echo "Downloading: $dest"
  wget --continue --show-progress --progress=bar:force:noscroll --output-document="$partial" "$url" || return
  if ! matches_checksum "$partial" "$checksum"; then
    echo "error: checksum mismatch; keeping partial file: $partial" >&2
    return 1
  fi
  mv -- "$partial" "$dest" || return
  rm -f -- "$partial.sha256-cache"
  local signature
  signature="$(file_signature "$dest")" || return
  save_checksum_cache "$dest" "$signature" "$checksum"
  echo "Download complete and checksum verified: $dest"
}

run_downloads() {
  echo "Queued ${#DOWNLOADS[@]} model downloads (MAX_JOBS=$MAX_JOBS)."
  [[ ${#DOWNLOADS[@]} -gt 0 ]] || return 0
  printf '%s\0' "${DOWNLOADS[@]}" \
    | xargs -0 -n 1 -P "$MAX_JOBS" bash -c 'download_one "$1"' _
}

export -f download_one matches_checksum file_signature save_checksum_cache

run_downloads

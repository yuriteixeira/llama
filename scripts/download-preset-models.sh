#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
PRESETS=(
  "$ROOT_DIR/model-presets.buun.ini"
  "$ROOT_DIR/model-presets.upstream.ini"
  "$ROOT_DIR/model-presets.turboquant.ini"
)

usage() {
  cat <<'EOF'
Usage: scripts/update-presets.sh [--preset <path>]...

Adds local .gguf models missing from preset files.
By default, updates all split preset files:
  - ./model-presets.buun.ini
  - ./model-presets.upstream.ini
  - ./model-presets.turboquant.ini
EOF
}

resolve_preset() {
  local preset="$1"
  if [[ "$preset" = /* ]]; then
    printf '%s\n' "$preset"
  else
    printf '%s\n' "$ROOT_DIR/$preset"
  fi
}

if [[ $# -gt 0 ]]; then
  PRESETS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --preset)
        PRESETS+=("$(resolve_preset "${2:?missing value for --preset}")")
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "error: unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
fi

if [[ ${#PRESETS[@]} -eq 0 ]]; then
  echo "error: no preset files selected" >&2
  exit 1
fi

for presets in "${PRESETS[@]}"; do
  if [[ ! -f "$presets" ]]; then
    echo "error: missing preset file: $presets" >&2
    exit 1
  fi

done

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "error: missing models directory: $MODELS_DIR" >&2
  exit 1
fi

update_preset() {
  local presets="$1"

  # Collect all .gguf filenames currently declared in preset sections (not commented out)
  # Extracts the basename of the model path from active [N-name] sections
  declare -A declared=()
  while IFS= read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^[[:space:]]*[\;#] ]] && continue
    [[ -z "${line// /}" ]] && continue

    # Match: model = /path/to/file.gguf
    if [[ "$line" =~ ^model[[:space:]]*=[[:space:]]*(.+) ]]; then
      filepath="${BASH_REMATCH[1]}"
      filename="${filepath##*/}"
      declared["$filename"]=1
    fi
  done < "$presets"

  # Find all local .gguf files not yet declared
  new_models=()
  while IFS= read -r gguf; do
    filename="${gguf##*/}"
    if [[ -z "${declared[$filename]+_}" ]]; then
      new_models+=("$gguf")
    fi
  done < <(find "$MODELS_DIR" -name '*.gguf' -type f 2>/dev/null | sort)

  if [[ ${#new_models[@]} -eq 0 ]]; then
    echo "No new models found for $presets. All local .gguf files are already in presets."
    return
  fi

  # Determine next preset number from existing sections
  next_num=1
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[([0-9]+)- ]]; then
      num="${BASH_REMATCH[1]}"
      # Remove leading zeros for arithmetic
      num=$((10#$num))
      if (( num >= next_num )); then
        next_num=$((num + 1))
      fi
    fi
  done < "$presets"

  first_added=$next_num

  # Append new presets
  {
    echo ""
    for gguf in "${new_models[@]}"; do
      filename="${gguf##*/}"
      name_slug="${filename%.gguf}"
      # Strip common prefixes/suffixes for a cleaner name
      name_slug="${name_slug%%-[A-Z]*}"  # remove -UD, -Q4_K_M, etc.
      name_slug="${name_slug,,}"          # lowercase
      name_slug="${name_slug// /-}"       # spaces to dashes
      name_slug="${name_slug//_/}"        # remove underscores

      echo "[$next_num-$name_slug]"
      echo "model = $gguf"
      echo ""
      (( next_num++ ))
    done
  } >> "$presets"

  echo "Added ${#new_models[@]} new preset(s) starting at [$first_added-...]. Updated $presets"
}

for presets in "${PRESETS[@]}"; do
  update_preset "$presets"
done

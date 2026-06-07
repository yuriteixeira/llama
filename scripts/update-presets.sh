#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESETS="$ROOT_DIR/model-presets.ini"
MODELS_DIR="$ROOT_DIR/models"

# Collect all .gguf filenames currently declared in preset sections (not commented out)
# Extracts the basename of the model path from active [N-name] sections
declare -A declared
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
done < "$PRESETS"

# Find all local .gguf files not yet declared
new_models=()
while IFS= read -r gguf; do
  filename="${gguf##*/}"
  if [[ -z "${declared[$filename]+_}" ]]; then
    new_models+=("$gguf")
  fi
done < <(find "$MODELS_DIR" -name '*.gguf' -type f 2>/dev/null | sort)

if [[ ${#new_models[@]} -eq 0 ]]; then
  echo "No new models found. All local .gguf files are already in presets."
  exit 0
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
done < "$PRESETS"

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
} >> "$PRESETS"

echo "Added ${#new_models[@]} new preset(s) starting at [$((next_num - ${#new_models[@]}))-...]. Updated $PRESETS"

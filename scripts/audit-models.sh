#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
MANIFEST="$ROOT_DIR/models.txt"

usage() {
  cat <<'EOF'
Usage: scripts/audit-models.sh [--preset <path>]...

Audits the model inventory across:
  - ./models directory (.gguf files, recursively)
  - ./models.txt download manifest
  - preset model entries from each selected preset file

By default, audits all existing ./model-presets.*.ini files.
Missing preset files are skipped gracefully.
EOF
}

declare -a preset_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      preset_args+=("${2:?missing value for --preset}")
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

resolve_preset() {
  local preset="$1"
  if [[ "$preset" = /* ]]; then
    printf '%s\n' "$preset"
  else
    printf '%s\n' "$ROOT_DIR/$preset"
  fi
}

declare -a preset_files=()
if [[ ${#preset_args[@]} -gt 0 ]]; then
  for preset in "${preset_args[@]}"; do
    path="$(resolve_preset "$preset")"
    [[ -f "$path" ]] && preset_files+=("$path")
  done
else
  shopt -s nullglob
  for path in "$ROOT_DIR"/model-presets.*.ini; do
    [[ -f "$path" ]] && preset_files+=("$path")
  done
  shopt -u nullglob
fi

declare -A in_models_dir=()
declare -A in_manifest=()
declare -A in_preset=()
declare -A all_models=()

add_model() {
  local name="$1"
  [[ -z "$name" ]] && return
  all_models["$name"]=1
}

# Local inventory: recursively list downloaded GGUF files under ./models.
if [[ -d "$MODELS_DIR" ]]; then
  while IFS= read -r model_path; do
    name="${model_path##*/}"
    in_models_dir["$name"]=1
    add_model "$name"
  done < <(find "$MODELS_DIR" -type f -name '*.gguf' 2>/dev/null | sort)
fi

# Manifest inventory: extract destination filenames from one URL per line.
if [[ -f "$MANIFEST" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*[#\;] ]] && continue
    [[ -z "${line//[[:space:]]/}" ]] && continue

    url="${line%%[[:space:]]*}"
    name="${url##*/}"
    name="${name%%\?*}"

    [[ "$name" == *.gguf ]] || continue
    in_manifest["$name"]=1
    add_model "$name"
  done < "$MANIFEST"
fi

# Preset inventory: extract basenames from active model = ... lines per preset file.
for i in "${!preset_files[@]}"; do
  preset="${preset_files[$i]}"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*[#\;] ]] && continue
    [[ -z "${line//[[:space:]]/}" ]] && continue

    if [[ "$line" =~ ^[[:space:]]*model[[:space:]]*=[[:space:]]*(.+)$ ]]; then
      model_path="${BASH_REMATCH[1]}"
      model_path="${model_path%%[[:space:]]\;*}"
      model_path="${model_path%%[[:space:]]#*}"
      name="${model_path##*/}"

      [[ "$name" == *.gguf ]] || continue
      in_preset["$i:$name"]=1
      add_model "$name"
    fi
  done < "$preset"
done

yes_no() {
  if [[ "$1" == true ]]; then printf 'yes'; else printf 'no'; fi
}

preset_label() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR"/* ]]; then
    printf '%s' "${path#$ROOT_DIR/}"
  else
    printf '%s' "$path"
  fi
}

declare -a headers widths
headers=("name" "in ./models" "in ./models.txt")
widths=(58 11 15)
for preset in "${preset_files[@]}"; do
  label="$(preset_label "$preset")"
  headers+=("$label")
  widths+=(${#label})
done

for i in "${!widths[@]}"; do
  (( widths[$i] < ${#headers[$i]} )) && widths[$i]=${#headers[$i]}
done

repeat_char() {
  local char="$1" count="$2" output=''
  printf -v output '%*s' "$count" ''
  printf '%s' "${output// /$char}"
}

print_border() {
  local left="$1" junction="$2" right="$3"
  printf '%s' "$left"
  for i in "${!widths[@]}"; do
    printf '%s' "$(repeat_char '─' "$((widths[$i] + 2))")"
    if (( i == ${#widths[@]} - 1 )); then printf '%s\n' "$right"; else printf '%s' "$junction"; fi
  done
}

print_row() {
  local -a values=("$@")
  printf '│'
  for i in "${!widths[@]}"; do
    printf ' %-*s │' "${widths[$i]}" "${values[$i]:-}"
  done
  printf '\n'
}

print_border '┌' '┬' '┐'
print_row "${headers[@]}"
print_border '├' '┼' '┤'

if [[ ${#all_models[@]} -eq 0 ]]; then
  print_row 'No models found.' '' ''
  print_border '└' '┴' '┘'
  exit 0
fi

while IFS= read -r name; do
  values=("$name")
  if [[ -n "${in_models_dir[$name]+_}" ]]; then values+=(yes); else values+=(no); fi
  if [[ -n "${in_manifest[$name]+_}" ]]; then values+=(yes); else values+=(no); fi

  for i in "${!preset_files[@]}"; do
    if [[ -n "${in_preset[$i:$name]+_}" ]]; then values+=(yes); else values+=(no); fi
  done

  print_row "${values[@]}"
done < <(printf '%s\n' "${!all_models[@]}" | sort)

print_border '└' '┴' '┘'

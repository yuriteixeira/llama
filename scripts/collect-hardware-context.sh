#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

warn() {
  echo "warning: $*; continuing without these hardware specs" >&2
}

print_specs() {
  local title="$1" output
  shift
  printf '\n%s:\n' "$title"
  if command -v "$1" >/dev/null 2>&1 && output="$("$@" 2>/dev/null)" && [[ -n "$output" ]]; then
    printf '%s\n' "$output"
  else
    warn "$1 is missing or failed"
    printf 'Unavailable.\n'
  fi
}

printf 'Detected Linux hardware (context, not evidence of source changes):\n'
printf '\nCPU:\n'
if command -v lscpu >/dev/null 2>&1 && cpu="$(lscpu 2>/dev/null)" && [[ -n "$cpu" ]]; then
  while IFS= read -r line; do
    case "$line" in
      Architecture:*|'CPU(s):'*|'Model name:'*|'Socket(s):'*|'Core(s) per socket:'*|'Thread(s) per core:'*|Flags:*)
        printf '%s\n' "$line"
        ;;
    esac
  done <<< "$cpu"
else
  warn 'lscpu is missing or failed'
  printf 'Unavailable.\n'
fi

print_specs 'System memory and swap in MiB (current snapshot)' free --mebi

printf '\nGPU specs (index, name, total memory, compute capability, driver):\n'
if command -v nvidia-smi >/dev/null 2>&1 && \
  gpu="$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader 2>/dev/null)" && \
  [[ -n "$gpu" ]]; then
  printf '%s\n' "$gpu"
elif command -v nvidia-smi >/dev/null 2>&1 && \
  gpu="$(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null)" && \
  [[ -n "$gpu" ]]; then
  warn 'NVIDIA compute capability query failed'
  printf 'Compute capability unavailable. Columns: index, name, total memory, driver.\n%s\n' "$gpu"
else
  warn 'nvidia-smi is missing or failed'
  printf 'NVIDIA specs unavailable. Trying PCI display devices; memory and compute capability are unknown.\n'
  if command -v lspci >/dev/null 2>&1 && pci="$(lspci 2>/dev/null)"; then
    display=''
    while IFS= read -r line; do
      case "$line" in
        *'VGA compatible controller'*|*'3D controller'*|*'Display controller'*)
          display+="$line"$'\n'
          ;;
      esac
    done <<< "$pci"
    printf '%s\n' "${display:-No display devices reported.}"
  else
    warn 'lspci is missing or failed'
    printf 'PCI display device specs unavailable.\n'
  fi
fi

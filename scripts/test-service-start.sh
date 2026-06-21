#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/test-service-start.sh [--instance <name>] [--cycles <n>] [--timeout <seconds>]

Cold-start test for llama-server@<instance>.service.
For each cycle it stops llama-server and nvidia-persistenced, starts only the llama
server service, verifies nvidia-persistenced became active, waits for the llama
server to become active, then waits for HTTP readiness.
USAGE
}

read_env_value() {
  local key=$1
  local file=$2
  [[ -r "$file" ]] || return 1
  awk -F= -v key="$key" '$1 == key { print substr($0, length(key) + 2); exit }' "$file"
}

wait_for_active() {
  local service=$1
  local timeout=$2
  local start
  start=$(date +%s)

  while true; do
    if systemctl is-active --quiet "$service"; then
      return 0
    fi

    if [[ $(($(date +%s) - start)) -ge $timeout ]]; then
      echo "error: timed out waiting for $service to become active" >&2
      systemctl status "$service" --no-pager -l || true
      return 1
    fi

    sleep 2
  done
}

wait_for_http() {
  local url=$1
  local timeout=$2
  local start
  start=$(date +%s)

  while true; do
    if curl -4 -fsS -I --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi

    if [[ $(($(date +%s) - start)) -ge $timeout ]]; then
      echo "error: timed out waiting for HTTP readiness at $url" >&2
      return 1
    fi

    sleep 2
  done
}

INSTANCE="${LLAMA_SERVICE_INSTANCE:-${USER}}"
CYCLES=3
TIMEOUT=1800

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance)
      INSTANCE=${2:?missing value for --instance}
      shift 2
      ;;
    --cycles)
      CYCLES=${2:?missing value for --cycles}
      shift 2
      ;;
    --timeout)
      TIMEOUT=${2:?missing value for --timeout}
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

SERVICE_NAME="llama-server@${INSTANCE}.service"
ENV_FILE="/etc/llama-server/${INSTANCE}.env"
LLAMA_PORT=$(read_env_value LLAMA_PORT "$ENV_FILE" || true)
LLAMA_PORT=${LLAMA_PORT:-8080}
READINESS_URL="http://127.0.0.1:${LLAMA_PORT}/"

for cycle in $(seq 1 "$CYCLES"); do
  echo "== Cycle $cycle/$CYCLES =="
  echo "Stopping $SERVICE_NAME and nvidia-persistenced.service"
  sudo systemctl stop "$SERVICE_NAME" || true
  sudo systemctl stop nvidia-persistenced.service || true

  echo "Starting only $SERVICE_NAME"
  sudo systemctl start "$SERVICE_NAME"

  echo "Waiting for nvidia-persistenced.service"
  wait_for_active nvidia-persistenced.service 60

  echo "Waiting for $SERVICE_NAME"
  wait_for_active "$SERVICE_NAME" "$TIMEOUT"

  echo "Waiting for HTTP readiness at $READINESS_URL"
  wait_for_http "$READINESS_URL" "$TIMEOUT"

  echo "Cycle $cycle succeeded"
  echo
done

echo "All $CYCLES cold-start cycles succeeded."

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/install-system-service.sh [options]

Install the llama-server systemd template and create the instance environment file.

Options:
  --user <name>         Linux user/group to run the service as (default: $SUDO_USER or $USER)
  --instance <name>     Systemd template instance name (default: same as --user)
  --host <host>         Bind host (default: 0.0.0.0)
  --port <port>         Bind port (default: 8080)
  --device <device>     CUDA readiness probe device selector (default: CUDA0)
  --models-preset <path> Model presets path passed to llama-server (default: ./model-presets.buun.ini)
  --models-max <n>      Maximum number of loaded models (default: 1)
  --parallel <n>        Parallel requests/sequences (default: 1)
  --workdir <path>      Repository/service working directory (default: repo root)
  --bin <path>          llama-server binary path (default: <workdir>/versions/buun/src/build/bin/llama-server)
  --no-start            Install files and reload systemd, but do not enable/start services
  -h, --help            Show this help
USAGE
}

repo_dir() {
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

write_env_file() {
  local target=$1
  local tmp_file
  tmp_file=$(mktemp)
  trap 'rm -f "$tmp_file"' RETURN

  cat >"$tmp_file" <<EOF
LLAMA_WORKDIR=${LLAMA_WORKDIR}
LLAMA_BIN=${LLAMA_BIN}
LLAMA_HOST=${LLAMA_HOST}
LLAMA_PORT=${LLAMA_PORT}
LLAMA_MODELS_PRESET=${LLAMA_MODELS_PRESET}
LLAMA_MODELS_MAX=${LLAMA_MODELS_MAX}
LLAMA_PARALLEL=${LLAMA_PARALLEL}
LLAMA_DEVICE=${LLAMA_DEVICE}
LLAMA_CUDA_PROBE_ATTEMPTS=120
LLAMA_NVIDIA_SMI_TIMEOUT=5
LLAMA_LIST_DEVICES_TIMEOUT=10
LLAMA_CUDA_PROBE_SLEEP=2
EOF

  sudo install -d -m 0755 /etc/llama-server
  sudo install -m 0644 "$tmp_file" "$target"
}

REPO_DIR=$(repo_dir)
SERVICE_USER=${SUDO_USER:-${USER}}
INSTANCE=""
LLAMA_WORKDIR="$REPO_DIR"
LLAMA_BIN=""
LLAMA_HOST="0.0.0.0"
LLAMA_PORT="8080"
LLAMA_DEVICE="CUDA0"
LLAMA_MODELS_PRESET="./model-presets.buun.ini"
LLAMA_MODELS_MAX="1"
LLAMA_PARALLEL="1"
START_SERVICES=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      SERVICE_USER=${2:?missing value for --user}
      shift 2
      ;;
    --instance)
      INSTANCE=${2:?missing value for --instance}
      shift 2
      ;;
    --host)
      LLAMA_HOST=${2:?missing value for --host}
      shift 2
      ;;
    --port)
      LLAMA_PORT=${2:?missing value for --port}
      shift 2
      ;;
    --device)
      LLAMA_DEVICE=${2:?missing value for --device}
      shift 2
      ;;
    --models-preset)
      LLAMA_MODELS_PRESET=${2:?missing value for --models-preset}
      shift 2
      ;;
    --models-max)
      LLAMA_MODELS_MAX=${2:?missing value for --models-max}
      shift 2
      ;;
    --parallel)
      LLAMA_PARALLEL=${2:?missing value for --parallel}
      shift 2
      ;;
    --workdir)
      LLAMA_WORKDIR=${2:?missing value for --workdir}
      shift 2
      ;;
    --bin)
      LLAMA_BIN=${2:?missing value for --bin}
      shift 2
      ;;
    --no-start)
      START_SERVICES=0
      shift
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

INSTANCE=${INSTANCE:-$SERVICE_USER}
LLAMA_BIN=${LLAMA_BIN:-$LLAMA_WORKDIR/versions/buun/src/build/bin/llama-server}
SERVICE_NAME="llama-server@${INSTANCE}.service"
ENV_FILE="/etc/llama-server/${INSTANCE}.env"
UNIT_SOURCE="$REPO_DIR/llama-server@.service"

require_command sudo

if ! getent passwd "$SERVICE_USER" >/dev/null; then
  echo "error: user does not exist: $SERVICE_USER" >&2
  exit 1
fi

if ! getent group "$SERVICE_USER" >/dev/null; then
  echo "error: group does not exist: $SERVICE_USER" >&2
  echo "       the template uses Group=%i, so --instance must match an existing group" >&2
  exit 1
fi

if [[ "$INSTANCE" != "$SERVICE_USER" ]]; then
  echo "error: instance must match --user for the template User=%i/Group=%i design" >&2
  echo "       got instance=$INSTANCE user=$SERVICE_USER" >&2
  exit 1
fi

if [[ ! -f "$UNIT_SOURCE" ]]; then
  echo "error: missing systemd template: $UNIT_SOURCE" >&2
  exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "error: llama-server binary is not executable: $LLAMA_BIN" >&2
  exit 1
fi

if [[ "$LLAMA_MODELS_PRESET" = /* ]]; then
  MODELS_PRESET_PATH="$LLAMA_MODELS_PRESET"
else
  MODELS_PRESET_PATH="$LLAMA_WORKDIR/$LLAMA_MODELS_PRESET"
fi

if [[ ! -f "$MODELS_PRESET_PATH" ]]; then
  echo "error: missing model presets file: $MODELS_PRESET_PATH" >&2
  exit 1
fi

write_env_file "$ENV_FILE"
sudo cp "$UNIT_SOURCE" /etc/systemd/system/llama-server@.service
sudo systemctl daemon-reload

if [[ "$START_SERVICES" -eq 1 ]]; then
  sudo systemctl enable --now nvidia-persistenced.service
  sudo systemctl enable --now "$SERVICE_NAME"
fi

cat <<EOF
Installed $SERVICE_NAME

Environment file: $ENV_FILE
Template unit:     /etc/systemd/system/llama-server@.service

Validation commands:
  systemctl status nvidia-persistenced "$SERVICE_NAME" --no-pager -l
  journalctl -u "$SERVICE_NAME" -f
  curl -4 -I "http://127.0.0.1:${LLAMA_PORT}/"
EOF

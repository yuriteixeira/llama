#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./manage-server.sh [--instance <name>] <action> [installer options]

Actions:
  install      Install/update the systemd template service and env file
  download-models  Download and verify models from the selected preset
  enable       Enable the service at boot
  start        Start the service
  stop         Stop the service
  restart      Restart the service
  status       Show service status
  logs         Follow service logs
  snapshot     Print useful diagnostics for nvidia-persistenced and the service
  test-start   Run the cold-start validation script

Instance defaults to $LLAMA_SERVICE_INSTANCE or $USER.
Installer options after "install" are forwarded to scripts/install-system-service.sh.
The download-models action uses PRESET and MAX_JOBS from the environment.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE="${LLAMA_SERVICE_INSTANCE:-${USER}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance)
      INSTANCE=${2:?missing value for --instance}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

ACTION=${1:-}
if [[ -z "$ACTION" ]]; then
  usage >&2
  exit 1
fi
shift || true

SERVICE_NAME="llama-server@${INSTANCE}.service"

case "$ACTION" in
  install)
    exec "$SCRIPT_DIR/scripts/install-system-service.sh" --instance "$INSTANCE" --user "$INSTANCE" "$@"
    ;;
  download-models)
    exec "$SCRIPT_DIR/scripts/download-preset-models.sh" "$@"
    ;;
  enable)
    sudo systemctl enable "$SERVICE_NAME"
    ;;
  start)
    sudo systemctl start "$SERVICE_NAME"
    ;;
  stop)
    sudo systemctl stop "$SERVICE_NAME"
    ;;
  restart)
    sudo systemctl restart "$SERVICE_NAME"
    ;;
  status)
    systemctl status nvidia-persistenced.service "$SERVICE_NAME" --no-pager -l
    ;;
  logs)
    journalctl -u "$SERVICE_NAME" -f
    ;;
  snapshot)
    echo "# systemd status"
    systemctl status nvidia-persistenced.service "$SERVICE_NAME" --no-pager -l || true
    echo
    echo "# recent service logs"
    journalctl -u "$SERVICE_NAME" -n 120 --no-pager || true
    echo
    echo "# environment file"
    if [[ -r "/etc/llama-server/${INSTANCE}.env" ]]; then
      sed 's/^/  /' "/etc/llama-server/${INSTANCE}.env"
    else
      echo "  /etc/llama-server/${INSTANCE}.env is not readable"
    fi
    ;;
  test-start)
    exec "$SCRIPT_DIR/scripts/test-service-start.sh" --instance "$INSTANCE" "$@"
    ;;
  *)
    echo "error: unknown action: $ACTION" >&2
    usage >&2
    exit 1
    ;;
esac

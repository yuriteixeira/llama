#!/usr/bin/env bash
# manage-server.sh — Manage llama-server systemd user services
# Usage: ./manage-server.sh {vanilla|buun} {start|stop|restart|status|enable|disable|logs|install|snapshot}

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
USAGE="Usage: $0 {vanilla|buun} {start|stop|restart|status|enable|disable|logs|install|snapshot}"
MODE="${1:-}"
ACTION="${2:-status}"

if [ -z "$MODE" ]; then
  echo "$USAGE"
  exit 1
fi

case "$MODE" in
  vanilla) SERVICE_NAME="llama-server.service" ;;
  buun)    SERVICE_NAME="llama-server-buun.service" ;;
  *)       echo "Unknown mode: $MODE (use vanilla or buun)"; exit 1 ;;
esac

SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME"
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"
INSTALLED_SERVICE_FILE="$USER_SYSTEMD_DIR/$SERVICE_NAME"

install_unit() {
  mkdir -p "$USER_SYSTEMD_DIR"
  ln -sfn "$SERVICE_FILE" "$INSTALLED_SERVICE_FILE"
  systemctl --user daemon-reload
  echo "Installed $SERVICE_NAME as symlink: $INSTALLED_SERVICE_FILE -> $SERVICE_FILE"
}

case "$ACTION" in
  install)
    install_unit
    ;;
  logs)
    journalctl --user -u "$SERVICE_NAME" -f -n 100
    ;;
  snapshot)
    echo "== systemd status =="
    systemctl --user --no-pager --full status "$SERVICE_NAME" || true
    echo
    echo "== recent logs =="
    journalctl --user -u "$SERVICE_NAME" --no-pager -n 80 || true
    echo
    echo "== nvidia-smi =="
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv || true
    echo
    echo "== llama-server metrics (:8080/metrics) =="
    curl -fsS --max-time 3 http://127.0.0.1:8080/metrics || true
    ;;
  enable)
    install_unit
    systemctl --user enable "$SERVICE_NAME"
    echo "Enabled $SERVICE_NAME. It will start when your user systemd session starts."
    echo "For startup before login after reboot, run: sudo loginctl enable-linger $USER"
    ;;
  disable)
    systemctl --user disable "$SERVICE_NAME"
    echo "Disabled $SERVICE_NAME."
    ;;
  start|stop|restart|status)
    if [ ! -f "$INSTALLED_SERVICE_FILE" ]; then
      install_unit
    fi
    systemctl --user "$ACTION" "$SERVICE_NAME"
    ;;
  *)
    echo "Unknown action: $ACTION"
    echo "Usage: $0 {vanilla|buun} {start|stop|restart|status|enable|disable|logs|install|snapshot}"
    exit 1
    ;;
esac

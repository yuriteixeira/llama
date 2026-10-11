#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "error: $*" >&2
  exit 1
}

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/version-source.sh" "$@"
[[ -e "$SRC_DIR/.git" ]] || fail "no source checkout: $SRC_DIR; clone the source before requesting a changelog"

PREVIOUS_COMMIT="$(git -C "$SRC_DIR" rev-parse HEAD)"
git -C "$SRC_DIR" fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
NEW_COMMIT="$(git -C "$SRC_DIR" rev-parse "refs/remotes/origin/$BRANCH")"
if [[ "$PREVIOUS_COMMIT" = "$NEW_COMMIT" ]]; then
  echo "warning: $VERSION (branch: $BRANCH) is already at the latest remote commit $NEW_COMMIT; no report was generated" >&2
  exit 0
fi
command -v pi >/dev/null 2>&1 || fail 'pi is required for changelog'
PRESET_FILE="${PRESET:-$ROOT_DIR/model-presets.$VERSION.ini}"
if [[ -z "${PRESET:-}" && ! -f "$PRESET_FILE" ]]; then
  PRESET_FILE="$ROOT_DIR/archived/model-presets.$VERSION.ini"
fi
if [[ ! -r "$PRESET_FILE" ]]; then
  [[ -z "${PRESET:-}" ]] || fail "cannot read preset file: $PRESET_FILE"
  echo "warning: no preset file for $VERSION; preset relevance cannot be assessed" >&2
  PRESET_FILE=''
fi
REPORT_DATE="$(date +%F)"
SUMMARY_FILE="/tmp/llama-$VERSION-summary-$REPORT_DATE.md"
bash "$ROOT_DIR/scripts/summarize-source-update.sh" \
  "$SRC_DIR" "$PREVIOUS_COMMIT" "$NEW_COMMIT" "$SUMMARY_FILE" "$VERSION" "$BRANCH" "$REPORT_DATE" "$PRESET_FILE"

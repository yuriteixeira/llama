#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "error: $*" >&2
  exit 1
}

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/version-source.sh" "$@"
BUILD_SCRIPT="$ROOT_DIR/versions/$VERSION/build-$VERSION-llama-cpp.sh"
[[ -f "$BUILD_SCRIPT" ]] || fail "missing build script: $BUILD_SCRIPT"
if [[ ! -e "$SRC_DIR" ]]; then
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$SRC_DIR"
else
  [[ -e "$SRC_DIR/.git" ]] || fail "source directory is not a Git checkout: $SRC_DIR"
  [[ -z "$(git -C "$SRC_DIR" status --porcelain)" ]] ||
    fail 'source has local changes; commit or stash them before updating'
  git -C "$SRC_DIR" fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
  if git -C "$SRC_DIR" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git -C "$SRC_DIR" checkout "$BRANCH"
  else
    git -C "$SRC_DIR" checkout --no-track -b "$BRANCH" "refs/remotes/origin/$BRANCH"
  fi
  git -C "$SRC_DIR" merge --ff-only "refs/remotes/origin/$BRANCH"
  [[ "$(git -C "$SRC_DIR" rev-parse HEAD)" = "$(git -C "$SRC_DIR" rev-parse "refs/remotes/origin/$BRANCH")" ]] ||
    fail 'local branch has commits not present on the remote; resolve them before building'
fi

echo "Building $VERSION from $BRANCH at $(git -C "$SRC_DIR" rev-parse --short HEAD)"
exec bash "$BUILD_SCRIPT"

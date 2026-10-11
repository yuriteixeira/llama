#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "error: $*" >&2
  exit 1
}

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-upstream}"
[[ $# -le 2 ]] || fail 'usage: ./manage-server.sh update-build [version] [branch]'

case "$VERSION" in
  upstream)
    REPO_URL='https://github.com/ggml-org/llama.cpp.git'
    DEFAULT_BRANCH=master
    ;;
  buun)
    REPO_URL='https://github.com/spiritbuun/buun-llama-cpp.git'
    DEFAULT_BRANCH=master
    ;;
  turboquant)
    REPO_URL='https://github.com/TheTom/llama-cpp-turboquant.git'
    DEFAULT_BRANCH=feature/turboquant-kv-cache
    ;;
  *) fail "unknown version: $VERSION (use upstream, buun, or turboquant)" ;;
esac

BRANCH="${2:-$DEFAULT_BRANCH}"
git check-ref-format "refs/heads/$BRANCH" >/dev/null || fail "invalid branch: $BRANCH"
[[ "$BRANCH" != -* ]] || fail "invalid branch: $BRANCH"
SRC_DIR="$ROOT_DIR/versions/$VERSION/src"
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

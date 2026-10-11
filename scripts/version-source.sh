#!/usr/bin/env bash
# Shared source settings. The caller supplies fail() and ROOT_DIR.
VERSION="${1:-upstream}"
[[ $# -le 2 ]] || fail 'expected optional version and branch arguments'
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

#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_URL="https://github.com/ggml-org/llama.cpp.git"
readonly SRC_DIR="$ROOT_DIR/src"
readonly BUILD_DIR="$SRC_DIR/build"
readonly JOBS="$(nproc)"

# Current GPU: RTX 5060 Ti (compute capability 12.0).
# Future/rollback GPU: RTX 3070 (compute capability 8.6).
readonly CUDA_ARCHITECTURES="86;120"

export PATH="/opt/cuda/bin:$PATH"

if [[ -d "$SRC_DIR/.git" ]]; then
  echo "using existing upstream llama.cpp checkout: $(git -C "$SRC_DIR" describe --tags --always --dirty 2>/dev/null || git -C "$SRC_DIR" rev-parse --short HEAD)"
else
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone "$REPO_URL" "$SRC_DIR"
fi

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=ON

cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo
echo "upstream llama.cpp source: $SRC_DIR"
echo "upstream llama.cpp build:  $BUILD_DIR"
echo "CUDA arch:                 $CUDA_ARCHITECTURES"
echo "server binary:             $BUILD_DIR/bin/llama-server"
echo "cli binary:                $BUILD_DIR/bin/llama-cli"

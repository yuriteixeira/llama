#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_URL="https://github.com/TheTom/llama-cpp-turboquant.git"
readonly SRC_DIR="$ROOT_DIR/src"
readonly BUILD_DIR="$SRC_DIR/build"
readonly JOBS="$(nproc)"

# Current GPU: RTX 5060 Ti (compute capability 12.0).
# Future/rollback GPU: RTX 3070 (compute capability 8.6).
readonly CUDA_ARCHITECTURES="86;120"
readonly DEFAULT_BRANCH="feature/turboquant-kv-cache"

export PATH="/opt/cuda/bin:$PATH"

if git -C "$SRC_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  # Do not force checkout/pull here.
  # This lets the build use whatever llama-cpp-turboquant version is currently
  # checked out: a branch, tag, pinned commit, or local working tree. That avoids
  # accidentally replacing a deliberately selected version. This also supports
  # submodule checkouts, where .git is a file instead of a directory.
  echo "using existing llama-cpp-turboquant checkout: $(git -C "$SRC_DIR" describe --tags --always --dirty 2>/dev/null || git -C "$SRC_DIR" rev-parse --short HEAD)"
else
  # Clone only when the source tree does not exist yet. The active TurboQuant+
  # branch is not the repository's historical master branch.
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --branch "$DEFAULT_BRANCH" "$REPO_URL" "$SRC_DIR"
fi

# CMake option notes:
# - GGML_CUDA=ON enables NVIDIA CUDA acceleration.
# - GGML_NATIVE=ON lets ggml add -march=native for the CPU backend, which is
#   best for a local-only optimized build but not portable to older CPUs.
# - GGML_CUDA_FA=ON enables CUDA FlashAttention kernels used by TurboQuant KV.
# - GGML_CUDA_FA_ALL_QUANTS=ON compiles FlashAttention support for all KV cache
#   quantization combinations. This takes longer to compile, but keeps the build
#   flexible for q8_0/turbo* and f16/turbo* preset experiments.
# - GGML_CUDA_GRAPHS=ON keeps parity with the upstream build script.
# - CMAKE_CUDA_ARCHITECTURES targets both the current RTX 5060 Ti and RTX 3070
#   rollback/future secondary GPU instead of relying on CMake's native GPU detection.
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
echo "llama-cpp-turboquant source: $SRC_DIR"
echo "llama-cpp-turboquant build:  $BUILD_DIR"
echo "CUDA arch:                  $CUDA_ARCHITECTURES"
echo "server binary:              $BUILD_DIR/bin/llama-server"
echo "cli binary:                 $BUILD_DIR/bin/llama-cli"

#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_URL="https://github.com/spiritbuun/buun-llama-cpp.git"
readonly SRC_DIR="$ROOT_DIR/src"
readonly BUILD_DIR="$SRC_DIR/build"
readonly JOBS="$(nproc)"

# Current GPU: RTX 5060 Ti (compute capability 12.0).
# Rollback/future secondary GPU: RTX 3070 (compute capability 8.6).
# GGML_NATIVE=ON below handles CPU-side machine-local optimization; this keeps
# only the GPU-specific architecture setting that CUDA needs.
readonly CUDA_ARCHITECTURES="86;120"

# Install build dependencies on Arch Linux.
# These packages are in the official repositories, so pacman is sufficient; no
# AUR helper such as yay is needed. Obvious system tools are assumed present to
# keep this personal build script simple.
sudo pacman -S --needed --noconfirm git cmake ninja base-devel cuda

export PATH="/opt/cuda/bin:$PATH"

if [[ -d "$SRC_DIR/.git" ]]; then
  # Do not force checkout/pull here.
  # This lets the build use whatever buun-llama-cpp version is currently checked
  # out: a branch, tag, pinned commit, or local working tree. That avoids
  # accidentally replacing a deliberately selected version.
  echo "using existing buun-llama-cpp checkout: $(git -C "$SRC_DIR" describe --tags --always --dirty 2>/dev/null || git -C "$SRC_DIR" rev-parse --short HEAD)"
else
  # Clone only when the source tree does not exist yet. We intentionally avoid
  # auto-updating existing checkouts above so builds are reproducible.
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --depth 1 "$REPO_URL" "$SRC_DIR"
fi

# CMake option notes:
# - GGML_CUDA=ON enables NVIDIA CUDA acceleration.
# - GGML_NATIVE=ON lets ggml add -march=native for the CPU backend, which is
#   best for a local-only optimized build but not portable to older CPUs.
# - GGML_CUDA_FA=ON enables CUDA FlashAttention kernels; the buun README uses
#   runtime -fa with TCQ KV-cache modes, so build the kernels explicitly.
# - GGML_CUDA_FA_ALL_QUANTS=ON compiles FlashAttention support for all quantized
#   formats, which is recommended by the buun README for its TCQ workflows.
# - CMAKE_CUDA_ARCHITECTURES targets both the current RTX 5060 Ti and RTX 3070
#   rollback/future secondary GPU instead of relying on CMake's native GPU detection.
# - GGML_CUDA_NCCL=OFF avoids a harmless missing-NCCL warning on this single-GPU
#   setup; NCCL mainly matters for multi-GPU performance.
# - LLAMA_BUILD_TESTS=OFF skips test binaries to reduce build time.
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=OFF \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \

cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo
echo "buun-llama-cpp source: $SRC_DIR"
echo "buun-llama-cpp build:  $BUILD_DIR"
echo "CUDA arch:             $CUDA_ARCHITECTURES"
echo "server binary:         $BUILD_DIR/bin/llama-server"
echo "cli binary:            $BUILD_DIR/bin/llama-cli"

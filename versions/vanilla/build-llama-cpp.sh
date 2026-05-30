#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_URL="https://github.com/ggml-org/llama.cpp.git"
readonly SRC_DIR="$ROOT_DIR/src/llama.cpp"
readonly BUILD_DIR="$ROOT_DIR/build"
readonly INSTALL_PREFIX="$ROOT_DIR/install"
readonly INSTALL_RPATH='$ORIGIN/../lib'
readonly JOBS="$(nproc)"

# RTX 3070 is Ampere GA104, CUDA compute capability 8.6.
# GGML_NATIVE=ON below handles CPU-side machine-local optimization; this keeps
# only the GPU-specific architecture setting that CUDA needs.
readonly CUDA_ARCHITECTURES="86"

# Install build dependencies on Arch Linux.
# These packages are in the official repositories, so pacman is sufficient; no
# AUR helper such as yay is needed. Obvious system tools are assumed present to
# keep this personal build script simple.
sudo pacman -S --needed --noconfirm git cmake ninja base-devel cuda

export PATH="/opt/cuda/bin:$PATH"

if [[ -d "$SRC_DIR/.git" ]]; then
  # Do not force checkout/pull here.
  # This lets the build use whatever llama.cpp version is currently checked out:
  # a branch, tag, pinned commit, or local working tree. That avoids accidentally
  # replacing a deliberately selected version with latest master.
  echo "using existing llama.cpp checkout: $(git -C "$SRC_DIR" describe --tags --always --dirty 2>/dev/null || git -C "$SRC_DIR" rev-parse --short HEAD)"
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
# - GGML_CUDA_NCCL=OFF avoids a harmless missing-NCCL warning on this single-GPU
#   RTX 3070 setup; NCCL mainly matters for multi-GPU performance.
# - LLAMA_BUILD_TESTS=OFF skips test binaries to reduce build time and install
#   size when the goal is installing the llama tools/server.
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=OFF \
  -DGGML_NATIVE=ON \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \ # TODO: either remove or normalize in both build scripts
  -DCMAKE_INSTALL_RPATH="$INSTALL_RPATH"  # TODO: either remove or normalize in both build scripts

cmake --build "$BUILD_DIR" --parallel "$JOBS"
cmake --install "$BUILD_DIR"

echo
echo "llama.cpp source: $SRC_DIR"
echo "llama.cpp build:  $BUILD_DIR"
echo "installed into:   $INSTALL_PREFIX"
echo "CUDA arch:        $CUDA_ARCHITECTURES"
echo "main binary:      $INSTALL_PREFIX/bin/llama-cli"

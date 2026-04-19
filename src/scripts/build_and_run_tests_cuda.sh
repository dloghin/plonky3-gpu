#!/bin/bash
# Build and run tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${SRC_DIR}/build_cuda"

if [ "${1:-}" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    exit 0
fi

# Try to find CUDA
CUDA_PATH=""
if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}" ]; then
    CUDA_PATH="${CUDA_HOME}"
elif command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
    CUDA_PATH="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
elif [ -d "/usr/local/cuda-12.9" ]; then
    CUDA_PATH="/usr/local/cuda-12.9"
elif [ -d "/usr/local/cuda-12.6" ]; then
    CUDA_PATH="/usr/local/cuda-12.6"
elif [ -d "/usr/local/cuda-12" ]; then
    CUDA_PATH="/usr/local/cuda-12"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Configuring with CMake..."
if [ -n "${CUDA_PATH}" ]; then
    echo "Found CUDA at: ${CUDA_PATH}"
    export PATH="${CUDA_PATH}/bin:$PATH"
    cmake "$SRC_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=ON \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_BENCHMARKS=ON \
        -DCMAKE_CXX_FLAGS="-fopenmp" \
        -DENABLE_CUDA=ON \
        -DBUILD_CUDA_EXAMPLES=ON \
        -DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc"
else
    echo "CUDA not found, building CPU-only"
    cmake "$SRC_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=ON \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_BENCHMARKS=ON \
        -DCMAKE_CXX_FLAGS="-fopenmp" \
        -DENABLE_CUDA=OFF
fi

echo "Building..."
cmake --build . --parallel "$(nproc)"

echo ""
echo "Running tests..."
echo ""

ctest --output-on-failure

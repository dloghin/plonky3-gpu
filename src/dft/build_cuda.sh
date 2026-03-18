#!/bin/bash
# Build script for p3_dft with CUDA NTT support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_cuda"

if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    exit 0
fi

# Try to find CUDA
CUDA_PATH=""
if [ -n "${CUDA_HOME}" ] && [ -d "${CUDA_HOME}" ]; then
    CUDA_PATH="${CUDA_HOME}"
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
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=ON \
        -DBUILD_CUDA_EXAMPLES=ON \
        -DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc"
else
    echo "CUDA not found, building CPU-only (NttCuda will use Radix2Dit fallback)"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=ON \
        -DBUILD_CUDA_EXAMPLES=OFF
fi

echo "Building..."
cmake --build . --parallel "$(nproc)"

echo ""
echo "Build complete!"
echo ""
echo "Run tests with: cd ${BUILD_DIR} && ctest --output-on-failure"
if [ -d "${BUILD_DIR}/bin" ]; then
    echo ""
    echo "Built executables:"
    ls -la "${BUILD_DIR}/bin/"
fi

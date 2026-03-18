#!/bin/bash
# Build script for p3_fri with CUDA example support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_cuda"

# Clean previous build
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    exit 0
fi

# Try to find CUDA
CUDA_PATH=""
if [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
elif [ -d "/usr/local/cuda-12.9" ]; then
    CUDA_PATH="/usr/local/cuda-12.9"
elif [ -d "/usr/local/cuda-12.6" ]; then
    CUDA_PATH="/usr/local/cuda-12.6"
elif [ -d "/usr/local/cuda-12" ]; then
    CUDA_PATH="/usr/local/cuda-12"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo "Configuring with CMake..."
if [ -n "${CUDA_PATH}" ]; then
    echo "Found CUDA at: ${CUDA_PATH}"
    export PATH="${CUDA_PATH}/bin:$PATH"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
        -DBUILD_CUDA_EXAMPLES=ON \
        -DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc"
else
    echo "CUDA not found, building CPU-only targets"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
        -DBUILD_CUDA_EXAMPLES=OFF
fi

# Build
echo "Building..."
cmake --build . --parallel $(nproc)

echo ""
echo "Build complete!"
echo ""
echo "Binaries are in: ${BUILD_DIR}/bin/"
echo ""

# List built executables
if [ -d "${BUILD_DIR}/bin" ]; then
    echo "Built executables:"
    ls -la "${BUILD_DIR}/bin/"
fi


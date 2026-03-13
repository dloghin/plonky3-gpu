#!/bin/bash
# Build script for p3_util (CUDA wrapper for consistency)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_cuda"

# Clean previous build
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    exit 0
fi

# Try to find CUDA (informational only for util)
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
if [ -n "${CUDA_PATH}" ]; then
    echo "Found CUDA at: ${CUDA_PATH}"
    echo "Note: p3_util is currently CPU-only; building with standard C++ toolchain."
else
    echo "CUDA not found; building p3_util in CPU-only mode."
fi

echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DENABLE_WARNINGS=ON

# Build
echo "Building..."
cmake --build . --parallel $(nproc)

echo ""
echo "Build complete!"
echo ""
echo "Build directory: ${BUILD_DIR}"
echo ""

# List built test binaries
if [ -d "${BUILD_DIR}/tests" ]; then
    echo "Built test artifacts:"
    ls -la "${BUILD_DIR}/tests/"
fi

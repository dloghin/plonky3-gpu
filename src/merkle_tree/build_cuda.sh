#!/bin/bash
# Build script for p3_merkle_tree with CUDA support

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
for candidate in /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.6 /usr/local/cuda-12; do
    if [ -d "${candidate}" ]; then
        CUDA_PATH="${candidate}"
        break
    fi
done

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Configuring with CMake..."
if [ -n "${CUDA_PATH}" ]; then
    echo "Found CUDA at: ${CUDA_PATH}"
    export PATH="${CUDA_PATH}/bin:$PATH"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
        -DBUILD_CUDA_TESTS=ON \
        -DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc"
else
    echo "CUDA not found, building CPU-only"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
        -DBUILD_CUDA_TESTS=ON
fi

echo "Building..."
cmake --build . --parallel "$(nproc)"

echo ""
echo "Build complete!"
echo ""
if [ -d "${BUILD_DIR}/bin" ]; then
    echo "Built executables:"
    ls -la "${BUILD_DIR}/bin/"
    echo ""
    echo "Run: ${BUILD_DIR}/bin/merkle_tree_cuda_test"
fi

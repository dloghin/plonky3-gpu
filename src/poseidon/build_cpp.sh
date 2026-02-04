#!/bin/bash
# Build script for poseidon with CPU-only support (examples + benchmarks)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_cpp"

# Clean previous build
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    exit 0
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake (CPU-only)
echo "Configuring with CMake (CPU-only)..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_BENCHMARKS=ON \
    -DENABLE_CUDA=OFF

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


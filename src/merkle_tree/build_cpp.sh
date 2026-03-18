#!/bin/bash
# Build script for p3_merkle_tree with CPU-only support

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
    -DBUILD_TESTS=ON \
    -DENABLE_WARNINGS=ON

# Build
echo "Building..."
cmake --build . --parallel "$(nproc)"

echo ""
echo "Build complete!"
echo ""
echo "Build directory: ${BUILD_DIR}"
echo ""

if [ -d "${BUILD_DIR}/tests" ]; then
    echo "Built test artifacts:"
    ls -la "${BUILD_DIR}/tests/"
fi


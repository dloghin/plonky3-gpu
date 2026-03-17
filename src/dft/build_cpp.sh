#!/bin/bash
# Build script for p3_dft (CPU-only, C++ backend)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_cpp"

if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    exit 0
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Configuring with CMake (CPU-only)..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DENABLE_CUDA=OFF

echo "Building..."
cmake --build . --parallel "$(nproc)"

echo ""
echo "Build complete!  Run tests with: cd ${BUILD_DIR} && ctest --output-on-failure"

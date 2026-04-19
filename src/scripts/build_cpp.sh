#!/bin/bash
# Build plonky3-gpu (CPU-only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${SRC_DIR}/build_cpp"

if [ "${1:-}" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    exit 0
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Configuring with CMake (CPU-only)..."
cmake "$SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_BENCHMARKS=ON \
    -DCMAKE_CXX_FLAGS="-fopenmp" \
    -DENABLE_CUDA=OFF

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

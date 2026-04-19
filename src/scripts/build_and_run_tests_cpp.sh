#!/bin/bash
# Build and run tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${SRC_DIR}/build_cpp"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Configuring with CMake..."
cmake "$SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DCMAKE_CXX_FLAGS="-fopenmp" \
    -DENABLE_CUDA=OFF

echo "Building tests..."
cmake --build . --parallel "$(nproc)"

echo ""
echo "Running tests..."
echo ""

ctest --output-on-failure

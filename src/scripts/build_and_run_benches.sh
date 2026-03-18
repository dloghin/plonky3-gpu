#!/bin/bash
# Build and run benchmarks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${SRC_DIR}/build_cpp"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Configuring with CMake..."
cmake "$SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=ON \
    -DENABLE_CUDA=OFF

echo "Building benchmarks..."
cmake --build . --parallel "$(nproc)"

echo ""
echo "Running benchmarks..."
echo ""

if [ -f bench_field ]; then
    ./bench_field
elif [ -f benches/bench_field ]; then
    ./benches/bench_field
fi

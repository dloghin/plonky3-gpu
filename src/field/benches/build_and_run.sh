#!/bin/bash
# Build and run BabyBear field benchmarks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building BabyBear Field Benchmarks${NC}"
echo "===================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SCRIPT_DIR}/.."
BUILD_DIR="${PROJECT_ROOT}/build"

# Parse arguments
BUILD_TYPE="${1:-Release}"
RUN_BENCHMARKS="${2:-yes}"

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Build Type: ${BUILD_TYPE}"
echo "  Project Root: ${PROJECT_ROOT}"
echo "  Build Directory: ${BUILD_DIR}"

# Clean and create build directory
echo -e "\n${YELLOW}Setting up build directory...${NC}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Configure with CMake
echo -e "\n${YELLOW}Configuring with CMake...${NC}"
cd "${BUILD_DIR}"
cmake .. \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_BENCHMARKS=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF

# Build
echo -e "\n${YELLOW}Building benchmarks...${NC}"
cmake --build . --target bench_field -j$(nproc)

echo -e "\n${GREEN}✓ Build complete!${NC}"
echo -e "Benchmark executable: ${BUILD_DIR}/benches/bench_field"

# Run benchmarks if requested
if [ "${RUN_BENCHMARKS}" = "yes" ]; then
    echo -e "\n${YELLOW}Running benchmarks...${NC}"
    echo "====================================="
    ./benches/bench_field "$@"
else
    echo -e "\n${YELLOW}To run benchmarks:${NC}"
    echo "  cd ${BUILD_DIR}"
    echo "  ./benches/bench_field"
    echo ""
    echo "Example commands:"
    echo "  ./benches/bench_field --benchmark_filter=\"Add.*\""
    echo "  ./benches/bench_field --benchmark_format=json --benchmark_out=results.json"
    echo "  ./benches/bench_field --help"
fi



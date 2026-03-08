#!/bin/bash

# Build script for C++ Matrix Library
# This script builds the matrix library using CMake

set -e  # Exit on error

# Default configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-build_cpp}"
ENABLE_EXAMPLES="${ENABLE_EXAMPLES:-ON}"
ENABLE_TESTS="${ENABLE_TESTS:-ON}"
ENABLE_BENCHMARKS="${ENABLE_BENCHMARKS:-ON}"
ENABLE_WARNINGS="${ENABLE_WARNINGS:-ON}"
RUN_TESTS="${RUN_TESTS:-false}"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        --release)
            BUILD_TYPE=Release
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --no-examples)
            ENABLE_EXAMPLES=OFF
            shift
            ;;
        --no-tests)
            ENABLE_TESTS=OFF
            shift
            ;;
        --no-benchmarks)
            ENABLE_BENCHMARKS=OFF
            shift
            ;;
        --no-warnings)
            ENABLE_WARNINGS=OFF
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --clean)
            echo "Cleaning build directory..."
            rm -rf "$BUILD_DIR"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug              Build in Debug mode (default: Release)"
            echo "  --release            Build in Release mode"
            echo "  --build-dir DIR      Specify build directory (default: build)"
            echo "  --no-examples        Don't build examples"
            echo "  --no-tests           Don't build tests"
            echo "  --no-benchmarks      Don't build benchmarks"
            echo "  --no-warnings        Disable compiler warnings"
            echo "  --test               Run tests after building"
            echo "  --clean              Clean build directory before building"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  BUILD_TYPE           Build type (Debug/Release)"
            echo "  BUILD_DIR            Build directory"
            echo "  ENABLE_EXAMPLES      Enable examples (ON/OFF)"
            echo "  ENABLE_TESTS         Enable tests (ON/OFF)"
            echo "  ENABLE_BENCHMARKS    Enable benchmarks (ON/OFF)"
            echo "  ENABLE_WARNINGS      Enable warnings (ON/OFF)"
            echo "  RUN_TESTS            Run tests after build (true/false)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Building C++ Matrix Library"
echo "=========================================="
echo "Build type:      $BUILD_TYPE"
echo "Build directory: $BUILD_DIR"
echo "Examples:        $ENABLE_EXAMPLES"
echo "Tests:           $ENABLE_TESTS"
echo "Benchmarks:      $ENABLE_BENCHMARKS"
echo "Warnings:        $ENABLE_WARNINGS"
echo "=========================================="
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_EXAMPLES="$ENABLE_EXAMPLES" \
    -DBUILD_TESTS="$ENABLE_TESTS" \
    -DBUILD_BENCHMARKS="$ENABLE_BENCHMARKS" \
    -DENABLE_WARNINGS="$ENABLE_WARNINGS"

# Build
echo ""
echo "Building..."
cmake --build . -j$(nproc)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="

# Run tests if requested
if [ "$RUN_TESTS" = "true" ] && [ "$ENABLE_TESTS" = "ON" ]; then
    echo ""
    echo "Running tests..."
    ctest --output-on-failure
    echo ""
    echo "Tests completed!"
fi

echo ""
echo "Build artifacts:"
echo "  - Examples:    $BUILD_DIR/examples/"
echo "  - Tests:       $BUILD_DIR/tests/"
echo "  - Benchmarks:  $BUILD_DIR/benches/"
echo ""

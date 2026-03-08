#include <benchmark/benchmark.h>
#include "dense_matrix.hpp"
#include "util.hpp"
#include <random>

using namespace p3_matrix;

// Random number generator for reproducibility
static std::mt19937 rng(12345);

/**
 * @brief Benchmark matrix transpose operation
 *
 * This mirrors the Rust transpose benchmark which tests various matrix sizes
 */
template<size_t LOG_ROWS, size_t LOG_COLS>
static void BM_Transpose(benchmark::State& state) {
    const size_t nrows = 1ULL << LOG_ROWS;
    const size_t ncols = 1ULL << LOG_COLS;

    // Generate random matrix
    auto matrix1 = RowMajorMatrix<uint32_t>::rand(rng, nrows, ncols);
    RowMajorMatrix<uint32_t> matrix2(ncols, nrows, 0);

    // Set throughput
    size_t bytes = nrows * ncols * sizeof(uint32_t);
    state.SetBytesProcessed(static_cast<int64_t>(bytes * state.iterations()));

    for (auto _ : state) {
        matrix1.transpose_into(matrix2);
        benchmark::DoNotOptimize(matrix2.values.data());
    }
}

// Small matrices (for quick tests)
static void BM_Transpose_4x4(benchmark::State& state) {
    BM_Transpose<4, 4>(state);
}
BENCHMARK(BM_Transpose_4x4);

static void BM_Transpose_8x8(benchmark::State& state) {
    BM_Transpose<8, 8>(state);
}
BENCHMARK(BM_Transpose_8x8);

static void BM_Transpose_10x10(benchmark::State& state) {
    BM_Transpose<10, 10>(state);
}
BENCHMARK(BM_Transpose_10x10);

static void BM_Transpose_12x12(benchmark::State& state) {
    BM_Transpose<12, 12>(state);
}
BENCHMARK(BM_Transpose_12x12);

// Large matrices (as in Rust version)
static void BM_Transpose_20x8(benchmark::State& state) {
    BM_Transpose<20, 8>(state);
}
BENCHMARK(BM_Transpose_20x8);

static void BM_Transpose_8x20(benchmark::State& state) {
    BM_Transpose<8, 20>(state);
}
BENCHMARK(BM_Transpose_8x20);

static void BM_Transpose_21x8(benchmark::State& state) {
    BM_Transpose<21, 8>(state);
}
BENCHMARK(BM_Transpose_21x8);

static void BM_Transpose_8x21(benchmark::State& state) {
    BM_Transpose<8, 21>(state);
}
BENCHMARK(BM_Transpose_8x21);

static void BM_Transpose_22x8(benchmark::State& state) {
    BM_Transpose<22, 8>(state);
}
BENCHMARK(BM_Transpose_22x8);

static void BM_Transpose_8x22(benchmark::State& state) {
    BM_Transpose<8, 22>(state);
}
BENCHMARK(BM_Transpose_8x22);

static void BM_Transpose_23x8(benchmark::State& state) {
    BM_Transpose<23, 8>(state);
}
BENCHMARK(BM_Transpose_23x8);

static void BM_Transpose_8x23(benchmark::State& state) {
    BM_Transpose<8, 23>(state);
}
BENCHMARK(BM_Transpose_8x23);

BENCHMARK_MAIN();


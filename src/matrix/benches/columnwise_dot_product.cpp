#include <benchmark/benchmark.h>
#include "dense_matrix.hpp"
#include "util.hpp"
#include <random>

using namespace p3_matrix;

// Random number generator for reproducibility
static std::mt19937_64 rng(12345);

/**
 * @brief Generate a random non-zero value
 */
static uint32_t random_nonzero() {
    uint32_t val;
    do {
        val = static_cast<uint32_t>(rng());
    } while (val == 0);
    return val;
}

/**
 * @brief Benchmark columnwise dot product operation
 *
 * This mirrors the Rust columnwise_dot_product benchmark.
 * Computes M^T * v where M is a matrix and v is a vector.
 */
static void BM_ColumnwiseDotProduct(benchmark::State& state) {
    const size_t LOG_ROWS = 16;  // 2^16 = 65536 rows
    const size_t LOG_COLS = 12;  // 2^12 = 4096 columns
    const size_t nrows = 1ULL << LOG_ROWS;
    const size_t ncols = 1ULL << LOG_COLS;

    for (auto _ : state) {
        state.PauseTiming();

        // Generate random matrix with non-zero values
        std::vector<uint32_t> matrix_vals(nrows * ncols);
        for (auto& v : matrix_vals) {
            v = random_nonzero();
        }
        RowMajorMatrix<uint32_t> mat(std::move(matrix_vals), ncols);

        // Generate random vector with non-zero values
        std::vector<uint32_t> vec(nrows);
        for (auto& v : vec) {
            v = random_nonzero();
        }

        state.ResumeTiming();

        auto result = columnwise_dot_product(mat, vec);
        benchmark::DoNotOptimize(result.data());
    }
}
BENCHMARK(BM_ColumnwiseDotProduct)
    ->Unit(benchmark::kMillisecond);

/**
 * @brief Benchmark with smaller matrix for quicker iteration
 */
static void BM_ColumnwiseDotProduct_Small(benchmark::State& state) {
    const size_t nrows = 1024;
    const size_t ncols = 256;

    for (auto _ : state) {
        state.PauseTiming();

        std::vector<uint32_t> matrix_vals(nrows * ncols);
        for (auto& v : matrix_vals) {
            v = random_nonzero();
        }
        RowMajorMatrix<uint32_t> mat(std::move(matrix_vals), ncols);

        std::vector<uint32_t> vec(nrows);
        for (auto& v : vec) {
            v = random_nonzero();
        }

        state.ResumeTiming();

        auto result = columnwise_dot_product(mat, vec);
        benchmark::DoNotOptimize(result.data());
    }
}
BENCHMARK(BM_ColumnwiseDotProduct_Small);

/**
 * @brief Benchmark with medium matrix
 */
static void BM_ColumnwiseDotProduct_Medium(benchmark::State& state) {
    const size_t nrows = 8192;   // 2^13
    const size_t ncols = 1024;   // 2^10

    for (auto _ : state) {
        state.PauseTiming();

        std::vector<uint32_t> matrix_vals(nrows * ncols);
        for (auto& v : matrix_vals) {
            v = random_nonzero();
        }
        RowMajorMatrix<uint32_t> mat(std::move(matrix_vals), ncols);

        std::vector<uint32_t> vec(nrows);
        for (auto& v : vec) {
            v = random_nonzero();
        }

        state.ResumeTiming();

        auto result = columnwise_dot_product(mat, vec);
        benchmark::DoNotOptimize(result.data());
    }
}
BENCHMARK(BM_ColumnwiseDotProduct_Medium);

BENCHMARK_MAIN();


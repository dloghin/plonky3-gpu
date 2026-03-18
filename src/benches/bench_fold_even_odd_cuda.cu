#include <benchmark/benchmark.h>
#include "fri_fold_cuda.hpp"
#include "baby_bear.hpp"
#include "extension_field.hpp"
#include <random>
#include <vector>

using namespace p3_field;
using namespace p3_fri;

// CUDA counterpart of bench_fold_even_odd.cpp.
// Benchmarks fold_matrix_cuda which dispatches to GPU kernels when available,
// falling back to the CPU implementation otherwise.
//
// Note: BabyBear-as-challenge (F==EF) is omitted because the CUDA kernels
// require halve() and from_base() which are only available on extension fields.

// --------------------------------------------------------------------------
// BabyBear4 (F = BabyBear, EF = BinomialExtensionField<BabyBear, 4, 11>)
// --------------------------------------------------------------------------
static void BM_FoldMatrixCuda_BabyBear4(benchmark::State& state) {
    const size_t log_size = static_cast<size_t>(state.range(0));
    const size_t n = size_t(1) << log_size;
    const size_t log_arity = 1;
    const size_t log_height = log_size + log_arity;

    std::mt19937_64 local_rng(n);
    std::uniform_int_distribution<uint32_t> dist(0, BabyBear::PRIME - 1);

    std::array<BabyBear, 4> beta_coeffs;
    for (auto& c : beta_coeffs) {
        c = BabyBear(dist(local_rng));
    }
    BabyBear4 beta(beta_coeffs);

    std::vector<BabyBear4> mat(2 * n);
    for (auto& v : mat) {
        std::array<BabyBear, 4> coeffs;
        for (auto& c : coeffs) {
            c = BabyBear(dist(local_rng));
        }
        v = BabyBear4(coeffs);
    }

    // Warm-up GPU
    fold_matrix_cuda<BabyBear, BabyBear4>(beta, log_arity, log_height, mat);

    for (auto _ : state) {
        auto result = fold_matrix_cuda<BabyBear, BabyBear4>(
            beta, log_arity, log_height, mat);
        benchmark::DoNotOptimize(result.data());
    }
}

BENCHMARK(BM_FoldMatrixCuda_BabyBear4)
    ->Args({12})->Args({14})->Args({16})->Args({18})->Args({20})->Args({22})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

BENCHMARK_MAIN();

#include <benchmark/benchmark.h>
#include "fri_folding.hpp"
#include "baby_bear.hpp"
#include "extension_field.hpp"
#include <random>
#include <vector>

using namespace p3_field;
using namespace p3_fri;

// Mirrors plonky3/fri/benches/fold_even_odd.rs
//
// The Rust benchmark creates an n×2 RowMajorMatrix of random EF elements,
// then calls fold_matrix(beta, log_arity=1, mat).  In our C++ API the matrix
// is represented as a flat vector of length 2*n and fold_matrix takes
// (log_height, log_arity, beta, current).

// --------------------------------------------------------------------------
// BabyBear (F = EF = BabyBear)
// --------------------------------------------------------------------------
static void BM_FoldMatrix_BabyBear(benchmark::State& state) {
    const size_t log_size = static_cast<size_t>(state.range(0));
    const size_t n = size_t(1) << log_size;
    const size_t log_arity = 1;
    const size_t log_height = log_size + log_arity; // total length = n * arity

    // Generate random input: flat vector of length 2*n
    std::mt19937_64 local_rng(n);
    std::uniform_int_distribution<uint32_t> dist(0, BabyBear::PRIME - 1);

    BabyBear beta(dist(local_rng));
    std::vector<BabyBear> mat(2 * n);
    for (auto& v : mat) {
        v = BabyBear(dist(local_rng));
    }

    for (auto _ : state) {
        auto result = TwoAdicFriFolding<BabyBear, BabyBear>::fold_matrix(
            log_height, log_arity, beta, mat);
        benchmark::DoNotOptimize(result.data());
    }
}

BENCHMARK(BM_FoldMatrix_BabyBear)
    ->Args({12})->Args({14})->Args({16})->Args({18})->Args({20})->Args({22})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// --------------------------------------------------------------------------
// BabyBear4 (F = BabyBear, EF = BinomialExtensionField<BabyBear, 4, 11>)
// --------------------------------------------------------------------------
static void BM_FoldMatrix_BabyBear4(benchmark::State& state) {
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

    for (auto _ : state) {
        auto result = TwoAdicFriFolding<BabyBear, BabyBear4>::fold_matrix(
            log_height, log_arity, beta, mat);
        benchmark::DoNotOptimize(result.data());
    }
}

BENCHMARK(BM_FoldMatrix_BabyBear4)
    ->Args({12})->Args({14})->Args({16})->Args({18})->Args({20})->Args({22})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

BENCHMARK_MAIN();

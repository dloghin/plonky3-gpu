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

// Helper to generate random field elements.
template<typename T> T random_element(std::mt19937_64& rng);

template<>
BabyBear random_element<BabyBear>(std::mt19937_64& rng) {
    std::uniform_int_distribution<uint32_t> dist(0, BabyBear::PRIME - 1);
    return BabyBear(dist(rng));
}

template<>
BabyBear4 random_element<BabyBear4>(std::mt19937_64& rng) {
    std::uniform_int_distribution<uint32_t> dist(0, BabyBear::PRIME - 1);
    std::array<BabyBear, 4> coeffs;
    for (auto& c : coeffs) {
        c = BabyBear(dist(rng));
    }
    return BabyBear4(coeffs);
}

template <typename Challenge>
static void BM_FoldMatrix(benchmark::State& state) {
    using Val = BabyBear;
    const size_t log_size = static_cast<size_t>(state.range(0));
    const size_t n = size_t(1) << log_size;
    const size_t log_arity = 1;
    const size_t log_height = log_size + log_arity;

    std::mt19937_64 local_rng(n);

    Challenge beta = random_element<Challenge>(local_rng);
    std::vector<Challenge> mat(2 * n);
    for (auto& v : mat) {
        v = random_element<Challenge>(local_rng);
    }

    for (auto _ : state) {
        auto result = TwoAdicFriFolding<Val, Challenge>::fold_matrix(
            log_height, log_arity, beta, mat);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK_TEMPLATE(BM_FoldMatrix, BabyBear)
    ->Args({12})->Args({14})->Args({16})->Args({18})->Args({20})->Args({22})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_FoldMatrix, BabyBear4)
    ->Args({12})->Args({14})->Args({16})->Args({18})->Args({20})->Args({22})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

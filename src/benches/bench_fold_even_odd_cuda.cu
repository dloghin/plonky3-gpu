#include <benchmark/benchmark.h>
#include "fri_fold_cuda.hpp"
#include "baby_bear.hpp"
#include "extension_field.hpp"
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace p3_field;
using namespace p3_fri;

// CUDA counterpart of bench_fold_even_odd.cpp.
// Times only the FRI fold kernels on pre-allocated device buffers (launch +
// cudaDeviceSynchronize). Host alloc, H2D setup, and D2H teardown are outside
// the timed region so results reflect folding work, not cudaMalloc/Memcpy/Free.
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

    BabyBear4* d_input = nullptr;
    BabyBear4* d_output = nullptr;
    P3_CUDA_CHECK(cudaMalloc(&d_input, mat.size() * sizeof(BabyBear4)));
    P3_CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(BabyBear4)));
    P3_CUDA_CHECK(cudaMemcpy(
        d_input, mat.data(), mat.size() * sizeof(BabyBear4), cudaMemcpyHostToDevice));

    fold_matrix_cuda_device<BabyBear, BabyBear4>(
        d_input, d_output, beta, log_arity, log_height, n);

    for (auto _ : state) {
        fold_matrix_cuda_device<BabyBear, BabyBear4>(
            d_input, d_output, beta, log_arity, log_height, n);
        benchmark::DoNotOptimize(d_output);
    }

    P3_CUDA_CHECK(cudaFree(d_input));
    P3_CUDA_CHECK(cudaFree(d_output));
}

BENCHMARK(BM_FoldMatrixCuda_BabyBear4)
    ->Args({12})->Args({14})->Args({16})->Args({18})->Args({20})->Args({22})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

BENCHMARK_MAIN();

/**
 * @file fri_fold_cuda.cu
 * @brief Correctness verification and performance benchmark for the CUDA FRI fold_matrix kernel.
 *
 * For each test configuration the GPU result is compared element-by-element
 * with the reference CPU result from TwoAdicFriFolding::fold_matrix.
 *
 * Tests:
 *   1. Arity-2 fold with BabyBear4 challenge (small size)
 *   2. Arity-4 fold with BabyBear4 challenge
 *   3. Arity-8 fold with BabyBear4 challenge
 *   4. Arity-2 fold with large input (n = 2^20)
 *   5. Performance benchmark: arity-2, n = 2^20
 */

#include "fri_fold_cuda.hpp"
#include "fri_folding.hpp"
#include "baby_bear.hpp"
#include "extension_field.hpp"
#include "p3_util/util.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>

using namespace p3_field;
using namespace p3_fri;
using namespace p3_util;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using Val = BabyBear;
using EF  = BabyBear4;           // BinomialExtensionField<BabyBear, 4, 11>

// ---------------------------------------------------------------------------
// CUDA error-check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print_pass_fail(const char* test_name, bool pass) {
    printf("  %-60s %s\n", test_name, pass ? "PASS" : "FAIL");
}

/** Generate a random BabyBear4 element. */
static EF random_ef(std::mt19937& rng) {
    return EF({
        Val(static_cast<uint32_t>(rng() % Val::PRIME)),
        Val(static_cast<uint32_t>(rng() % Val::PRIME)),
        Val(static_cast<uint32_t>(rng() % Val::PRIME)),
        Val(static_cast<uint32_t>(rng() % Val::PRIME))
    });
}

/** Generate a random BabyBear4 vector of length `n`. */
static std::vector<EF> random_ef_vec(size_t n, std::mt19937& rng) {
    std::vector<EF> v(n);
    for (auto& e : v) e = random_ef(rng);
    return v;
}

/** Check two EF vectors for equality. */
static bool vectors_equal(const std::vector<EF>& a, const std::vector<EF>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Correctness tests
// ---------------------------------------------------------------------------

static int run_correctness_tests() {
    printf("\n=== Correctness Tests ===\n");
    int failures = 0;
    std::mt19937 rng(0xdeadbeef);

    // Sizes and arities to test.
    // log_height is the log2 of the total (unfolded) vector length.
    struct TestCase {
        size_t log_height;
        size_t log_arity;
        const char* label;
    };

    TestCase cases[] = {
        {  4, 1, "arity=2  n=8   (log_height=4, log_arity=1)" },
        {  6, 1, "arity=2  n=32  (log_height=6, log_arity=1)" },
        { 10, 1, "arity=2  n=512 (log_height=10, log_arity=1)"},
        {  5, 2, "arity=4  n=8   (log_height=5, log_arity=2)" },
        {  7, 2, "arity=4  n=32  (log_height=7, log_arity=2)" },
        {  6, 3, "arity=8  n=8   (log_height=6, log_arity=3)" },
        {  9, 3, "arity=8  n=64  (log_height=9, log_arity=3)" },
    };

    for (const auto& tc : cases) {
        size_t total = size_t(1) << tc.log_height;  // total elements
        auto   input = random_ef_vec(total, rng);
        EF     beta  = random_ef(rng);

        // CPU reference
        auto cpu_out = TwoAdicFriFolding<Val, EF>::fold_matrix(
            tc.log_height, tc.log_arity, beta, input);

        // GPU (or CPU fallback)
        auto gpu_out = fold_matrix_cuda<Val, EF>(
            beta, tc.log_arity, tc.log_height, input);

        bool ok = vectors_equal(cpu_out, gpu_out);
        print_pass_fail(tc.label, ok);
        if (!ok) {
            ++failures;
            // Print first mismatch for debugging
            for (size_t i = 0; i < cpu_out.size(); ++i) {
                if (cpu_out[i] != gpu_out[i]) {
                    printf("    First mismatch at index %zu:\n", i);
                    printf("      CPU: [%u, %u, %u, %u]\n",
                           cpu_out[i][0].value(), cpu_out[i][1].value(),
                           cpu_out[i][2].value(), cpu_out[i][3].value());
                    printf("      GPU: [%u, %u, %u, %u]\n",
                           gpu_out[i][0].value(), gpu_out[i][1].value(),
                           gpu_out[i][2].value(), gpu_out[i][3].value());
                    break;
                }
            }
        }
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Large-input correctness test
// ---------------------------------------------------------------------------

static int run_large_test() {
    printf("\n=== Large-input Correctness Test (n = 2^20, arity=2) ===\n");

    std::mt19937 rng(0xbeefdead);
    size_t log_height = 20;
    size_t log_arity  = 1;
    size_t total      = size_t(1) << log_height;

    auto input = random_ef_vec(total, rng);
    EF   beta  = random_ef(rng);

    // CPU reference
    auto t0 = std::chrono::high_resolution_clock::now();
    auto cpu_out = TwoAdicFriFolding<Val, EF>::fold_matrix(
        log_height, log_arity, beta, input);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU
    auto t2 = std::chrono::high_resolution_clock::now();
    auto gpu_out = fold_matrix_cuda<Val, EF>(
        beta, log_arity, log_height, input);
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    bool ok = vectors_equal(cpu_out, gpu_out);
    printf("  %-60s %s\n", "arity=2  n=2^20  (correctness)", ok ? "PASS" : "FAIL");
    printf("  CPU time: %.2f ms,  GPU time: %.2f ms", cpu_ms, gpu_ms);
    if (gpu_ms > 0 && cpu_ms > 0) {
        printf(",  speedup: %.2fx", cpu_ms / gpu_ms);
    }
    printf("\n");

    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Performance benchmark
// ---------------------------------------------------------------------------

static void run_benchmark() {
    printf("\n=== Performance Benchmark ===\n");
    printf("  %-10s  %-8s  %-12s  %-12s  %-10s\n",
           "log_height", "arity", "CPU (ms)", "GPU (ms)", "Speedup");
    printf("  %s\n", std::string(60, '-').c_str());

    std::mt19937 rng(0xcafebabe);

    size_t log_arities[] = {1, 2, 3};

    for (size_t log_height : {16u, 18u, 20u}) {
        for (size_t log_arity : log_arities) {
            if (log_arity >= log_height) continue;

            size_t total = size_t(1) << log_height;
            auto input   = random_ef_vec(total, rng);
            EF   beta    = random_ef(rng);

            // CPU
            auto t0 = std::chrono::high_resolution_clock::now();
            auto cpu_out = TwoAdicFriFolding<Val, EF>::fold_matrix(
                log_height, log_arity, beta, input);
            auto t1 = std::chrono::high_resolution_clock::now();
            double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // GPU (warm-up: run once, then time)
            fold_matrix_cuda<Val, EF>(beta, log_arity, log_height, input);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto gpu_out = fold_matrix_cuda<Val, EF>(beta, log_arity, log_height, input);
            auto t3 = std::chrono::high_resolution_clock::now();
            double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

            double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0.0;
            printf("  %-10zu  %-8zu  %-12.2f  %-12.2f  %-10.2f\n",
                   log_height, size_t(1) << log_arity,
                   cpu_ms, gpu_ms, speedup);
            (void)cpu_out; (void)gpu_out;  // suppress unused-variable warning
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    printf("=============================================================\n");
    printf("  FRI fold_matrix CUDA Correctness & Benchmark\n");
    printf("  Val = BabyBear, Challenge = BabyBear4\n");
    printf("=============================================================\n");

    // Check CUDA availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("\nNo CUDA device found — all tests run on CPU fallback.\n");
    } else {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("\nCUDA device: %s (SM %d.%d, %zu MB global mem)\n",
               prop.name, prop.major, prop.minor,
               static_cast<size_t>(prop.totalGlobalMem) >> 20);
    }

    int total_failures = 0;
    total_failures += run_correctness_tests();
    total_failures += run_large_test();
    run_benchmark();

    printf("\n=============================================================\n");
    if (total_failures == 0) {
        printf("  All correctness tests PASSED.\n");
    } else {
        printf("  %d test(s) FAILED.\n", total_failures);
    }
    printf("=============================================================\n");

    return (total_failures == 0) ? 0 : 1;
}

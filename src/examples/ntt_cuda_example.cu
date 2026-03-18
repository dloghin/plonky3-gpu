/**
 * @file ntt_cuda.cu
 * @brief CUDA NTT example: correctness verification and performance benchmark.
 *
 * Demonstrates the NttCuda<BabyBear> class against the reference Radix2Dit<BabyBear>:
 *
 *   Test 1 – Forward NTT matches CPU
 *   Test 2 – INTT(NTT(x)) == x
 *   Test 3 – Coset DFT matches CPU
 *   Test 4 – coset_lde_batch matches CPU
 *   Test 5 – Batch (multi-column) correctness
 *   Test 6 – Performance benchmark for n = 2^16 .. 2^20
 */

#include "ntt_cuda.hpp"
#include "radix2_dit.hpp"
#include "baby_bear.hpp"
#include "dense_matrix.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

using namespace p3_field;
using namespace p3_matrix;
using namespace p3_dft;

// ============================================================
// CUDA error-check macro
// ============================================================

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_err));         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// ============================================================
// Helpers
// ============================================================

static RowMajorMatrix<BabyBear> make_random_matrix(
    size_t height, size_t width, std::mt19937& rng)
{
    std::vector<BabyBear> vals(height * width);
    for (auto& v : vals) {
        v = BabyBear(static_cast<uint32_t>(rng() % BabyBear::PRIME));
    }
    return RowMajorMatrix<BabyBear>(std::move(vals), width);
}

static bool matrices_equal(
    const RowMajorMatrix<BabyBear>& a,
    const RowMajorMatrix<BabyBear>& b)
{
    if (a.height() != b.height() || a.width() != b.width()) return false;
    for (size_t r = 0; r < a.height(); ++r) {
        for (size_t c = 0; c < a.width(); ++c) {
            if (a.get_unchecked(r, c) != b.get_unchecked(r, c)) return false;
        }
    }
    return true;
}

static void print_pass_fail(const char* test_name, bool pass) {
    printf("  %-45s %s\n", test_name, pass ? "PASS" : "FAIL");
}

// ============================================================
// Correctness tests
// ============================================================

static int run_correctness_tests() {
    printf("\n=== Correctness Tests ===\n");
    int failures = 0;

    NttCuda<BabyBear>    gpu_ntt;
    Radix2Dit<BabyBear>  cpu_ntt;
    std::mt19937         rng(0xdeadbeef);

    BabyBear shift(BabyBear::GENERATOR_VAL);

    // -----------------------------------------------------------------
    // Test 1: Forward NTT matches CPU  (several sizes, 1 and 4 columns)
    // -----------------------------------------------------------------
    for (size_t log_h : {2u, 4u, 6u, 8u, 10u}) {
        for (size_t w : {1u, 4u}) {
            size_t h   = 1u << log_h;
            auto   mat = make_random_matrix(h, w, rng);
            auto   mat_cpu = mat;
            auto   mat_gpu = mat;

            auto cpu_out = cpu_ntt.dft_batch(std::move(mat_cpu));
            auto gpu_out = gpu_ntt.dft_batch(std::move(mat_gpu));

            char name[64];
            snprintf(name, sizeof(name), "dft_batch n=%zu w=%zu", h, w);
            bool pass = matrices_equal(cpu_out, gpu_out);
            print_pass_fail(name, pass);
            failures += pass ? 0 : 1;
        }
    }

    // -----------------------------------------------------------------
    // Test 2: INTT(NTT(x)) == x   (round-trip identity)
    // -----------------------------------------------------------------
    for (size_t log_h : {2u, 5u, 8u, 10u}) {
        for (size_t w : {1u, 3u}) {
            size_t h   = 1u << log_h;
            auto   mat = make_random_matrix(h, w, rng);
            auto   original = mat;

            auto recovered = gpu_ntt.idft_batch(gpu_ntt.dft_batch(mat));

            char name[64];
            snprintf(name, sizeof(name), "idft(dft(x))==x n=%zu w=%zu", h, w);
            bool pass = matrices_equal(recovered, original);
            print_pass_fail(name, pass);
            failures += pass ? 0 : 1;
        }
    }

    // -----------------------------------------------------------------
    // Test 3: Coset DFT matches CPU
    // -----------------------------------------------------------------
    for (size_t log_h : {3u, 6u, 10u}) {
        size_t h = 1u << log_h;
        for (size_t w : {1u, 4u}) {
            auto mat = make_random_matrix(h, w, rng);

            auto cpu_out = cpu_ntt.coset_dft_batch(mat, shift);
            auto gpu_out = gpu_ntt.coset_dft_batch(mat, shift);

            char name[64];
            snprintf(name, sizeof(name), "coset_dft n=%zu w=%zu", h, w);
            bool pass = matrices_equal(cpu_out, gpu_out);
            print_pass_fail(name, pass);
            failures += pass ? 0 : 1;
        }
    }

    // -----------------------------------------------------------------
    // Test 4: coset_lde_batch matches CPU
    // -----------------------------------------------------------------
    for (size_t log_h : {3u, 6u, 8u}) {
        size_t h = 1u << log_h;
        for (size_t added : {1u, 2u}) {
            for (size_t w : {1u, 4u}) {
                auto mat = make_random_matrix(h, w, rng);

                auto cpu_lde = cpu_ntt.coset_lde_batch(mat, added, shift);
                auto gpu_lde = gpu_ntt.coset_lde_batch(mat, added, shift);

                char name[64];
                snprintf(name, sizeof(name),
                         "coset_lde n=%zu +bits=%zu w=%zu", h, added, w);
                bool pass = matrices_equal(cpu_lde, gpu_lde);
                print_pass_fail(name, pass);
                failures += pass ? 0 : 1;
            }
        }
    }

    // -----------------------------------------------------------------
    // Test 5: Coset IDFT matches CPU
    // -----------------------------------------------------------------
    for (size_t log_h : {3u, 8u}) {
        size_t h = 1u << log_h;
        auto   mat = make_random_matrix(h, 2, rng);

        auto cpu_out = cpu_ntt.coset_idft_batch(mat, shift);
        auto gpu_out = gpu_ntt.coset_idft_batch(mat, shift);

        char name[64];
        snprintf(name, sizeof(name), "coset_idft n=%zu", h);
        bool pass = matrices_equal(cpu_out, gpu_out);
        print_pass_fail(name, pass);
        failures += pass ? 0 : 1;
    }

    // -----------------------------------------------------------------
    // Test 6: Round-trip  coset_idft(coset_dft(x)) == x
    // -----------------------------------------------------------------
    {
        size_t h  = 1u << 8;
        size_t w  = 4;
        auto   mat = make_random_matrix(h, w, rng);
        auto   orig = mat;

        auto recovered = gpu_ntt.coset_idft_batch(
                             gpu_ntt.coset_dft_batch(mat, shift), shift);
        bool pass = matrices_equal(recovered, orig);
        print_pass_fail("coset_idft(coset_dft(x))==x n=256 w=4", pass);
        failures += pass ? 0 : 1;
    }

    return failures;
}

// ============================================================
// Performance benchmark
// ============================================================

static void run_benchmark() {
    printf("\n=== Performance Benchmark (GPU vs CPU) ===\n");
    printf("  %-10s  %-6s  %-12s  %-12s  %-8s\n",
           "n", "width", "GPU (ms)", "CPU (ms)", "Speedup");
    printf("  %s\n", std::string(60, '-').c_str());

    NttCuda<BabyBear>    gpu_ntt;
    Radix2Dit<BabyBear>  cpu_ntt;
    std::mt19937         rng(42);

    for (size_t log_h = 12; log_h <= 20; log_h += 2) {
        size_t h = 1u << log_h;
        size_t w = 16;

        auto mat = make_random_matrix(h, w, rng);

        // ----- GPU -----
        // Warm-up
        {
            auto tmp = mat;
            gpu_ntt.dft_batch(std::move(tmp));
        }
        cudaDeviceSynchronize();

        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        auto tmp_gpu = mat;
        CUDA_CHECK(cudaEventRecord(ev_start));
        gpu_ntt.dft_batch(std::move(tmp_gpu));
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

        // ----- CPU -----
        auto tmp_cpu = mat;
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_ntt.dft_batch(std::move(tmp_cpu));
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("  2^%-7zu  %-6zu  %-12.3f  %-12.3f  %-8.1fx\n",
               log_h, w, gpu_ms, cpu_ms, cpu_ms / gpu_ms);
    }
}

// ============================================================
// main
// ============================================================

int main() {
    printf("=========================================\n");
    printf("  NttCuda CUDA NTT Example - BabyBear\n");
    printf("  Prime: 2^31 - 2^27 + 1 = %u\n", BabyBear::PRIME);
    printf("=========================================\n");

    // Print GPU info
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        printf("\nGPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    } else {
        printf("No CUDA-capable GPU found; GPU tests will use CPU fallback.\n");
    }

    int failures = run_correctness_tests();

    if (failures == 0) {
        printf("\nAll correctness tests PASSED.\n");
    } else {
        printf("\n%d correctness test(s) FAILED.\n", failures);
    }

    run_benchmark();

    printf("\n=========================================\n");
    printf("  NTT CUDA example done!\n");
    printf("=========================================\n");

    return (failures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

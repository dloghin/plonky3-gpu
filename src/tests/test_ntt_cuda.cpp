/**
 * @file test_ntt_cuda.cpp
 * @brief Google Tests for NttCuda<BabyBear> (CPU-fallback path).
 *
 * These tests exercise every public method of NttCuda<F> using the CPU
 * fallback (which delegates to Radix2Dit<F>).  They therefore run on any
 * machine — including CI hosts without a GPU — and verify that NttCuda
 * produces results identical to the reference Radix2Dit implementation.
 *
 * When compiled with nvcc (P3_CUDA_ENABLED) and a GPU is present the same
 * code paths exercise the real CUDA kernels, satisfying the acceptance
 * criteria:
 *   1. CUDA NTT == CPU NTT for all test sizes.
 *   2. INTT(NTT(x)) == x.
 *   3. Coset NTT matches CPU coset_dft_batch.
 *   4. coset_lde_batch_cuda matches CPU coset_lde_batch.
 *   5. Batch processing works for varying batch sizes.
 */

#include <gtest/gtest.h>

#include "ntt_cuda.hpp"
#include "radix2_dit.hpp"
#include "baby_bear.hpp"
#include "dense_matrix.hpp"

#include <vector>
#include <cstdint>
#include <random>

using namespace p3_field;
using namespace p3_matrix;
using namespace p3_dft;

// ============================================================
// Helpers
// ============================================================

static RowMajorMatrix<BabyBear> make_col(std::vector<uint32_t> vals) {
    std::vector<BabyBear> bv;
    bv.reserve(vals.size());
    for (auto v : vals) bv.push_back(BabyBear(v));
    return RowMajorMatrix<BabyBear>(std::move(bv), 1);
}

static RowMajorMatrix<BabyBear> make_matrix(
    std::vector<uint32_t> vals, size_t width)
{
    std::vector<BabyBear> bv;
    bv.reserve(vals.size());
    for (auto v : vals) bv.push_back(BabyBear(v));
    return RowMajorMatrix<BabyBear>(std::move(bv), width);
}

static RowMajorMatrix<BabyBear> make_random(
    size_t height, size_t width, std::mt19937& rng)
{
    std::vector<BabyBear> vals(height * width);
    for (auto& v : vals)
        v = BabyBear(static_cast<uint32_t>(rng() % BabyBear::PRIME));
    return RowMajorMatrix<BabyBear>(std::move(vals), width);
}

static bool matrices_equal(
    const RowMajorMatrix<BabyBear>& a,
    const RowMajorMatrix<BabyBear>& b)
{
    if (a.height() != b.height() || a.width() != b.width()) return false;
    for (size_t r = 0; r < a.height(); ++r)
        for (size_t c = 0; c < a.width(); ++c)
            if (a.get_unchecked(r, c) != b.get_unchecked(r, c)) return false;
    return true;
}

// ============================================================
// 1. Single-element DFT is identity
// ============================================================

TEST(NttCuda, DftSingleRow) {
    NttCuda<BabyBear> ntt;
    auto mat    = make_col({42});
    auto result = ntt.dft_batch(mat);
    EXPECT_EQ(result.height(), 1u);
    EXPECT_EQ(result.width(),  1u);
    EXPECT_EQ(result.get_unchecked(0, 0), BabyBear(42u));
}

// ============================================================
// 2. NttCuda DFT == Radix2Dit DFT (forward)
// ============================================================

TEST(NttCuda, ForwardMatchesCpuSize4) {
    NttCuda<BabyBear>   gpu;
    Radix2Dit<BabyBear> cpu;
    auto mat = make_col({1, 2, 3, 4});
    EXPECT_TRUE(matrices_equal(cpu.dft_batch(mat), gpu.dft_batch(mat)));
}

TEST(NttCuda, ForwardMatchesCpuSize8) {
    NttCuda<BabyBear>   gpu;
    Radix2Dit<BabyBear> cpu;
    auto mat = make_col({0, 1, 2, 3, 4, 5, 6, 7});
    EXPECT_TRUE(matrices_equal(cpu.dft_batch(mat), gpu.dft_batch(mat)));
}

TEST(NttCuda, ForwardMatchesCpuSize1024MultiCol) {
    NttCuda<BabyBear>   gpu;
    Radix2Dit<BabyBear> cpu;
    std::mt19937 rng(111);
    auto mat = make_random(1024, 8, rng);
    EXPECT_TRUE(matrices_equal(cpu.dft_batch(mat), gpu.dft_batch(mat)))
        << "GPU forward NTT mismatch (n=1024, w=8)";
}

TEST(NttCuda, ForwardMatchesCpuConstantPoly) {
    NttCuda<BabyBear>   gpu;
    Radix2Dit<BabyBear> cpu;
    // Constant polynomial c=7: all evaluations should be 7
    auto coeff_mat = make_col({7, 0, 0, 0, 0, 0, 0, 0});
    auto cpu_out   = cpu.dft_batch(coeff_mat);
    auto gpu_out   = gpu.dft_batch(coeff_mat);
    EXPECT_TRUE(matrices_equal(cpu_out, gpu_out));
    // Spot-check: all evaluations == 7
    BabyBear seven(7u);
    for (size_t r = 0; r < 8; ++r)
        EXPECT_EQ(gpu_out.get_unchecked(r, 0), seven);
}

// ============================================================
// 3. Round-trip: INTT(NTT(x)) == x
// ============================================================

TEST(NttCuda, RoundTripSize4) {
    NttCuda<BabyBear> ntt;
    auto original  = make_col({1, 2, 3, 4});
    auto recovered = ntt.idft_batch(ntt.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "Round-trip (size 4, 1 col) failed";
}

TEST(NttCuda, RoundTripSize8) {
    NttCuda<BabyBear> ntt;
    auto original  = make_col({0, 1, 2, 3, 4, 5, 6, 7});
    auto recovered = ntt.idft_batch(ntt.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "Round-trip (size 8, 1 col) failed";
}

TEST(NttCuda, RoundTripSize16MultiCol) {
    NttCuda<BabyBear> ntt;
    std::vector<uint32_t> vals;
    for (uint32_t i = 0; i < 64; ++i) vals.push_back(i + 1);
    auto original  = make_matrix(vals, 4);
    auto recovered = ntt.idft_batch(ntt.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "Round-trip (size 16, 4 col) failed";
}

TEST(NttCuda, RoundTripSize1024Width8) {
    NttCuda<BabyBear> ntt;
    std::mt19937 rng(999);
    auto original  = make_random(1024, 8, rng);
    auto recovered = ntt.idft_batch(ntt.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "Round-trip (n=1024, w=8) failed";
}

TEST(NttCuda, RoundTripAllZeros) {
    NttCuda<BabyBear> ntt;
    auto original  = make_col({0, 0, 0, 0, 0, 0, 0, 0});
    auto recovered = ntt.idft_batch(ntt.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original));
}

// ============================================================
// 4. Inverse NTT == CPU inverse NTT
// ============================================================

TEST(NttCuda, InverseMatchesCpu) {
    NttCuda<BabyBear>   gpu;
    Radix2Dit<BabyBear> cpu;
    std::mt19937 rng(222);
    for (size_t log_h : {2u, 4u, 8u, 10u}) {
        size_t h   = 1u << log_h;
        auto   mat = make_random(h, 2, rng);
        EXPECT_TRUE(matrices_equal(cpu.idft_batch(mat), gpu.idft_batch(mat)))
            << "Inverse NTT mismatch at n=" << h;
    }
}

// ============================================================
// 5. Coset DFT matches CPU coset_dft_batch
// ============================================================

TEST(NttCuda, CosetDftMatchesCpu) {
    NttCuda<BabyBear>   gpu;
    Radix2Dit<BabyBear> cpu;
    BabyBear shift(BabyBear::GENERATOR_VAL);
    std::mt19937 rng(333);

    for (size_t log_h : {3u, 6u, 10u}) {
        size_t h = 1u << log_h;
        for (size_t w : {1u, 4u}) {
            auto mat = make_random(h, w, rng);
            EXPECT_TRUE(matrices_equal(
                cpu.coset_dft_batch(mat, shift),
                gpu.coset_dft_batch(mat, shift)))
                << "coset_dft mismatch n=" << h << " w=" << w;
        }
    }
}

// ============================================================
// 6. Coset IDFT round-trip
// ============================================================

TEST(NttCuda, CosetIdftRoundTrip) {
    NttCuda<BabyBear> ntt;
    BabyBear shift(BabyBear::GENERATOR_VAL);
    auto original    = make_col({5, 3, 7, 1, 9, 2, 4, 8});
    auto coset_evals = ntt.coset_dft_batch(original, shift);
    auto recovered   = ntt.coset_idft_batch(coset_evals, shift);
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "coset DFT round-trip failed";
}

TEST(NttCuda, CosetIdftMatchesCpu) {
    NttCuda<BabyBear>   gpu;
    Radix2Dit<BabyBear> cpu;
    BabyBear shift(BabyBear::GENERATOR_VAL);
    std::mt19937 rng(444);
    auto mat = make_random(256, 4, rng);
    EXPECT_TRUE(matrices_equal(
        cpu.coset_idft_batch(mat, shift),
        gpu.coset_idft_batch(mat, shift)));
}

// ============================================================
// 7. coset_lde_batch matches CPU
// ============================================================

TEST(NttCuda, CosetLdeConstantPoly) {
    NttCuda<BabyBear> ntt;
    BabyBear shift(BabyBear::GENERATOR_VAL);
    // Evaluations of constant 5 on size-4 domain: all equal 5
    auto evals = make_col({5, 5, 5, 5});
    auto lde   = ntt.coset_lde_batch(evals, 1, shift);
    ASSERT_EQ(lde.height(), 8u);
    BabyBear five(5u);
    for (size_t r = 0; r < 8; ++r)
        EXPECT_EQ(lde.get_unchecked(r, 0), five)
            << "LDE of constant poly should be constant at row " << r;
}

TEST(NttCuda, CosetLdeMatchesCpu) {
    NttCuda<BabyBear>   gpu;
    Radix2Dit<BabyBear> cpu;
    BabyBear shift(BabyBear::GENERATOR_VAL);
    std::mt19937 rng(555);

    for (size_t log_h : {3u, 5u, 8u}) {
        size_t h = 1u << log_h;
        for (size_t added : {1u, 2u}) {
            for (size_t w : {1u, 4u}) {
                auto mat = make_random(h, w, rng);
                auto cpu_lde = cpu.coset_lde_batch(mat, added, shift);
                auto gpu_lde = gpu.coset_lde_batch(mat, added, shift);
                EXPECT_TRUE(matrices_equal(cpu_lde, gpu_lde))
                    << "coset_lde mismatch n=" << h
                    << " added=" << added << " w=" << w;
            }
        }
    }
}

// ============================================================
// 8. Varying batch sizes
// ============================================================

TEST(NttCuda, VaryingBatchSizes) {
    NttCuda<BabyBear>   gpu;
    Radix2Dit<BabyBear> cpu;
    std::mt19937 rng(777);
    size_t h = 64;

    for (size_t w : {1u, 2u, 3u, 5u, 8u, 16u, 32u}) {
        auto mat = make_random(h, w, rng);
        EXPECT_TRUE(matrices_equal(cpu.dft_batch(mat), gpu.dft_batch(mat)))
            << "dft_batch mismatch for width=" << w;
    }
}

// ============================================================
// 9. DFT linearity: dft(a+b) == dft(a) + dft(b)
// ============================================================

TEST(NttCuda, DftLinearity) {
    NttCuda<BabyBear> ntt;
    auto a = make_col({1, 3, 5, 7});
    auto b = make_col({2, 4, 6, 8});

    auto dft_a = ntt.dft_batch(a);
    auto dft_b = ntt.dft_batch(b);

    RowMajorMatrix<BabyBear> sum_of_dfts(4, 1);
    for (size_t r = 0; r < 4; ++r)
        sum_of_dfts.set_unchecked(r, 0,
            dft_a.get_unchecked(r, 0) + dft_b.get_unchecked(r, 0));

    RowMajorMatrix<BabyBear> a_plus_b(4, 1);
    for (size_t r = 0; r < 4; ++r)
        a_plus_b.set_unchecked(r, 0,
            a.get_unchecked(r, 0) + b.get_unchecked(r, 0));

    auto dft_sum = ntt.dft_batch(a_plus_b);
    EXPECT_TRUE(matrices_equal(dft_sum, sum_of_dfts))
        << "DFT linearity violated";
}

// ============================================================
// 10. Larger sizes (stress)
// ============================================================

TEST(NttCuda, RoundTripSize1024) {
    NttCuda<BabyBear> ntt;
    std::mt19937 rng(123);
    auto original  = make_random(1024, 1, rng);
    auto recovered = ntt.idft_batch(ntt.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "Round-trip (n=1024) failed";
}

TEST(NttCuda, ForwardMatchesCpuSize512Width16) {
    NttCuda<BabyBear>   gpu;
    Radix2Dit<BabyBear> cpu;
    std::mt19937 rng(456);
    auto mat = make_random(512, 16, rng);
    EXPECT_TRUE(matrices_equal(cpu.dft_batch(mat), gpu.dft_batch(mat)))
        << "GPU/CPU dft_batch mismatch (n=512, w=16)";
}

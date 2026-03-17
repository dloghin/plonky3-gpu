/**
 * @file test_dft.cpp
 * @brief Google Tests for Radix2Dit NTT/DFT over BabyBear and BabyBear4.
 *
 * Acceptance criteria checked:
 *  1. dft_batch on single-column matrix produces known output for simple polys.
 *  2. idft_batch(dft_batch(mat)) == mat  (round-trip identity).
 *  3. coset_lde_batch produces correct LDE evaluations.
 *  4. coset_lde_batch with added_bits=1 and shift=GENERATOR is self-consistent.
 *  5. Extension field DFT/IDFT round-trip works.
 *  6. Twiddle caching: second call with same size reuses cached values.
 */

#include <gtest/gtest.h>

#include "radix2_dit.hpp"
#include "baby_bear.hpp"
#include "extension_field.hpp"
#include "dense_matrix.hpp"

#include <vector>
#include <cstdint>

using namespace p3_field;
using namespace p3_matrix;
using namespace p3_dft;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/// Check two BabyBear matrices are equal.
static bool matrices_equal(const RowMajorMatrix<BabyBear>& a,
                            const RowMajorMatrix<BabyBear>& b) {
    if (a.height() != b.height() || a.width() != b.width()) return false;
    for (size_t r = 0; r < a.height(); ++r) {
        for (size_t c = 0; c < a.width(); ++c) {
            if (a.get_unchecked(r, c) != b.get_unchecked(r, c)) return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 1: Single-element (height=1) DFT is identity.
// ---------------------------------------------------------------------------

TEST(Radix2Dit, DftBatchSingleRow) {
    Radix2Dit<BabyBear> dft;
    auto mat = make_col({42});
    auto result = dft.dft_batch(mat);
    EXPECT_EQ(result.height(), 1u);
    EXPECT_EQ(result.width(), 1u);
    EXPECT_EQ(result.get_unchecked(0, 0), BabyBear(42u));
}

// ---------------------------------------------------------------------------
// Test 2: Round-trip identity  idft(dft(mat)) == mat
// ---------------------------------------------------------------------------

TEST(Radix2Dit, RoundTripSize4) {
    Radix2Dit<BabyBear> dft;
    auto original = make_col({1, 2, 3, 4});
    auto transformed = dft.dft_batch(original);
    auto recovered = dft.idft_batch(transformed);
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "Round-trip (size 4, 1 col) failed";
}

TEST(Radix2Dit, RoundTripSize8) {
    Radix2Dit<BabyBear> dft;
    auto original = make_col({0, 1, 2, 3, 4, 5, 6, 7});
    auto recovered = dft.idft_batch(dft.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "Round-trip (size 8, 1 col) failed";
}

TEST(Radix2Dit, RoundTripSize16MultiCol) {
    Radix2Dit<BabyBear> dft;
    // 16 rows x 4 columns of distinct values
    std::vector<uint32_t> vals;
    for (uint32_t i = 0; i < 64; ++i) vals.push_back(i + 1);
    auto original = make_matrix(vals, 4);
    auto recovered = dft.idft_batch(dft.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "Round-trip (size 16, 4 col) failed";
}

TEST(Radix2Dit, RoundTripAllZeros) {
    Radix2Dit<BabyBear> dft;
    auto original = make_col({0, 0, 0, 0, 0, 0, 0, 0});
    auto recovered = dft.idft_batch(dft.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original));
}

TEST(Radix2Dit, RoundTripDegreeOne) {
    // Polynomial f(x) = 1 + x; coefficients [1, 1, 0, 0, 0, 0, 0, 0]
    Radix2Dit<BabyBear> dft;
    auto original = make_col({1, 1, 0, 0, 0, 0, 0, 0});
    auto recovered = dft.idft_batch(dft.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original));
}

// ---------------------------------------------------------------------------
// Test 3: DFT of constant polynomial
//
// f(x) = c  =>  coefficients = [c, 0, 0, ..., 0]
// After DFT:    all evaluations equal c
// ---------------------------------------------------------------------------

TEST(Radix2Dit, DftConstantPolynomial) {
    Radix2Dit<BabyBear> dft;
    // Constant polynomial c=7: coefficients [7, 0, 0, 0, 0, 0, 0, 0]
    auto coeff_mat = make_col({7, 0, 0, 0, 0, 0, 0, 0});
    auto eval_mat = dft.dft_batch(coeff_mat);
    ASSERT_EQ(eval_mat.height(), 8u);
    BabyBear seven(7u);
    for (size_t r = 0; r < 8; ++r) {
        EXPECT_EQ(eval_mat.get_unchecked(r, 0), seven)
            << "Constant poly eval mismatch at row " << r;
    }
}

// ---------------------------------------------------------------------------
// Test 4: DFT of "all ones" polynomial
//
// f(x) = 1 + x + x^2 + ... + x^(n-1)
// Evaluations: f(omega^k) = sum_{j=0}^{n-1} omega^{jk}
//   = n  if k == 0
//   = 0  if k != 0  (geometric sum of roots of unity)
// ---------------------------------------------------------------------------

TEST(Radix2Dit, DftAllOnesPolynomial) {
    Radix2Dit<BabyBear> dft;
    auto coeff_mat = make_col({1, 1, 1, 1, 1, 1, 1, 1});
    auto eval_mat = dft.dft_batch(coeff_mat);
    ASSERT_EQ(eval_mat.height(), 8u);
    BabyBear eight(8u);
    BabyBear zero = BabyBear::zero_val();
    EXPECT_EQ(eval_mat.get_unchecked(0, 0), eight) << "f(1) should be n=8";
    for (size_t r = 1; r < 8; ++r) {
        EXPECT_EQ(eval_mat.get_unchecked(r, 0), zero)
            << "f(omega^" << r << ") should be 0";
    }
}

// ---------------------------------------------------------------------------
// Test 5: Twiddle caching
//
// Running dft_batch twice for the same size should produce identical results.
// We can't directly observe cache hits, but we verify determinism.
// ---------------------------------------------------------------------------

TEST(Radix2Dit, TwiddleCachingDeterminism) {
    Radix2Dit<BabyBear> dft;
    auto mat = make_col({3, 1, 4, 1, 5, 9, 2, 6});
    auto result1 = dft.dft_batch(mat);
    auto result2 = dft.dft_batch(mat);  // should reuse cached twiddles
    EXPECT_TRUE(matrices_equal(result1, result2))
        << "Second DFT call with cached twiddles gave different result";
}

TEST(Radix2Dit, TwiddleCachingDifferentSize) {
    // Cache entries for different sizes should not interfere
    Radix2Dit<BabyBear> dft;
    auto mat4 = make_col({1, 2, 3, 4});
    auto mat8 = make_col({1, 2, 3, 4, 5, 6, 7, 8});
    auto r4a = dft.dft_batch(mat4);
    auto r8  = dft.dft_batch(mat8);
    auto r4b = dft.dft_batch(mat4);  // reuses cache for size 4
    EXPECT_TRUE(matrices_equal(r4a, r4b))
        << "Size-4 DFT changed after size-8 DFT was computed";
    (void)r8;
}

// ---------------------------------------------------------------------------
// Test 6: Coset DFT / IDFT round-trip
// ---------------------------------------------------------------------------

TEST(Radix2Dit, CosetDftIdftRoundTrip) {
    Radix2Dit<BabyBear> dft;
    BabyBear shift(BabyBear::GENERATOR_VAL);
    auto original = make_col({5, 3, 7, 1, 9, 2, 4, 8});
    auto coset_evals = dft.coset_dft_batch(original, shift);
    auto recovered   = dft.coset_idft_batch(coset_evals, shift);
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "coset DFT round-trip failed";
}

TEST(Radix2Dit, CosetDftIdftRoundTripMultiCol) {
    Radix2Dit<BabyBear> dft;
    BabyBear shift(7u);
    std::vector<uint32_t> vals;
    for (uint32_t i = 1; i <= 32; ++i) vals.push_back(i);
    auto original = make_matrix(vals, 4);
    auto coset_evals = dft.coset_dft_batch(original, shift);
    auto recovered   = dft.coset_idft_batch(coset_evals, shift);
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "coset DFT multi-col round-trip failed";
}

// ---------------------------------------------------------------------------
// Test 7: coset_lde_batch  -- LDE consistency check
//
// The LDE of f from H to H' (H' = shift * coset of size n*2^added_bits)
// must satisfy:
//   lde[0] == f evaluated at shift*omega^0 = shift*1 = shift
//
// We verify this for the constant polynomial f(x) = c (coefficients [c,0,...]).
// LDE of constant c: all evaluations = c.
// ---------------------------------------------------------------------------

TEST(Radix2Dit, CosetLdeBatchConstantPoly) {
    Radix2Dit<BabyBear> dft;
    BabyBear shift(BabyBear::GENERATOR_VAL);
    size_t added_bits = 1;
    // Evaluations of constant polynomial c=5 on H (size 4): all equal 5
    auto evals = make_col({5, 5, 5, 5});
    auto lde = dft.coset_lde_batch(evals, added_bits, shift);
    ASSERT_EQ(lde.height(), 8u);
    BabyBear five(5u);
    for (size_t r = 0; r < 8; ++r) {
        EXPECT_EQ(lde.get_unchecked(r, 0), five)
            << "LDE of constant poly should be constant at row " << r;
    }
}

TEST(Radix2Dit, CosetLdeBatchRoundTrip) {
    // The LDE of f should agree with the original evaluations when re-evaluated
    // on the original subgroup.  We verify this indirectly: DFT the LDE,
    // do inverse DFT on the smaller domain (the first n evals), recover original.
    // Instead we use the simpler check: idft(lde)[0..n-1] == idft(evals).
    Radix2Dit<BabyBear> dft;
    BabyBear shift(BabyBear::GENERATOR_VAL);
    auto original = make_col({1, 2, 3, 4});
    auto lde = dft.coset_lde_batch(original, 1, shift);
    ASSERT_EQ(lde.height(), 8u);

    // The LDE on the full coset should have the same polynomial coefficients
    // as the original (with zero-padded high-degree coefficients).
    auto coeffs = dft.coset_idft_batch(lde, shift);
    // First 4 entries = original polynomial coefficients; entries 4-7 = 0
    auto orig_coeffs = dft.idft_batch(original);
    for (size_t r = 0; r < 4; ++r) {
        EXPECT_EQ(coeffs.get_unchecked(r, 0), orig_coeffs.get_unchecked(r, 0))
            << "LDE coefficient mismatch at row " << r;
    }
    BabyBear zero = BabyBear::zero_val();
    for (size_t r = 4; r < 8; ++r) {
        EXPECT_EQ(coeffs.get_unchecked(r, 0), zero)
            << "LDE high-degree coefficient should be 0 at row " << r;
    }
}

TEST(Radix2Dit, CosetLdeBatchGeneratorShift) {
    // Criterion 4: coset_lde_batch with added_bits=1 and shift=GENERATOR
    Radix2Dit<BabyBear> dft;
    BabyBear gen(BabyBear::GENERATOR_VAL);
    std::vector<uint32_t> vals = {3, 7, 1, 5, 9, 2, 6, 4};
    auto evals = make_col(vals);
    auto lde = dft.coset_lde_batch(evals, 1, gen);
    ASSERT_EQ(lde.height(), 16u);
    ASSERT_EQ(lde.width(),  1u);
    // Verify the LDE is self-consistent: apply coset_lde_batch again for
    // added_bits=0 (no-op extension) and verify it matches input evals.
    // (We can't get back to evals from lde easily without a full oracle,
    //  so we verify the polynomial interpolates correctly via round-trip.)
    auto recovered_coeffs = dft.coset_idft_batch(lde, gen);
    ASSERT_EQ(recovered_coeffs.height(), 16u);
    // High-degree coefficients should all be zero (we extended from degree<8 poly)
    BabyBear zero = BabyBear::zero_val();
    for (size_t r = 8; r < 16; ++r) {
        EXPECT_EQ(recovered_coeffs.get_unchecked(r, 0), zero)
            << "High-degree coeff should be 0 at row " << r;
    }
}

// ---------------------------------------------------------------------------
// Test 8: Extension field DFT/IDFT round-trip
// ---------------------------------------------------------------------------

TEST(Radix2Dit, ExtDftIdftRoundTrip) {
    Radix2Dit<BabyBear> dft;
    size_t h = 8;
    size_t w = 2;
    RowMajorMatrix<BabyBear4> mat(h, w);
    // Fill with a simple pattern
    uint32_t v = 1;
    for (size_t r = 0; r < h; ++r) {
        for (size_t c = 0; c < w; ++c) {
            BabyBear4 elem;
            for (size_t d = 0; d < 4; ++d) {
                elem[d] = BabyBear(v++);
            }
            mat.set_unchecked(r, c, elem);
        }
    }
    auto original = mat;

    auto transformed = dft.dft_algebra_batch(mat);
    auto recovered   = dft.idft_algebra_batch(transformed);

    ASSERT_EQ(recovered.height(), original.height());
    ASSERT_EQ(recovered.width(),  original.width());
    for (size_t r = 0; r < h; ++r) {
        for (size_t c = 0; c < w; ++c) {
            EXPECT_EQ(recovered.get_unchecked(r, c), original.get_unchecked(r, c))
                << "Ext field round-trip mismatch at (" << r << "," << c << ")";
        }
    }
}

TEST(Radix2Dit, IdftAlgebraVec) {
    Radix2Dit<BabyBear> dft;
    // Build a vector of extension field evaluations
    std::vector<BabyBear4> vec;
    for (size_t i = 0; i < 8; ++i) {
        BabyBear4 elem;
        elem[0] = BabyBear(static_cast<uint32_t>(i + 1));
        elem[1] = BabyBear(static_cast<uint32_t>(i * 2 + 3));
        elem[2] = BabyBear(static_cast<uint32_t>(i * 3 + 5));
        elem[3] = BabyBear(static_cast<uint32_t>(i * 4 + 7));
        vec.push_back(elem);
    }

    // DFT then IDFT via the single-vector API
    RowMajorMatrix<BabyBear4> mat(8, 1);
    for (size_t i = 0; i < 8; ++i) mat.set_unchecked(i, 0, vec[i]);
    auto dft_mat = dft.dft_algebra_batch(mat);
    std::vector<BabyBear4> dft_vec(8);
    for (size_t i = 0; i < 8; ++i) dft_vec[i] = dft_mat.get_unchecked(i, 0);

    auto recovered = dft.idft_algebra(dft_vec);

    ASSERT_EQ(recovered.size(), vec.size());
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(recovered[i], vec[i])
            << "idft_algebra round-trip mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 9: DFT linearity   dft(a + b) == dft(a) + dft(b)
// ---------------------------------------------------------------------------

TEST(Radix2Dit, DftLinearity) {
    Radix2Dit<BabyBear> dft;
    auto a = make_col({1, 3, 5, 7});
    auto b = make_col({2, 4, 6, 8});

    // Compute dft(a) + dft(b)
    auto dft_a = dft.dft_batch(a);
    auto dft_b = dft.dft_batch(b);
    RowMajorMatrix<BabyBear> sum_of_dfts(4, 1);
    for (size_t r = 0; r < 4; ++r) {
        sum_of_dfts.set_unchecked(r, 0,
            dft_a.get_unchecked(r, 0) + dft_b.get_unchecked(r, 0));
    }

    // Compute dft(a + b)
    RowMajorMatrix<BabyBear> a_plus_b(4, 1);
    for (size_t r = 0; r < 4; ++r) {
        a_plus_b.set_unchecked(r, 0,
            a.get_unchecked(r, 0) + b.get_unchecked(r, 0));
    }
    auto dft_sum = dft.dft_batch(a_plus_b);

    EXPECT_TRUE(matrices_equal(dft_sum, sum_of_dfts))
        << "DFT linearity violated";
}

// ---------------------------------------------------------------------------
// Test 10: Larger round-trip (size 64)
// ---------------------------------------------------------------------------

TEST(Radix2Dit, RoundTripSize64) {
    Radix2Dit<BabyBear> dft;
    std::vector<uint32_t> vals(64);
    for (uint32_t i = 0; i < 64; ++i) vals[i] = (i * 1234567 + 89) % BabyBear::PRIME;
    auto original = make_col(vals);
    auto recovered = dft.idft_batch(dft.dft_batch(original));
    EXPECT_TRUE(matrices_equal(recovered, original))
        << "Round-trip (size 64) failed";
}

// ---------------------------------------------------------------------------
// Test 11: DFT of standard basis vector e_k  ->  evaluations of x^k
//
// The DFT of e_k (1 at position k, 0 elsewhere) gives the evaluations of
// the monomial x^k at {omega^0, omega^1, ..., omega^{n-1}}.
// So eval_mat[j] = omega^{j*k}.
// ---------------------------------------------------------------------------

TEST(Radix2Dit, DftBasisVector) {
    Radix2Dit<BabyBear> dft;
    size_t n = 8;
    size_t k = 2;
    std::vector<uint32_t> basis(n, 0);
    basis[k] = 1;
    auto e_k = make_col(basis);
    auto eval_mat = dft.dft_batch(e_k);

    // Compute expected: omega^{j*k} for j = 0..n-1
    BabyBear omega = BabyBear::two_adic_generator(3);  // primitive 8th root
    for (size_t j = 0; j < n; ++j) {
        BabyBear expected = omega.exp_u64(static_cast<uint64_t>(j * k));
        EXPECT_EQ(eval_mat.get_unchecked(j, 0), expected)
            << "DFT basis vector e_" << k << " mismatch at j=" << j;
    }
}

// ---------------------------------------------------------------------------
// Test 12: Verify DFT evaluates poly at correct points
//
// For a known polynomial f(x) = a0 + a1*x + ... + a_{n-1}*x^{n-1},
// DFT coefficients [a0,...,a_{n-1}] should give evaluations at omega^j.
// We check a simple case: f(x) = 1 + 2x, n=4.
// ---------------------------------------------------------------------------

TEST(Radix2Dit, DftEvaluatesAtOmegaPowers) {
    Radix2Dit<BabyBear> dft;
    // f(x) = 1 + 2x; coefficients: [1, 2, 0, 0]
    auto coeff_mat = make_col({1, 2, 0, 0});
    auto eval_mat = dft.dft_batch(coeff_mat);

    BabyBear omega = BabyBear::two_adic_generator(2);  // primitive 4th root
    BabyBear one(1u);
    BabyBear two(2u);
    for (size_t j = 0; j < 4; ++j) {
        BabyBear omega_j = omega.exp_u64(j);
        BabyBear expected = one + two * omega_j;  // f(omega^j)
        EXPECT_EQ(eval_mat.get_unchecked(j, 0), expected)
            << "f(omega^" << j << ") mismatch";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

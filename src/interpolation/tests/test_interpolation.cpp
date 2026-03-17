/**
 * @file test_interpolation.cpp
 * @brief Google Test suite for p3_interpolation.
 *
 * Validates batch_multiplicative_inverse, compute_diff_invs, and
 * interpolate_coset_with_precomputation against known mathematical
 * properties and concrete reference values derived from the formulae
 * in plonky3/interpolation/src/lib.rs.
 *
 * Field types used:
 *   F  = BabyBear         (base field, p = 2013265921)
 *   EF = BabyBear4        (degree-4 extension, alpha^4 = 11)
 *
 * Mathematical properties tested:
 *   1. batch_multiplicative_inverse: values[i] * inv[i] == 1 for all i.
 *   2. compute_diff_invs: hardcoded values for n=2 coset {1, -1}, shift=1.
 *   3. Constant polynomial f(x) = c returns c at any point.
 *   4. Linear polynomial f(x) = x on n=2 coset, base-field point z=5 → 5.
 *   5. Linear polynomial f(x) = x on n=2 coset, extension-field point z
 *      returns z (tests EF path).
 *   6. Cubic polynomial f(x) = x^3 on n=4 coset (H=4th roots, shift=1):
 *      - z=5 (base field) → 125
 *      - z=[5,1,0,0] (extension field) → [125,75,15,1]
 *   7. Precomputation can be reused for multiple query points.
 */

#include <gtest/gtest.h>
#include "interpolation.hpp"
#include "baby_bear.hpp"
#include "extension_field.hpp"

#include <vector>
#include <array>

using BB  = p3_field::BabyBear;
using EF  = p3_field::BabyBear4;    // BinomialExtensionField<BabyBear, 4, 11>

using namespace p3_interpolation;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr uint32_t P = BB::PRIME;  // 2013265921

static BB bb(uint32_t v) { return BB(v); }

static EF ef(uint32_t a0, uint32_t a1 = 0, uint32_t a2 = 0, uint32_t a3 = 0) {
    return EF(std::array<BB,4>{BB(a0), BB(a1), BB(a2), BB(a3)});
}

static void expect_ef_eq(const EF& actual,
                         uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3) {
    EXPECT_EQ(actual.coeffs[0].value(), e0) << "coeff[0] mismatch";
    EXPECT_EQ(actual.coeffs[1].value(), e1) << "coeff[1] mismatch";
    EXPECT_EQ(actual.coeffs[2].value(), e2) << "coeff[2] mismatch";
    EXPECT_EQ(actual.coeffs[3].value(), e3) << "coeff[3] mismatch";
}

// ---------------------------------------------------------------------------
// 1. batch_multiplicative_inverse -- BabyBear
// ---------------------------------------------------------------------------

TEST(BatchInverse, SingleElement) {
    std::vector<BB> v = {bb(5u)};
    auto inv = batch_multiplicative_inverse(v);
    ASSERT_EQ(inv.size(), 1u);
    EXPECT_EQ((v[0] * inv[0]).value(), 1u);
}

TEST(BatchInverse, TwoElements) {
    std::vector<BB> v = {bb(2u), bb(3u)};
    auto inv = batch_multiplicative_inverse(v);
    ASSERT_EQ(inv.size(), 2u);
    EXPECT_EQ((v[0] * inv[0]).value(), 1u);
    EXPECT_EQ((v[1] * inv[1]).value(), 1u);
}

TEST(BatchInverse, MultipleElements) {
    // Non-trivial values: 2, 3, 5, 7, 11
    std::vector<BB> v = {bb(2u), bb(3u), bb(5u), bb(7u), bb(11u)};
    auto inv = batch_multiplicative_inverse(v);
    ASSERT_EQ(inv.size(), 5u);
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ((v[i] * inv[i]).value(), 1u) << "failed at index " << i;
    }
}

TEST(BatchInverse, LargerValues) {
    // Include P-1 (= -1) and verify inv(-1) = -1
    std::vector<BB> v = {bb(P - 1u), bb(1000000000u), bb(42u), bb(P - 42u)};
    auto inv = batch_multiplicative_inverse(v);
    ASSERT_EQ(inv.size(), 4u);
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ((v[i] * inv[i]).value(), 1u) << "failed at index " << i;
    }
}

TEST(BatchInverse, EmptyInput) {
    std::vector<BB> v;
    auto inv = batch_multiplicative_inverse(v);
    EXPECT_TRUE(inv.empty());
}

// batch_multiplicative_inverse on extension-field elements
TEST(BatchInverse, ExtensionField) {
    std::vector<EF> v = {
        ef(1u, 2u, 3u, 4u),
        ef(5u, 6u, 7u, 8u),
        ef(100u, 0u, 0u, 0u),
    };
    auto inv = batch_multiplicative_inverse(v);
    ASSERT_EQ(inv.size(), 3u);
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(v[i] * inv[i], EF::one_val()) << "failed at index " << i;
    }
}

// ---------------------------------------------------------------------------
// 2. compute_diff_invs -- known reference values
// ---------------------------------------------------------------------------

// H = {1, p-1}, shift = 1.
// diff_invs[i] = h_i / (n * g^{n-1}) = h_i / (2 * 1) = h_i / 2.
//   diff_invs[0] = 1/2 = (P+1)/2 = 1006632961
//   diff_invs[1] = (P-1)/2      = 1006632960
TEST(ComputeDiffInvs, N2Shift1) {
    std::vector<BB> H = {bb(1u), bb(P - 1u)};
    BB shift(1u);

    auto d = compute_diff_invs(H, shift);

    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(d[0].value(), (P + 1u) / 2u) << "diff_invs[0] should be 1/2";
    EXPECT_EQ(d[1].value(), (P - 1u) / 2u) << "diff_invs[1] should be -1/2";
}

// H = {1}, shift = 7.  Trivial coset: diff_invs[0] = h_0 / (1 * 7^0) = 1/1 = 1.
TEST(ComputeDiffInvs, N1AnyShift) {
    std::vector<BB> H = {bb(1u)};
    BB shift(7u);

    auto d = compute_diff_invs(H, shift);

    ASSERT_EQ(d.size(), 1u);
    EXPECT_EQ(d[0].value(), 1u) << "n=1: diff_invs[0] should be 1";
}

// H = {1, p-1}, shift = 2.
// diff_invs[i] = h_i / (2 * 2^1) = h_i / 4.
//   diff_invs[0] = 1/4 = 4^{-1} mod p.
//   diff_invs[1] = (p-1)/4 = -(1/4) mod p.
// 4^{-1} mod p: 4 * 1509949441 = 6039797764 = 3*2013265921 + 1. => 1509949441.
TEST(ComputeDiffInvs, N2Shift2) {
    std::vector<BB> H = {bb(1u), bb(P - 1u)};
    BB shift(2u);

    auto d = compute_diff_invs(H, shift);

    ASSERT_EQ(d.size(), 2u);

    // Verify d[i] * prod_{j!=i}(shift*H[i] - shift*H[j]) == 1
    // prod_{j!=0}(2*1 - 2*(P-1)) = 2 - (2P-2) ≡ 4 mod p
    BB prod0 = bb(2u) * bb(1u) - bb(2u) * bb(P - 1u);    // = 4
    EXPECT_EQ((d[0] * prod0).value(), 1u);

    // prod_{j!=1}(2*(P-1) - 2*1) = (2P-2) - 2 ≡ -4 mod p
    BB prod1 = bb(2u) * bb(P - 1u) - bb(2u) * bb(1u);   // = -4
    EXPECT_EQ((d[1] * prod1).value(), 1u);
}

// ---------------------------------------------------------------------------
// 3. Constant polynomial f(x) = c
// ---------------------------------------------------------------------------

TEST(Interpolation, ConstantPolynomialN1) {
    // H = {1}, shift = 3, evals = [7]  =>  f(z) = 7 for any z.
    std::vector<BB> H = {bb(1u)};
    BB shift(3u);
    auto diff_invs = compute_diff_invs(H, shift);

    std::vector<EF> evals = {ef(7u)};
    EF z = ef(5u);  // arbitrary, not on coset {3}

    EF result = interpolate_coset_with_precomputation(evals, shift, z, H, diff_invs);
    expect_ef_eq(result, 7u, 0u, 0u, 0u);
}

TEST(Interpolation, ConstantPolynomialN2) {
    // H = {1, p-1}, shift = 1, evals = [42, 42]  =>  f(z) = 42.
    std::vector<BB> H = {bb(1u), bb(P - 1u)};
    BB shift(1u);
    auto diff_invs = compute_diff_invs(H, shift);

    std::vector<EF> evals = {ef(42u), ef(42u)};
    EF z = ef(100u);  // not in coset {1, p-1}

    EF result = interpolate_coset_with_precomputation(evals, shift, z, H, diff_invs);
    expect_ef_eq(result, 42u, 0u, 0u, 0u);
}

// ---------------------------------------------------------------------------
// 4. Linear polynomial f(x) = x,  base-field evaluation point
// ---------------------------------------------------------------------------
// H = {1, p-1}, shift = 1, coset = {1, p-1}.
// evals[i] = h_i  (f(h_i) = h_i for f=identity).
// For z = 5 (embedded as EF), expect result = [5,0,0,0].
//
// Manual verification (complex analogy, w = primitive 4th root = i):
//   vanishing = 5^2 - 1^2 = 24
//   sum = (1/2)*1/(5-1) + (-1/2)*(p-1)/(5-(p-1))
//       = 1/8 + (-1/2)*(-1)/6  = 1/8 + 1/12 = 3/24 + 2/24 = 5/24
//   result = 24 * 5/24 = 5  ✓

TEST(Interpolation, LinearPolynomialBaseFieldPoint) {
    std::vector<BB> H = {bb(1u), bb(P - 1u)};
    BB shift(1u);
    auto diff_invs = compute_diff_invs(H, shift);

    std::vector<EF> evals = {ef(1u), ef(P - 1u)};
    EF z = ef(5u);

    EF result = interpolate_coset_with_precomputation(evals, shift, z, H, diff_invs);
    EXPECT_EQ(result, z) << "f(x)=x should give back z=5";
}

// ---------------------------------------------------------------------------
// 5. Linear polynomial f(x) = x,  extension-field evaluation point
// ---------------------------------------------------------------------------
// Same coset as above but z = [7, 3, 0, 0] in EF.  Expect f(z) = z.

TEST(Interpolation, LinearPolynomialExtFieldPoint) {
    std::vector<BB> H = {bb(1u), bb(P - 1u)};
    BB shift(1u);
    auto diff_invs = compute_diff_invs(H, shift);

    std::vector<EF> evals = {ef(1u), ef(P - 1u)};
    EF z = ef(7u, 3u, 0u, 0u);

    EF result = interpolate_coset_with_precomputation(evals, shift, z, H, diff_invs);
    EXPECT_EQ(result, z) << "f(x)=x should give back z=[7,3,0,0]";
}

// Linear with non-trivial shift: H = {1, p-1}, shift = 2, coset = {2, p-2}.
// evals = [f(2), f(p-2)] = [2, p-2].  z = 5.  Expected 5.
TEST(Interpolation, LinearPolynomialNonTrivialShift) {
    std::vector<BB> H = {bb(1u), bb(P - 1u)};
    BB shift(2u);
    auto diff_invs = compute_diff_invs(H, shift);

    // evals of f(x) = x on coset {2, p-2}
    std::vector<EF> evals = {ef(2u), ef(P - 2u)};
    EF z = ef(5u);

    EF result = interpolate_coset_with_precomputation(evals, shift, z, H, diff_invs);
    EXPECT_EQ(result, z) << "f(x)=x with shift=2 should give back z=5";
}

// ---------------------------------------------------------------------------
// 6. Cubic polynomial f(x) = x^3 on 4-element coset
// ---------------------------------------------------------------------------
// H = {1, w, w^2, w^3} (4th roots of unity), shift = 1.
// evals[i] = h_i^3.
// diff_invs[i] = h_i / (4 * 1^3) = h_i / 4.
//
// Verification for z=5 (base field):
//   As shown in the mathematical background, the sum telescopes to 125/624
//   and vanishing = 5^4 - 1 = 624, so result = 125 = 5^3.
//
// Verification for z=[5,1,0,0] (extension field):
//   z^3 = (5+α)^3 where α^4 = 11.
//   z^2 = [25,10,1,0] (computed by schoolbook multiplication).
//   z^3 = [125,75,15,1].

TEST(Interpolation, CubicPolynomialBaseFieldPoint) {
    // Build H = {1, w, w^2, w^3}  (primitive 4th roots of unity)
    BB w = BB::two_adic_generator(2);  // primitive 4th root of unity
    std::vector<BB> H = {
        bb(1u),
        w,
        w.exp_u64(2),
        w.exp_u64(3)
    };
    BB shift(1u);
    auto diff_invs = compute_diff_invs(H, shift);

    // evals[i] = h_i^3 = h_i.exp_u64(3)
    std::vector<EF> evals;
    for (const auto& h : H) {
        evals.push_back(ef(h.exp_u64(3).value()));
    }

    // z = 5, expected f(5) = 125
    EF z = ef(5u);
    EF result = interpolate_coset_with_precomputation(evals, shift, z, H, diff_invs);
    expect_ef_eq(result, 125u, 0u, 0u, 0u);
}

TEST(Interpolation, CubicPolynomialExtFieldPoint) {
    // Same coset as above
    BB w = BB::two_adic_generator(2);
    std::vector<BB> H = {
        bb(1u),
        w,
        w.exp_u64(2),
        w.exp_u64(3)
    };
    BB shift(1u);
    auto diff_invs = compute_diff_invs(H, shift);

    std::vector<EF> evals;
    for (const auto& h : H) {
        evals.push_back(ef(h.exp_u64(3).value()));
    }

    // z = [5,1,0,0] = 5 + alpha
    // z^2 = [25,10,1,0]
    // z^3 = [125,75,15,1]
    EF z = ef(5u, 1u, 0u, 0u);
    EF expected = z.exp_u64(3);  // computed via field arithmetic

    // Verify our expected value matches the schoolbook calculation
    ASSERT_EQ(expected, ef(125u, 75u, 15u, 1u))
        << "test setup: (5+alpha)^3 should be [125,75,15,1]";

    EF result = interpolate_coset_with_precomputation(evals, shift, z, H, diff_invs);
    EXPECT_EQ(result, expected);
}

// ---------------------------------------------------------------------------
// 7. Precomputation reuse: same diff_invs for multiple query points
// ---------------------------------------------------------------------------

TEST(Interpolation, PrecomputationReuse) {
    // H = {1, p-1}, shift = 1, f(x) = x
    std::vector<BB> H = {bb(1u), bb(P - 1u)};
    BB shift(1u);
    auto diff_invs = compute_diff_invs(H, shift);   // precomputed once

    std::vector<EF> evals = {ef(1u), ef(P - 1u)};

    // Query at several points; each should return z (since f=identity)
    std::vector<EF> query_points = {
        ef(2u),
        ef(100u),
        ef(1000000u),
        ef(7u, 3u, 0u, 0u),
        ef(0u, 1u, 0u, 0u),
    };

    for (const auto& z : query_points) {
        EF result = interpolate_coset_with_precomputation(
            evals, shift, z, H, diff_invs);
        EXPECT_EQ(result, z) << "f(x)=x failed for query point";
    }
}

// ---------------------------------------------------------------------------
// 8. Degree-7 polynomial on 8-element coset  (larger subgroup)
// ---------------------------------------------------------------------------

TEST(Interpolation, Degree7PolynomialBaseFieldPoint) {
    // H = {w^0, ..., w^7} where w = primitive 8th root of unity
    // f(x) = x^7,  shift = 1
    BB w = BB::two_adic_generator(3);   // primitive 8th root of unity
    const size_t n = 8;
    std::vector<BB> H(n);
    H[0] = bb(1u);
    for (size_t i = 1; i < n; ++i) H[i] = H[i-1] * w;

    BB shift(1u);
    auto diff_invs = compute_diff_invs(H, shift);

    std::vector<EF> evals;
    for (const auto& h : H) {
        evals.push_back(ef(h.exp_u64(7).value()));
    }

    // z = 2, expected f(2) = 2^7 = 128
    EF z = ef(2u);
    EF result = interpolate_coset_with_precomputation(evals, shift, z, H, diff_invs);
    expect_ef_eq(result, 128u, 0u, 0u, 0u);
}

// ---------------------------------------------------------------------------
// 9. f(x) = x evaluated on large subgroup (sanity: should recover z)
// ---------------------------------------------------------------------------

TEST(Interpolation, LinearOnLargeSubgroup) {
    // H = 8-element subgroup, shift = 3, f(x) = x
    BB w = BB::two_adic_generator(3);
    const size_t n = 8;
    std::vector<BB> H(n);
    H[0] = bb(1u);
    for (size_t i = 1; i < n; ++i) H[i] = H[i-1] * w;

    BB shift(3u);
    // coset = {3*h_i}; evals of f(x)=x on coset
    std::vector<EF> evals;
    for (const auto& h : H) {
        evals.push_back(ef((shift * h).value()));
    }

    auto diff_invs = compute_diff_invs(H, shift);

    EF z = ef(7u, 2u, 0u, 0u);
    EF result = interpolate_coset_with_precomputation(evals, shift, z, H, diff_invs);
    EXPECT_EQ(result, z) << "f(x)=x with n=8, shift=3 should return z";
}

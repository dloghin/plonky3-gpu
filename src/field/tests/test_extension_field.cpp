/**
 * @file test_extension_field.cpp
 * @brief Google Test suite for BinomialExtensionField<BabyBear, 4>
 *
 * Validates arithmetic against known Rust (Plonky3) reference outputs.
 * Tests cover: +, -, *, /, inverse, halve, exp, Frobenius, constants.
 */

#include <gtest/gtest.h>
#include "extension_field.hpp"
#include "baby_bear.hpp"

using namespace p3_field;
using Ext4 = BabyBear4; // BinomialExtensionField<BabyBear, 4, 11>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Ext4 make(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    std::array<BabyBear, 4> coeffs{
        BabyBear(a0), BabyBear(a1), BabyBear(a2), BabyBear(a3)
    };
    return Ext4(coeffs);
}

static void expect_eq(const Ext4& a, uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3) {
    EXPECT_EQ(a.coeffs[0].value(), e0) << "coefficient 0 mismatch";
    EXPECT_EQ(a.coeffs[1].value(), e1) << "coefficient 1 mismatch";
    EXPECT_EQ(a.coeffs[2].value(), e2) << "coefficient 2 mismatch";
    EXPECT_EQ(a.coeffs[3].value(), e3) << "coefficient 3 mismatch";
}

// p = 2013265921
static constexpr uint32_t P = BabyBear::PRIME;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Constants, Zero) {
    Ext4 z = Ext4::ZERO;
    expect_eq(z, 0, 0, 0, 0);
}

TEST(BabyBearExt4Constants, One) {
    Ext4 o = Ext4::ONE;
    expect_eq(o, 1, 0, 0, 0);
}

TEST(BabyBearExt4Constants, Two) {
    Ext4 t = Ext4::TWO;
    expect_eq(t, 2, 0, 0, 0);
}

TEST(BabyBearExt4Constants, NegOne) {
    Ext4 n = Ext4::NEG_ONE;
    // NEG_ONE = p-1 in coefficient 0
    expect_eq(n, P - 1, 0, 0, 0);
}

TEST(BabyBearExt4Constants, Generator) {
    // EXT_GENERATOR = [6, 1, 0, 0]
    Ext4 g = Ext4::GENERATOR;
    expect_eq(g, 6, 1, 0, 0);
}

TEST(BabyBearExt4Constants, W) {
    // W = 11 for BabyBear quartic (binomial constant alpha^4 = 11)
    EXPECT_EQ(Ext4::BINOMIAL_W, 11u);
}

TEST(BabyBearExt4Constants, DthRoot) {
    // dth_root = W^((p-1)/4) = 11^503316480 mod p = 1728404513
    // This is a primitive 4th root of unity: dth_root^2 = -1, dth_root^4 = 1
    EXPECT_EQ(Ext4::dth_root().value(), 1728404513u);
    BabyBear dth = Ext4::dth_root();
    // DTH_ROOT^2 = p-1 = -1  (primitive 4th root of unity property)
    EXPECT_EQ(dth.exp_u64(2).value(), P - 1);
    // DTH_ROOT^4 = 1  (Frobenius^4 = identity requires this)
    EXPECT_EQ(dth.exp_u64(4).value(), 1u);
}

// ---------------------------------------------------------------------------
// from_base and is_in_basefield
// ---------------------------------------------------------------------------

TEST(BabyBearExt4, FromBase) {
    BabyBear b(42u);
    Ext4 e = Ext4::from_base(b);
    expect_eq(e, 42, 0, 0, 0);
    EXPECT_TRUE(e.is_in_basefield());
}

TEST(BabyBearExt4, IsInBasefield) {
    EXPECT_TRUE(make(7, 0, 0, 0).is_in_basefield());
    EXPECT_FALSE(make(7, 1, 0, 0).is_in_basefield());
    EXPECT_FALSE(make(0, 0, 3, 0).is_in_basefield());
    EXPECT_FALSE(make(0, 0, 0, 5).is_in_basefield());
}

// ---------------------------------------------------------------------------
// Addition
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Arithmetic, AdditionBasic) {
    Ext4 a = make(1, 2, 3, 4);
    Ext4 b = make(5, 6, 7, 8);
    Ext4 c = a + b;
    expect_eq(c, 6, 8, 10, 12);
}

TEST(BabyBearExt4Arithmetic, AdditionModulo) {
    // Test wrap-around: (P-1) + 2 = 1
    Ext4 a = make(P - 1, 0, 0, 0);
    Ext4 b = make(2, 0, 0, 0);
    Ext4 c = a + b;
    expect_eq(c, 1, 0, 0, 0);
}

TEST(BabyBearExt4Arithmetic, AddAssign) {
    Ext4 a = make(1, 2, 3, 4);
    a += make(10, 20, 30, 40);
    expect_eq(a, 11, 22, 33, 44);
}

TEST(BabyBearExt4Arithmetic, AddZero) {
    Ext4 a = make(1, 2, 3, 4);
    EXPECT_EQ(a + Ext4::ZERO, a);
    EXPECT_EQ(Ext4::ZERO + a, a);
}

// ---------------------------------------------------------------------------
// Subtraction / Negation
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Arithmetic, SubtractionBasic) {
    Ext4 a = make(10, 20, 30, 40);
    Ext4 b = make(1, 2, 3, 4);
    Ext4 c = a - b;
    expect_eq(c, 9, 18, 27, 36);
}

TEST(BabyBearExt4Arithmetic, SubtractionUnderflow) {
    // 0 - 1 = p-1
    Ext4 zero = Ext4::ZERO;
    Ext4 one  = Ext4::ONE;
    Ext4 c = zero - one;
    expect_eq(c, P - 1, 0, 0, 0);
}

TEST(BabyBearExt4Arithmetic, Negation) {
    Ext4 a = make(1, 2, 3, 4);
    Ext4 neg_a = -a;
    Ext4 sum = a + neg_a;
    expect_eq(sum, 0, 0, 0, 0);
}

TEST(BabyBearExt4Arithmetic, SubAssign) {
    Ext4 a = make(10, 20, 30, 40);
    a -= make(1, 2, 3, 4);
    expect_eq(a, 9, 18, 27, 36);
}

// ---------------------------------------------------------------------------
// Multiplication
// ---------------------------------------------------------------------------

// Test vector computed from schoolbook formula with W=11:
// (1,2,3,4) * (5,6,7,8):
//   c0 = 1*5 + 11*(2*8 + 3*7 + 4*6) = 5 + 11*61 = 676
//   c1 = 1*6 + 2*5 + 11*(3*8 + 4*7) = 16 + 11*52 = 588
//   c2 = 1*7 + 2*6 + 3*5 + 11*(4*8) = 34 + 352  = 386
//   c3 = 1*8 + 2*7 + 3*6 + 4*5      = 60
TEST(BabyBearExt4Arithmetic, MultiplicationTestVector1) {
    Ext4 a = make(1, 2, 3, 4);
    Ext4 b = make(5, 6, 7, 8);
    Ext4 c = a * b;
    expect_eq(c, 676, 588, 386, 60);
}

// (1,0,0,0) * (x,y,z,w) = (x,y,z,w)  -- one is identity
TEST(BabyBearExt4Arithmetic, MultiplicationByOne) {
    Ext4 a = make(3, 7, 11, 5);
    Ext4 c = Ext4::ONE * a;
    expect_eq(c, 3, 7, 11, 5);
    Ext4 d = a * Ext4::ONE;
    expect_eq(d, 3, 7, 11, 5);
}

// Commutativity: a*b == b*a
TEST(BabyBearExt4Arithmetic, MultiplicationCommutative) {
    Ext4 a = make(100, 200, 300, 400);
    Ext4 b = make(111, 222, 333, 444);
    EXPECT_EQ(a * b, b * a);
}

// Associativity: (a*b)*c == a*(b*c)
TEST(BabyBearExt4Arithmetic, MultiplicationAssociative) {
    Ext4 a = make(1, 2, 3, 4);
    Ext4 b = make(5, 6, 7, 8);
    Ext4 c = make(9, 10, 11, 12);
    EXPECT_EQ((a * b) * c, a * (b * c));
}

// Distributivity: a*(b+c) == a*b + a*c
TEST(BabyBearExt4Arithmetic, MultiplicationDistributive) {
    Ext4 a = make(1, 2, 3, 4);
    Ext4 b = make(5, 6, 7, 8);
    Ext4 c = make(9, 10, 11, 12);
    EXPECT_EQ(a * (b + c), a * b + a * c);
}

// Multiply by zero
TEST(BabyBearExt4Arithmetic, MultiplicationByZero) {
    Ext4 a = make(1, 2, 3, 4);
    EXPECT_EQ(a * Ext4::ZERO, Ext4::ZERO);
    EXPECT_EQ(Ext4::ZERO * a, Ext4::ZERO);
}

// MulAssign
TEST(BabyBearExt4Arithmetic, MulAssign) {
    Ext4 a = make(1, 2, 3, 4);
    Ext4 b = make(5, 6, 7, 8);
    Ext4 expected = a * b;
    a *= b;
    EXPECT_EQ(a, expected);
}

// ---------------------------------------------------------------------------
// Squaring
// ---------------------------------------------------------------------------

// (1,2,3,4)^2:
//   c0 = 1 + 11*(2*2*4 + 9) = 1 + 11*(16+9) = 1 + 275 = 276
//   c1 = 2*(1*2 + 11*3*4)   = 2*(2 + 132)   = 268
//   c2 = 2*1*3 + 4 + 11*16  = 6 + 4 + 176   = 186
//   c3 = 2*(1*4 + 2*3)       = 2*10          = 20
TEST(BabyBearExt4Arithmetic, SquaringTestVector) {
    Ext4 a = make(1, 2, 3, 4);
    Ext4 sq = a.square();
    expect_eq(sq, 276, 268, 186, 20);
}

// square() must equal mul(self)
TEST(BabyBearExt4Arithmetic, SquaringMatchesMul) {
    Ext4 a = make(7, 13, 19, 23);
    EXPECT_EQ(a.square(), a * a);
}

// ---------------------------------------------------------------------------
// Scalar multiplication
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Arithmetic, ScalarMul) {
    Ext4 a = make(1, 2, 3, 4);
    BabyBear s(5u);
    Ext4 c = a * s;
    expect_eq(c, 5, 10, 15, 20);
}

// ---------------------------------------------------------------------------
// Halving
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Arithmetic, Halve) {
    Ext4 a = make(2, 4, 6, 8);
    Ext4 h = a.halve();
    expect_eq(h, 1, 2, 3, 4);
}

TEST(BabyBearExt4Arithmetic, HalveRoundtrip) {
    Ext4 a = make(100, 200, 300, 400);
    // a.halve() * 2 should equal a
    Ext4 two = Ext4::TWO;
    EXPECT_EQ(a.halve() * two, a);
}

TEST(BabyBearExt4Arithmetic, HalveOdd) {
    // 1 / 2 = inv(2) = (P+1)/2 = 1006632961
    Ext4 one = Ext4::ONE;
    Ext4 h = one.halve();
    EXPECT_EQ(h.coeffs[0].value(), (P + 1) / 2);
}

// ---------------------------------------------------------------------------
// Inverse
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Arithmetic, InverseTimesOriginalIsOne) {
    Ext4 a = make(1, 2, 3, 4);
    Ext4 inv = a.inv();
    EXPECT_EQ(a * inv, Ext4::ONE);
    EXPECT_EQ(inv * a, Ext4::ONE);
}

TEST(BabyBearExt4Arithmetic, InverseOfOne) {
    EXPECT_EQ(Ext4::ONE.inv(), Ext4::ONE);
}

TEST(BabyBearExt4Arithmetic, InverseRandomElements) {
    // Test several random-ish elements
    std::array<std::array<uint32_t, 4>, 5> tests = {{
        {7, 13, 19, 23},
        {100, 200, 300, 400},
        {1000000000u, 1, 2, 3},
        {P - 1, P - 2, P - 3, P - 4},
        {6, 1, 0, 0}  // generator
    }};
    for (auto& t : tests) {
        Ext4 a = make(t[0], t[1], t[2], t[3]);
        Ext4 inv_a = a.inv();
        EXPECT_EQ(a * inv_a, Ext4::ONE) << "Failed for (" << t[0] << "," << t[1] << "," << t[2] << "," << t[3] << ")";
    }
}

// ---------------------------------------------------------------------------
// Division
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Arithmetic, DivisionIsMultiplyByInverse) {
    Ext4 a = make(5, 10, 15, 20);
    Ext4 b = make(1, 2, 3, 4);
    EXPECT_EQ(a / b, a * b.inv());
}

TEST(BabyBearExt4Arithmetic, DivideByOne) {
    Ext4 a = make(5, 10, 15, 20);
    EXPECT_EQ(a / Ext4::ONE, a);
}

TEST(BabyBearExt4Arithmetic, DivideBySelf) {
    Ext4 a = make(3, 7, 11, 5);
    EXPECT_EQ(a / a, Ext4::ONE);
}

// ---------------------------------------------------------------------------
// Exponentiation
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Exponentiation, ExpU64Zero) {
    Ext4 a = make(1, 2, 3, 4);
    EXPECT_EQ(a.exp_u64(0), Ext4::ONE);
}

TEST(BabyBearExt4Exponentiation, ExpU64One) {
    Ext4 a = make(1, 2, 3, 4);
    EXPECT_EQ(a.exp_u64(1), a);
}

TEST(BabyBearExt4Exponentiation, ExpU64Two) {
    Ext4 a = make(1, 2, 3, 4);
    EXPECT_EQ(a.exp_u64(2), a.square());
}

TEST(BabyBearExt4Exponentiation, ExpU64MatchesRepeatedMul) {
    Ext4 a = make(2, 1, 0, 0);  // generator-like element
    Ext4 a3 = a * a * a;
    EXPECT_EQ(a.exp_u64(3), a3);

    Ext4 a5 = a * a * a * a * a;
    EXPECT_EQ(a.exp_u64(5), a5);
}

TEST(BabyBearExt4Exponentiation, ExpPowerOf2) {
    Ext4 a = make(2, 1, 0, 0);
    // a^(2^0) = a
    EXPECT_EQ(a.exp_power_of_2(0), a);
    // a^(2^1) = a^2
    EXPECT_EQ(a.exp_power_of_2(1), a.square());
    // a^(2^2) = (a^2)^2 = a^4
    EXPECT_EQ(a.exp_power_of_2(2), a.square().square());
}

TEST(BabyBearExt4Exponentiation, InverseViaExp) {
    // a * a^(-1) = 1; also a^(order-1) = a^(-1)
    Ext4 a = make(1, 2, 3, 4);
    Ext4 inv_a = a.inv();
    // Verify via division
    EXPECT_EQ(Ext4::ONE / a, inv_a);
}

// ---------------------------------------------------------------------------
// Powers iterator (vector form)
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Powers, PowersVector) {
    Ext4 a = make(2, 1, 0, 0);
    auto pows = a.powers(5);
    ASSERT_EQ(pows.size(), 5u);
    Ext4 current_power = Ext4::ONE;
    for (const auto& p : pows) {
        EXPECT_EQ(p, current_power);
        current_power *= a;
    }
}

// ---------------------------------------------------------------------------
// Frobenius
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Frobenius, FrobeniusOfBaseElement) {
    // For base-field elements, Frobenius is the identity
    Ext4 a = Ext4::from_base(BabyBear(42u));
    EXPECT_EQ(a.frobenius(), a);
}

TEST(BabyBearExt4Frobenius, FrobeniusOfX) {
    // frobenius(0,1,0,0) = (0, DTH_ROOT, 0, 0)
    Ext4 x = make(0, 1, 0, 0);
    Ext4 fx = x.frobenius();
    EXPECT_EQ(fx.coeffs[0].value(), 0u);
    EXPECT_EQ(fx.coeffs[1].value(), 1728404513u);  // dth_root = W^((p-1)/4) mod p
    EXPECT_EQ(fx.coeffs[2].value(), 0u);
    EXPECT_EQ(fx.coeffs[3].value(), 0u);
}

TEST(BabyBearExt4Frobenius, FrobeniusIsFieldAuto) {
    // Frobenius is a field automorphism: frob(a*b) == frob(a)*frob(b)
    Ext4 a = make(1, 2, 3, 4);
    Ext4 b = make(5, 6, 7, 8);
    EXPECT_EQ((a * b).frobenius(), a.frobenius() * b.frobenius());
}

TEST(BabyBearExt4Frobenius, RepeatedFrobenius) {
    Ext4 a = make(1, 2, 3, 4);
    // repeated_frobenius(2) == frobenius().frobenius()
    EXPECT_EQ(a.repeated_frobenius(2), a.frobenius().frobenius());
    // repeated_frobenius(0) == identity
    EXPECT_EQ(a.repeated_frobenius(0), a);
    // For a degree-4 extension over F_p, Frobenius^4 = identity
    EXPECT_EQ(a.repeated_frobenius(4), a);
}

// ---------------------------------------------------------------------------
// Ring/field axioms
// ---------------------------------------------------------------------------

TEST(BabyBearExt4Axioms, AdditiveIdentity) {
    Ext4 a = make(3, 7, 11, 5);
    EXPECT_EQ(a + Ext4::ZERO, a);
    EXPECT_EQ(Ext4::ZERO + a, a);
}

TEST(BabyBearExt4Axioms, AdditiveInverse) {
    Ext4 a = make(3, 7, 11, 5);
    EXPECT_EQ(a + (-a), Ext4::ZERO);
}

TEST(BabyBearExt4Axioms, MultiplicativeIdentity) {
    Ext4 a = make(3, 7, 11, 5);
    EXPECT_EQ(a * Ext4::ONE, a);
    EXPECT_EQ(Ext4::ONE * a, a);
}

TEST(BabyBearExt4Axioms, TwoEqualsOneAddOne) {
    EXPECT_EQ(Ext4::TWO, Ext4::ONE + Ext4::ONE);
}

TEST(BabyBearExt4Axioms, NEG_ONE) {
    EXPECT_EQ(Ext4::ONE + Ext4::NEG_ONE, Ext4::ZERO);
}


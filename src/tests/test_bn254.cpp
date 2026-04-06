/**
 * @file test_bn254.cpp
 * @brief Google Test suite for Bn254 field element
 *
 * Test vectors generated from the Rust Plonky3 p3-bn254 implementation.
 */

#include <gtest/gtest.h>
#include "bn254.hpp"
#include "poseidon2_bn254.hpp"

using namespace p3_field;

// ===========================================================================
// Helpers
// ===========================================================================

/// Compare a Bn254 element against expected canonical limbs
static void expect_canonical(const Bn254& a, uint64_t e0, uint64_t e1, uint64_t e2, uint64_t e3) {
    uint64_t canonical[4];
    a.as_canonical(canonical);
    EXPECT_EQ(canonical[0], e0) << "limb 0 mismatch";
    EXPECT_EQ(canonical[1], e1) << "limb 1 mismatch";
    EXPECT_EQ(canonical[2], e2) << "limb 2 mismatch";
    EXPECT_EQ(canonical[3], e3) << "limb 3 mismatch";
}

/// Compare a Bn254 element against a small canonical value
static void expect_small(const Bn254& a, uint64_t expected) {
    expect_canonical(a, expected, 0, 0, 0);
}

// ===========================================================================
// Constants
// ===========================================================================

TEST(Bn254Constants, Zero) {
    Bn254 z = Bn254::zero_val();
    EXPECT_TRUE(z.is_zero());
    expect_small(z, 0);
}

TEST(Bn254Constants, One) {
    Bn254 o = Bn254::one_val();
    EXPECT_FALSE(o.is_zero());
    expect_small(o, 1);
}

TEST(Bn254Constants, Two) {
    Bn254 t = Bn254::two_val();
    expect_small(t, 2);
}

TEST(Bn254Constants, NegOne) {
    // -1 mod P = P - 1
    Bn254 n = Bn254::neg_one_val();
    uint64_t canonical[4];
    n.as_canonical(canonical);
    // P - 1 = 0x43e1f593f0000000, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029
    EXPECT_EQ(canonical[0], 0x43e1f593f0000000ULL);
    EXPECT_EQ(canonical[1], 0x2833e84879b97091ULL);
    EXPECT_EQ(canonical[2], 0xb85045b68181585dULL);
    EXPECT_EQ(canonical[3], 0x30644e72e131a029ULL);
}

TEST(Bn254Constants, Generator) {
    Bn254 g = Bn254::generator();
    expect_small(g, 5);
}

TEST(Bn254Constants, StaticConstants) {
    EXPECT_TRUE(Bn254::ZERO.is_zero());
    EXPECT_EQ(Bn254::ONE, Bn254::one_val());
    EXPECT_EQ(Bn254::TWO, Bn254::two_val());
    EXPECT_EQ(Bn254::NEG_ONE, Bn254::neg_one_val());
    EXPECT_EQ(Bn254::GENERATOR, Bn254::generator());
}

// ===========================================================================
// Construction
// ===========================================================================

TEST(Bn254Construction, FromU64) {
    Bn254 a = Bn254::from_u64(0);
    EXPECT_TRUE(a.is_zero());
    EXPECT_EQ(a, Bn254::zero_val());

    Bn254 b = Bn254::from_u64(1);
    EXPECT_EQ(b, Bn254::one_val());

    Bn254 c = Bn254::from_u64(5);
    EXPECT_EQ(c, Bn254::generator());

    Bn254 d = Bn254::from_u64(100);
    expect_small(d, 100);
}

TEST(Bn254Construction, FromCanonical) {
    // Test with canonical limbs for a large value
    uint64_t limbs[4] = { 0x1234567890abcdef, 0xfedcba0987654321, 0x1111111111111111, 0x0 };
    Bn254 c = Bn254::from_canonical(limbs);

    uint64_t canonical[4];
    c.as_canonical(canonical);
    EXPECT_EQ(canonical[0], limbs[0]);
    EXPECT_EQ(canonical[1], limbs[1]);
    EXPECT_EQ(canonical[2], limbs[2]);
    EXPECT_EQ(canonical[3], limbs[3]);
}

TEST(Bn254Construction, FromI64) {
    Bn254 pos = Bn254::from_i64(42);
    expect_small(pos, 42);

    Bn254 neg = Bn254::from_i64(-1);
    EXPECT_EQ(neg, Bn254::neg_one_val());

    Bn254 neg100 = Bn254::from_i64(-100);
    Bn254 pos100 = Bn254::from_u64(100);
    EXPECT_EQ(neg100, -pos100);
}

// ===========================================================================
// Arithmetic — test vectors from Rust
// a = 100, b = 200
// ===========================================================================

TEST(Bn254Arithmetic, Addition) {
    Bn254 a = Bn254::from_u64(100);
    Bn254 b = Bn254::from_u64(200);
    Bn254 sum = a + b;
    expect_small(sum, 300);
}

TEST(Bn254Arithmetic, Subtraction) {
    Bn254 a = Bn254::from_u64(100);
    Bn254 b = Bn254::from_u64(200);
    // a - b = 100 - 200 = P - 100
    Bn254 diff = a - b;
    Bn254 neg_a = -a;
    EXPECT_EQ(diff, neg_a);
    // Verify: diff + b = a
    EXPECT_EQ(diff + b, a);
}

TEST(Bn254Arithmetic, SubtractionSameValue) {
    Bn254 a = Bn254::from_u64(12345);
    Bn254 result = a - a;
    EXPECT_TRUE(result.is_zero());
}

TEST(Bn254Arithmetic, Multiplication) {
    Bn254 a = Bn254::from_u64(100);
    Bn254 b = Bn254::from_u64(200);
    Bn254 prod = a * b;
    expect_small(prod, 20000);
}

TEST(Bn254Arithmetic, MultiplicationByZero) {
    Bn254 a = Bn254::from_u64(12345);
    Bn254 z = Bn254::zero_val();
    EXPECT_TRUE((a * z).is_zero());
    EXPECT_TRUE((z * a).is_zero());
}

TEST(Bn254Arithmetic, MultiplicationByOne) {
    Bn254 a = Bn254::from_u64(12345);
    Bn254 o = Bn254::one_val();
    EXPECT_EQ(a * o, a);
    EXPECT_EQ(o * a, a);
}

TEST(Bn254Arithmetic, Negation) {
    Bn254 a = Bn254::from_u64(100);
    Bn254 neg_a = -a;
    // a + (-a) = 0
    EXPECT_TRUE((a + neg_a).is_zero());
    // (-(-a)) = a
    EXPECT_EQ(-neg_a, a);
}

TEST(Bn254Arithmetic, Double) {
    Bn254 a = Bn254::from_u64(100);
    expect_small(a.double_val(), 200);
}

TEST(Bn254Arithmetic, Square) {
    Bn254 a = Bn254::from_u64(100);
    expect_small(a.square(), 10000);
}

TEST(Bn254Arithmetic, Cube) {
    Bn254 a = Bn254::from_u64(100);
    expect_small(a.cube(), 1000000);
}

TEST(Bn254Arithmetic, Halve) {
    Bn254 a = Bn254::from_u64(100);
    expect_small(a.halve(), 50);

    // halve(1) * 2 = 1
    Bn254 half_one = Bn254::one_val().halve();
    EXPECT_EQ(half_one.double_val(), Bn254::one_val());

    // halve(odd) * 2 should recover the original
    Bn254 odd = Bn254::from_u64(7);
    EXPECT_EQ(odd.halve().double_val(), odd);
}

// ===========================================================================
// Inverse
// ===========================================================================

TEST(Bn254Inverse, SmallValues) {
    Bn254 a = Bn254::from_u64(100);
    Bn254 a_inv = a.inv();
    // a * a^{-1} = 1
    EXPECT_EQ(a * a_inv, Bn254::one_val());
}

TEST(Bn254Inverse, One) {
    Bn254 one = Bn254::one_val();
    EXPECT_EQ(one.inv(), one);
}

TEST(Bn254Inverse, NegOne) {
    Bn254 neg_one = Bn254::neg_one_val();
    EXPECT_EQ(neg_one.inv(), neg_one);
}

TEST(Bn254Inverse, Generator) {
    Bn254 g = Bn254::generator();
    Bn254 g_inv = g.inv();
    EXPECT_EQ(g * g_inv, Bn254::one_val());
}

TEST(Bn254Inverse, LargeValue) {
    uint64_t limbs[4] = { 0x1234567890abcdef, 0xfedcba0987654321, 0x1111111111111111, 0x0 };
    Bn254 c = Bn254::from_canonical(limbs);
    Bn254 c_inv = c.inv();
    EXPECT_EQ(c * c_inv, Bn254::one_val());
}

TEST(Bn254Inverse, MultipleValues) {
    for (uint64_t v = 1; v <= 20; ++v) {
        Bn254 a = Bn254::from_u64(v);
        Bn254 a_inv = a.inv();
        EXPECT_EQ(a * a_inv, Bn254::one_val()) << "Failed for v=" << v;
    }
}

TEST(Bn254Inverse, InverseOfInverse) {
    Bn254 a = Bn254::from_u64(42);
    EXPECT_EQ(a.inv().inv(), a);
}

TEST(Bn254Inverse, ZeroThrows) {
    Bn254 z = Bn254::zero_val();
    EXPECT_THROW(z.inv(), std::runtime_error);
}

// ===========================================================================
// Large value arithmetic (cross-checked with Rust)
// ===========================================================================

TEST(Bn254LargeArithmetic, AddLargeValues) {
    uint64_t c_limbs[4] = { 0x1234567890abcdef, 0xfedcba0987654321, 0x1111111111111111, 0x0 };
    uint64_t d_limbs[4] = { 0xabcdef0123456789, 0x9876543210fedcba, 0x2222222222222222, 0x0 };

    Bn254 c = Bn254::from_canonical(c_limbs);
    Bn254 d = Bn254::from_canonical(d_limbs);

    // c + d: verify by checking c + d - d == c
    Bn254 sum = c + d;
    EXPECT_EQ(sum - d, c);
    EXPECT_EQ(sum - c, d);
}

TEST(Bn254LargeArithmetic, SubLargeValues) {
    uint64_t c_limbs[4] = { 0x1234567890abcdef, 0xfedcba0987654321, 0x1111111111111111, 0x0 };
    uint64_t d_limbs[4] = { 0xabcdef0123456789, 0x9876543210fedcba, 0x2222222222222222, 0x0 };

    Bn254 c = Bn254::from_canonical(c_limbs);
    Bn254 d = Bn254::from_canonical(d_limbs);

    Bn254 diff = c - d;
    // Verify: diff + d == c
    EXPECT_EQ(diff + d, c);
}

TEST(Bn254LargeArithmetic, MulLargeValues) {
    uint64_t c_limbs[4] = { 0x1234567890abcdef, 0xfedcba0987654321, 0x1111111111111111, 0x0 };
    uint64_t d_limbs[4] = { 0xabcdef0123456789, 0x9876543210fedcba, 0x2222222222222222, 0x0 };

    Bn254 c = Bn254::from_canonical(c_limbs);
    Bn254 d = Bn254::from_canonical(d_limbs);

    // c * d / d == c
    Bn254 prod = c * d;
    EXPECT_EQ(prod * d.inv(), c);
}

TEST(Bn254LargeArithmetic, InverseLargeValue) {
    uint64_t c_limbs[4] = { 0x1234567890abcdef, 0xfedcba0987654321, 0x1111111111111111, 0x0 };

    Bn254 c = Bn254::from_canonical(c_limbs);
    Bn254 c_inv = c.inv();
    EXPECT_EQ(c * c_inv, Bn254::one_val());
}

// ===========================================================================
// Two-adic generators
// ===========================================================================

TEST(Bn254TwoAdic, GeneratorBits0) {
    // two_adic_generator(0) = 1
    Bn254 g = Bn254::two_adic_generator(0);
    EXPECT_EQ(g, Bn254::one_val());
}

TEST(Bn254TwoAdic, GeneratorBits1) {
    // two_adic_generator(1) = P - 1 (since (P-1)^2 = 1)
    Bn254 g = Bn254::two_adic_generator(1);
    EXPECT_EQ(g, Bn254::neg_one_val());
}

TEST(Bn254TwoAdic, GeneratorOrder) {
    // For each bits value, g^(2^bits) should equal 1
    for (size_t bits = 1; bits <= 10; ++bits) {
        Bn254 g = Bn254::two_adic_generator(bits);
        Bn254 power = g;
        for (size_t i = 0; i < bits; ++i) {
            power = power.square();
        }
        EXPECT_EQ(power, Bn254::one_val()) << "Failed for bits=" << bits;
    }
}

TEST(Bn254TwoAdic, GeneratorNotPremature) {
    // g^(2^(bits-1)) should NOT be 1 (it should be -1 for primitive root)
    for (size_t bits = 2; bits <= 10; ++bits) {
        Bn254 g = Bn254::two_adic_generator(bits);
        Bn254 power = g;
        for (size_t i = 0; i < bits - 1; ++i) {
            power = power.square();
        }
        EXPECT_NE(power, Bn254::one_val()) << "Premature order for bits=" << bits;
    }
}

TEST(Bn254TwoAdic, MaxAdicity) {
    // bits = 28 (TWO_ADICITY)
    Bn254 g = Bn254::two_adic_generator(28);
    Bn254 power = g;
    for (size_t i = 0; i < 28; ++i) {
        power = power.square();
    }
    EXPECT_EQ(power, Bn254::one_val());
}

TEST(Bn254TwoAdic, ExceedsTwoAdicityThrows) {
    EXPECT_THROW(Bn254::two_adic_generator(29), std::invalid_argument);
}

// ===========================================================================
// Exponentiation
// ===========================================================================

TEST(Bn254Exponentiation, ExpU64) {
    Bn254 base = Bn254::from_u64(3);

    // 3^0 = 1
    EXPECT_EQ(base.exp_u64(0), Bn254::one_val());
    // 3^1 = 3
    EXPECT_EQ(base.exp_u64(1), base);
    // 3^2 = 9
    expect_small(base.exp_u64(2), 9);
    // 3^10 = 59049
    expect_small(base.exp_u64(10), 59049);
}

TEST(Bn254Exponentiation, ExpU256) {
    Bn254 base = Bn254::from_u64(2);

    // 2^10 = 1024
    uint64_t exp[4] = { 10, 0, 0, 0 };
    expect_small(base.exp_u256(exp), 1024);
}

// ===========================================================================
// Operator assignments
// ===========================================================================

TEST(Bn254Operators, AddAssign) {
    Bn254 a = Bn254::from_u64(10);
    Bn254 b = Bn254::from_u64(20);
    a += b;
    expect_small(a, 30);
}

TEST(Bn254Operators, SubAssign) {
    Bn254 a = Bn254::from_u64(30);
    Bn254 b = Bn254::from_u64(20);
    a -= b;
    expect_small(a, 10);
}

TEST(Bn254Operators, MulAssign) {
    Bn254 a = Bn254::from_u64(10);
    Bn254 b = Bn254::from_u64(20);
    a *= b;
    expect_small(a, 200);
}

// ===========================================================================
// Field axioms
// ===========================================================================

TEST(Bn254FieldAxioms, Commutativity) {
    Bn254 a = Bn254::from_u64(123);
    Bn254 b = Bn254::from_u64(456);
    EXPECT_EQ(a + b, b + a);
    EXPECT_EQ(a * b, b * a);
}

TEST(Bn254FieldAxioms, Associativity) {
    Bn254 a = Bn254::from_u64(11);
    Bn254 b = Bn254::from_u64(22);
    Bn254 c = Bn254::from_u64(33);
    EXPECT_EQ((a + b) + c, a + (b + c));
    EXPECT_EQ((a * b) * c, a * (b * c));
}

TEST(Bn254FieldAxioms, Distributivity) {
    Bn254 a = Bn254::from_u64(7);
    Bn254 b = Bn254::from_u64(13);
    Bn254 c = Bn254::from_u64(19);
    EXPECT_EQ(a * (b + c), a * b + a * c);
}

TEST(Bn254FieldAxioms, AdditiveIdentity) {
    Bn254 a = Bn254::from_u64(12345);
    EXPECT_EQ(a + Bn254::zero_val(), a);
}

TEST(Bn254FieldAxioms, MultiplicativeIdentity) {
    Bn254 a = Bn254::from_u64(12345);
    EXPECT_EQ(a * Bn254::one_val(), a);
}

TEST(Bn254FieldAxioms, AdditiveInverse) {
    Bn254 a = Bn254::from_u64(12345);
    EXPECT_TRUE((a + (-a)).is_zero());
}

TEST(Bn254FieldAxioms, MultiplicativeInverse) {
    Bn254 a = Bn254::from_u64(12345);
    EXPECT_EQ(a * a.inv(), Bn254::one_val());
}

// ===========================================================================
// Injective exponentiation (S-box)
// ===========================================================================

TEST(Bn254SBox, InjectiveExpN) {
    // x^5 for small values
    Bn254 x = Bn254::from_u64(2);
    Bn254 x5 = x.injective_exp_n<5>();
    expect_small(x5, 32); // 2^5 = 32

    Bn254 y = Bn254::from_u64(3);
    Bn254 y5 = y.injective_exp_n<5>();
    expect_small(y5, 243); // 3^5 = 243
}

TEST(Bn254SBox, InjectiveExpRoundTrip) {
    // x^5 then fifth-root should recover x
    for (uint64_t v = 1; v <= 10; ++v) {
        Bn254 x = Bn254::from_u64(v);
        Bn254 x5 = x.injective_exp_n<5>();
        Bn254 x_back = x5.injective_exp_root_n<5>();
        EXPECT_EQ(x_back, x) << "Round trip failed for v=" << v;
    }
}

// ===========================================================================
// Poseidon2 BN254
// ===========================================================================

TEST(Bn254Poseidon2, InternalMatmul) {
    // Test the internal diffusion matrix:
    //   [2 1 1]
    //   [1 2 1]
    //   [1 1 3]
    std::array<Bn254, 3> state = {
        Bn254::from_u64(1), Bn254::from_u64(2), Bn254::from_u64(3)
    };
    poseidon2::bn254_matmul_internal(state);
    // sum = 1 + 2 + 3 = 6
    // new[0] = 1 + 6 = 7  (= 2*1 + 2 + 3)
    // new[1] = 2 + 6 = 8  (= 1 + 2*2 + 3)
    // new[2] = 2*3 + 6 = 12 (= 1 + 2 + 3*3)
    expect_small(state[0], 7);
    expect_small(state[1], 8);
    expect_small(state[2], 12);
}

TEST(Bn254Poseidon2, PermutationTestVector) {
    // Test vector from Rust: permute([1, 2, 3])
    auto poseidon2 = poseidon2::default_bn254_poseidon2();

    std::array<Bn254, 3> state = {
        Bn254::from_u64(1), Bn254::from_u64(2), Bn254::from_u64(3)
    };
    poseidon2->permute_mut(state);

    // Expected output (canonical form from Rust reference)
    expect_canonical(state[0],
        0x009aca578d210768ULL, 0xdd31c210c912aacaULL,
        0x1ec6fdbcd5dc92caULL, 0x0a799a621cac2ceaULL);
    expect_canonical(state[1],
        0x2191be0e1598876aULL, 0xd14b1f5c6b1856a1ULL,
        0xce99b299edfd70faULL, 0x1570f61795255f02ULL);
    expect_canonical(state[2],
        0x694ce6e9b8dc3019ULL, 0x2e722b76c3a25017ULL,
        0xdde96c087a306f40ULL, 0x285e9571da987271ULL);
}

TEST(Bn254Poseidon2, PermutationZeroInput) {
    // Permuting [0, 0, 0] should produce a non-trivial output
    auto poseidon2 = poseidon2::default_bn254_poseidon2();

    std::array<Bn254, 3> state = {
        Bn254::zero_val(), Bn254::zero_val(), Bn254::zero_val()
    };
    poseidon2->permute_mut(state);

    // Output should be non-zero
    EXPECT_FALSE(state[0].is_zero());
    EXPECT_FALSE(state[1].is_zero());
    EXPECT_FALSE(state[2].is_zero());
}

TEST(Bn254Poseidon2, PermutationDifferentInputs) {
    // Different inputs should produce different outputs
    auto poseidon2 = poseidon2::default_bn254_poseidon2();

    std::array<Bn254, 3> s1 = {
        Bn254::from_u64(1), Bn254::from_u64(2), Bn254::from_u64(3)
    };
    std::array<Bn254, 3> s2 = {
        Bn254::from_u64(4), Bn254::from_u64(5), Bn254::from_u64(6)
    };
    poseidon2->permute_mut(s1);
    poseidon2->permute_mut(s2);

    EXPECT_NE(s1[0], s2[0]);
    EXPECT_NE(s1[1], s2[1]);
    EXPECT_NE(s1[2], s2[2]);
}

/**
 * @file test_two_adic.cpp
 * @brief Google Tests for TwoAdicField traits (BabyBear and BinomialExtensionField<BabyBear,4>)
 *
 * Validates:
 *  - BabyBear::TWO_ADICITY == 27
 *  - BabyBear::two_adic_generator(k)^(2^k) == 1 for all k in [0, 27]
 *  - BabyBear::two_adic_generator(k)^(2^(k-1)) != 1 for k in [1, 27]
 *  - BabyBear::GENERATOR_VAL == 31 is a primitive root
 *  - Extension field TWO_ADICITY == 29
 *  - Extension two_adic_generator(k)^(2^k) == 1 for all k in [0, 29]
 *  - Extension two_adic_generator(k)^(2^(k-1)) != 1 for k in [1, 29]
 *  - GENERATOR constants match expected values
 *  - powers() iterator returns correct successive powers
 */

#include <gtest/gtest.h>
#include "baby_bear.hpp"
#include "extension_field.hpp"
#include <array>

using namespace p3_field;

// ---------------------------------------------------------------------------
// Helper: raise extension field element to 2^n
// ---------------------------------------------------------------------------
static BabyBear4 ext_pow2(const BabyBear4& g, size_t n) {
    BabyBear4 result = g;
    for (size_t i = 0; i < n; ++i) {
        result = result * result;
    }
    return result;
}

static BabyBear4 ext_one() { return BabyBear4::one_val(); }

// ---------------------------------------------------------------------------
// BabyBear two-adic generator tests
// ---------------------------------------------------------------------------

TEST(BabyBearTwoAdic, TWO_ADICITY) {
    EXPECT_EQ(BabyBear::TWO_ADICITY, 27u);
}

TEST(BabyBearTwoAdic, GeneratorVal) {
    EXPECT_EQ(BabyBear::GENERATOR_VAL, 31u);
}

TEST(BabyBearTwoAdic, GeneratorIsPrimitiveRoot) {
    // 31 is a primitive root mod p iff 31^((p-1)/q) != 1 for prime factors q of p-1
    // p-1 = 2^27 * 3 * 5
    BabyBear gen(static_cast<uint32_t>(31u));
    BabyBear one = BabyBear::one_val();

    uint64_t pm1 = BabyBear::PRIME - 1;
    EXPECT_NE(gen.exp_u64(pm1 / 2), one)  << "31 should not be a QR mod p";
    EXPECT_NE(gen.exp_u64(pm1 / 3), one)  << "31^((p-1)/3) should not be 1";
    EXPECT_NE(gen.exp_u64(pm1 / 5), one)  << "31^((p-1)/5) should not be 1";
    EXPECT_EQ(gen.exp_u64(pm1), one)       << "31^(p-1) should be 1 (Fermat)";
}

TEST(BabyBearTwoAdic, TwoAdicGeneratorBits0) {
    // Generator for bits=0 should be 1 (trivial)
    BabyBear g = BabyBear::two_adic_generator(0);
    EXPECT_EQ(g, BabyBear::one_val());
}

TEST(BabyBearTwoAdic, TwoAdicGeneratorBits1) {
    // Generator for bits=1 should be -1 (order 2)
    BabyBear g = BabyBear::two_adic_generator(1);
    BabyBear one = BabyBear::one_val();
    EXPECT_EQ(g * g, one) << "g^2 should be 1 for bits=1";
    EXPECT_NE(g, one)     << "g should not be 1 for bits=1";
}

TEST(BabyBearTwoAdic, TwoAdicGeneratorOrderProperty) {
    // For each k in [1, 27]: g^(2^k) == 1 and g^(2^(k-1)) != 1
    BabyBear one = BabyBear::one_val();
    for (size_t k = 1; k <= BabyBear::TWO_ADICITY; ++k) {
        BabyBear g = BabyBear::two_adic_generator(k);
        // g^(2^k) == 1
        BabyBear power = g;
        for (size_t i = 0; i < k; ++i) power = power * power;
        EXPECT_EQ(power, one) << "g^(2^" << k << ") should be 1";
        // g^(2^(k-1)) != 1
        BabyBear half_power = g;
        for (size_t i = 0; i < k - 1; ++i) half_power = half_power * half_power;
        EXPECT_NE(half_power, one) << "g^(2^" << (k-1) << ") should not be 1 for bits=" << k;
    }
}

TEST(BabyBearTwoAdic, TwoAdicGenerator27KnownValue) {
    // The primitive 2^27-th root of unity for BabyBear is 440564289
    BabyBear g27 = BabyBear::two_adic_generator(27);
    EXPECT_EQ(g27.value(), 440564289u);
}

TEST(BabyBearTwoAdic, TwoAdicGeneratorChain) {
    // g(k)^2 should have order 2^(k-1), i.e., g(k)^2 == g(k-1) is possible
    // but not guaranteed (could be a different primitive root of same order).
    // We verify g(k)^(2^(k-1)) is still a primitive root of unity of order 2^(k-1):
    // g(k)^2 has order exactly 2^(k-1).
    BabyBear one = BabyBear::one_val();
    for (size_t k = 2; k <= BabyBear::TWO_ADICITY; ++k) {
        BabyBear gk = BabyBear::two_adic_generator(k);
        BabyBear gk_sq = gk * gk; // should have order 2^(k-1)
        // gk_sq^(2^(k-1)) == 1
        BabyBear power = gk_sq;
        for (size_t i = 0; i < k - 1; ++i) power = power * power;
        EXPECT_EQ(power, one) << "g(k)^2 should have order dividing 2^(k-1) for k=" << k;
        // gk_sq^(2^(k-2)) != 1 (for k >= 2)
        if (k >= 2) {
            BabyBear half = gk_sq;
            for (size_t i = 0; i < k - 2; ++i) half = half * half;
            EXPECT_NE(half, one) << "g(k)^2 should not have order 2^(k-2) for k=" << k;
        }
    }
}

TEST(BabyBearTwoAdic, TwoAdicGeneratorOutOfRange) {
    EXPECT_THROW(BabyBear::two_adic_generator(28), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// BabyBear extension field two-adic generator tests (via ext_two_adic_generator)
// ---------------------------------------------------------------------------

TEST(BabyBearExtTwoAdic, ExtTwoAdicGeneratorBitsLE27EmbedFromBase) {
    // For bits <= 27, the extension generator embeds the base field generator
    for (size_t k = 0; k <= 27; ++k) {
        auto ext_gen = BabyBear::ext_two_adic_generator(k);
        BabyBear base_gen = BabyBear::two_adic_generator(k);
        EXPECT_EQ(ext_gen[0], base_gen) << "ext_gen[0] should equal base two_adic_generator for bits=" << k;
        EXPECT_EQ(ext_gen[1], BabyBear()) << "ext_gen[1] should be 0 for bits=" << k;
        EXPECT_EQ(ext_gen[2], BabyBear()) << "ext_gen[2] should be 0 for bits=" << k;
        EXPECT_EQ(ext_gen[3], BabyBear()) << "ext_gen[3] should be 0 for bits=" << k;
    }
}

TEST(BabyBearExtTwoAdic, ExtTwoAdicGeneratorBits28KnownValue) {
    // bits=28: [0, 0, 929455875, 0]
    auto g = BabyBear::ext_two_adic_generator(28);
    EXPECT_EQ(g[0].value(), 0u);
    EXPECT_EQ(g[1].value(), 0u);
    EXPECT_EQ(g[2].value(), 929455875u);
    EXPECT_EQ(g[3].value(), 0u);
}

TEST(BabyBearExtTwoAdic, ExtTwoAdicGeneratorBits29KnownValue) {
    // bits=29: [0, 0, 0, 1483681942]
    auto g = BabyBear::ext_two_adic_generator(29);
    EXPECT_EQ(g[0].value(), 0u);
    EXPECT_EQ(g[1].value(), 0u);
    EXPECT_EQ(g[2].value(), 0u);
    EXPECT_EQ(g[3].value(), 1483681942u);
}

// ---------------------------------------------------------------------------
// BinomialExtensionField<BabyBear, 4> two-adic generator tests
// ---------------------------------------------------------------------------

TEST(ExtFieldTwoAdic, TWO_ADICITY) {
    EXPECT_EQ(BabyBear4TwoAdic::TWO_ADICITY, 29u);
}

TEST(ExtFieldTwoAdic, GeneratorValue) {
    BabyBear4 gen = BabyBear4TwoAdic::generator();
    EXPECT_EQ(gen[0].value(), 6u);
    EXPECT_EQ(gen[1].value(), 1u);
    EXPECT_EQ(gen[2].value(), 0u);
    EXPECT_EQ(gen[3].value(), 0u);
}

TEST(ExtFieldTwoAdic, TwoAdicGeneratorBits0) {
    BabyBear4 g = BabyBear4TwoAdic::two_adic_generator(0);
    EXPECT_EQ(g, ext_one());
}

TEST(ExtFieldTwoAdic, TwoAdicGeneratorOrderPropertyAllBits) {
    // For each k in [1, 29]: g^(2^k) == 1 and g^(2^(k-1)) != 1
    BabyBear4 one = ext_one();
    for (size_t k = 1; k <= BabyBear4TwoAdic::TWO_ADICITY; ++k) {
        BabyBear4 g = BabyBear4TwoAdic::two_adic_generator(k);
        EXPECT_EQ(ext_pow2(g, k), one)
            << "ext g^(2^" << k << ") should be 1 for bits=" << k;
        EXPECT_NE(ext_pow2(g, k - 1), one)
            << "ext g^(2^" << (k-1) << ") should not be 1 for bits=" << k;
    }
}

TEST(ExtFieldTwoAdic, TwoAdicGeneratorBits27BasefieldConsistency) {
    // Extension generator for bits=27 should equal the base field generator embedded
    BabyBear4 g_ext = BabyBear4TwoAdic::two_adic_generator(27);
    BabyBear g_base = BabyBear::two_adic_generator(27);
    EXPECT_EQ(g_ext[0], g_base);
    EXPECT_EQ(g_ext[1], BabyBear());
    EXPECT_EQ(g_ext[2], BabyBear());
    EXPECT_EQ(g_ext[3], BabyBear());
}

TEST(ExtFieldTwoAdic, TwoAdicGeneratorBits28OrderExact) {
    BabyBear4 g = BabyBear4TwoAdic::two_adic_generator(28);
    // g^(2^28) == 1
    EXPECT_EQ(ext_pow2(g, 28), ext_one());
    // g^(2^27) != 1
    EXPECT_NE(ext_pow2(g, 27), ext_one());
}

TEST(ExtFieldTwoAdic, TwoAdicGeneratorBits29OrderExact) {
    BabyBear4 g = BabyBear4TwoAdic::two_adic_generator(29);
    // g^(2^29) == 1
    EXPECT_EQ(ext_pow2(g, 29), ext_one());
    // g^(2^28) != 1
    EXPECT_NE(ext_pow2(g, 28), ext_one());
}

TEST(ExtFieldTwoAdic, TwoAdicGeneratorBits29SquaredIsBits28) {
    // g29^2 should have order 2^28, so it is A primitive 2^28-th root
    BabyBear4 g29 = BabyBear4TwoAdic::two_adic_generator(29);
    BabyBear4 g29_sq = g29 * g29;
    // g29_sq^(2^28) == 1
    EXPECT_EQ(ext_pow2(g29_sq, 28), ext_one());
    // g29_sq^(2^27) != 1
    EXPECT_NE(ext_pow2(g29_sq, 27), ext_one());
}

TEST(ExtFieldTwoAdic, TwoAdicGeneratorOutOfRange) {
    EXPECT_THROW(BabyBear4TwoAdic::two_adic_generator(30), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// BabyBear powers() iterator tests
// ---------------------------------------------------------------------------

TEST(BabyBearPowers, PowersOf2) {
    BabyBear two(static_cast<uint32_t>(2u));
    BabyBear one = BabyBear::one_val();

    // Collect first 10 powers using the iterator
    auto it = two.powers().begin();
    std::array<BabyBear, 10> expected;
    BabyBear acc = one;
    for (size_t i = 0; i < 10; ++i) {
        expected[i] = acc;
        acc = acc * two;
    }

    for (size_t i = 0; i < 10; ++i, ++it) {
        EXPECT_EQ(*it, expected[i]) << "powers()[" << i << "] mismatch";
    }
}

TEST(BabyBearPowers, PowersStartAtOne) {
    BabyBear g = BabyBear::two_adic_generator(10);
    auto it = g.powers().begin();
    EXPECT_EQ(*it, BabyBear::one_val()) << "First power should be g^0 = 1";
    ++it;
    EXPECT_EQ(*it, g) << "Second power should be g^1";
    ++it;
    EXPECT_EQ(*it, g * g) << "Third power should be g^2";
}

// ---------------------------------------------------------------------------
// BinomialExtensionField<BabyBear, 4> powers() iterator tests
// ---------------------------------------------------------------------------

TEST(ExtFieldPowers, PowersOf2) {
    BabyBear4 two;
    two.coeffs[0] = BabyBear(static_cast<uint32_t>(2u));

    BabyBear4 one = ext_one();
    auto it = two.powers().begin();

    BabyBear4 acc = one;
    for (size_t i = 0; i < 8; ++i, ++it) {
        EXPECT_EQ(*it, acc) << "ext powers()[" << i << "] mismatch";
        acc = acc * two;
    }
}

TEST(ExtFieldPowers, TwoAdicTwiddlesUsagePattern) {
    // Simulate how twiddle factors are computed in DFT:
    // g = two_adic_generator(log_n), take first n/2 powers
    size_t log_n = 4;
    size_t n = 1 << log_n;  // 16
    BabyBear4 g = BabyBear4TwoAdic::two_adic_generator(log_n);

    // Collect n/2 twiddles via powers iterator
    std::vector<BabyBear4> twiddles;
    twiddles.reserve(n / 2);
    auto it = g.powers().begin();
    for (size_t i = 0; i < n / 2; ++i, ++it) {
        twiddles.push_back(*it);
    }

    // Verify first few manually
    EXPECT_EQ(twiddles[0], ext_one());   // g^0 = 1
    EXPECT_EQ(twiddles[1], g);           // g^1
    EXPECT_EQ(twiddles[2], g * g);       // g^2

    // Verify all twiddles satisfy twiddles[i] = g^i
    BabyBear4 expected = ext_one();
    for (size_t i = 0; i < n / 2; ++i) {
        EXPECT_EQ(twiddles[i], expected)
            << "twiddle[" << i << "] mismatch";
        expected = expected * g;
    }
}

// ---------------------------------------------------------------------------
// BinomialExtensionField arithmetic sanity tests
// ---------------------------------------------------------------------------

TEST(ExtFieldArithmetic, AdditionAndSubtraction) {
    BabyBear4 a, b;
    a.coeffs = {BabyBear(1u), BabyBear(2u), BabyBear(3u), BabyBear(4u)};
    b.coeffs = {BabyBear(5u), BabyBear(6u), BabyBear(7u), BabyBear(8u)};

    BabyBear4 sum = a + b;
    EXPECT_EQ(sum[0].value(), 6u);
    EXPECT_EQ(sum[1].value(), 8u);
    EXPECT_EQ(sum[2].value(), 10u);
    EXPECT_EQ(sum[3].value(), 12u);

    BabyBear4 diff = b - a;
    EXPECT_EQ(diff[0].value(), 4u);
    EXPECT_EQ(diff[1].value(), 4u);
    EXPECT_EQ(diff[2].value(), 4u);
    EXPECT_EQ(diff[3].value(), 4u);
}

TEST(ExtFieldArithmetic, MultiplicationByOne) {
    BabyBear4 a;
    a.coeffs = {BabyBear(3u), BabyBear(7u), BabyBear(11u), BabyBear(13u)};
    BabyBear4 one = ext_one();

    EXPECT_EQ(a * one, a);
    EXPECT_EQ(one * a, a);
}

TEST(ExtFieldArithmetic, MultiplicationRule) {
    // alpha^4 = 11, so [0,1,0,0]^2 = [0,0,1,0] (alpha^2)
    // and [0,0,1,0]^2 = [0,0,0,0] + W*[1,0,0,0] = [11,0,0,0] (alpha^4 = 11)
    BabyBear4 alpha;
    alpha.coeffs = {BabyBear(), BabyBear(1u), BabyBear(), BabyBear()};
    BabyBear4 alpha_sq = alpha * alpha;
    EXPECT_EQ(alpha_sq[0].value(), 0u);
    EXPECT_EQ(alpha_sq[1].value(), 0u);
    EXPECT_EQ(alpha_sq[2].value(), 1u);
    EXPECT_EQ(alpha_sq[3].value(), 0u);

    BabyBear4 alpha_4 = alpha_sq * alpha_sq;
    EXPECT_EQ(alpha_4[0].value(), 11u);  // alpha^4 = 11
    EXPECT_EQ(alpha_4[1].value(), 0u);
    EXPECT_EQ(alpha_4[2].value(), 0u);
    EXPECT_EQ(alpha_4[3].value(), 0u);
}

TEST(ExtFieldArithmetic, InverseOfOne) {
    BabyBear4 one = ext_one();
    BabyBear4 inv = one.inv();
    EXPECT_EQ(inv, one);
}

TEST(ExtFieldArithmetic, InverseOfBaseFieldElement) {
    // [3, 0, 0, 0]^(-1) = [inv(3), 0, 0, 0]
    BabyBear4 a;
    a.coeffs[0] = BabyBear(3u);
    BabyBear4 inv_a = a.inv();
    BabyBear4 product = a * inv_a;
    EXPECT_EQ(product, ext_one());
}

TEST(ExtFieldArithmetic, InverseGeneral) {
    // Verify a * a^(-1) = 1 for a non-trivial element
    BabyBear4 a;
    a.coeffs = {BabyBear(2u), BabyBear(3u), BabyBear(5u), BabyBear(7u)};
    BabyBear4 inv_a = a.inv();
    BabyBear4 product = a * inv_a;
    EXPECT_EQ(product, ext_one());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

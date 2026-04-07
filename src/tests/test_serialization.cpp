#include "../include/serialization.hpp"
#include "../include/proof_serialization.hpp"
#include "../include/goldilocks.hpp"
#include "../include/koala_bear.hpp"
#include "../include/mersenne31.hpp"
#include "../include/bn254.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cstdint>

using namespace p3_field;

class SerializationTest : public ::testing::Test {};

TEST_F(SerializationTest, GoldilocksRoundTrip) {
    std::vector<Goldilocks> elements = {
        Goldilocks::zero_val(),
        Goldilocks::one_val(),
        Goldilocks(static_cast<uint64_t>(123456789ULL)),
        Goldilocks(static_cast<uint64_t>(0xFFFFFFFF00000001ULL - 1ULL))
    };

    std::vector<uint8_t> bytes = Serialization::encode_goldilocks(elements);
    std::vector<Goldilocks> decoded = Serialization::decode_goldilocks(bytes);

    ASSERT_EQ(elements.size(), decoded.size());
    for (size_t i = 0; i < elements.size(); ++i) {
        EXPECT_EQ(elements[i].as_canonical_u64(), decoded[i].as_canonical_u64());
    }
}

TEST_F(SerializationTest, KoalaBearRoundTrip) {
    std::vector<KoalaBear> elements = {
        KoalaBear::zero_val(),
        KoalaBear::one_val(),
        KoalaBear(12345u)
    };

    std::vector<uint8_t> bytes = Serialization::encode_koalabear(elements);
    std::vector<KoalaBear> decoded = Serialization::decode_koalabear(bytes);

    ASSERT_EQ(elements.size(), decoded.size());
    for (size_t i = 0; i < elements.size(); ++i) {
        EXPECT_EQ(elements[i].as_canonical_u64(), decoded[i].as_canonical_u64());
    }
}

TEST_F(SerializationTest, Mersenne31RoundTrip) {
    std::vector<Mersenne31> elements = {
        Mersenne31::zero_val(),
        Mersenne31::one_val(),
        Mersenne31(987654321u)
    };

    std::vector<uint8_t> bytes = Serialization::encode_mersenne31(elements);
    std::vector<Mersenne31> decoded = Serialization::decode_mersenne31(bytes);

    ASSERT_EQ(elements.size(), decoded.size());
    for (size_t i = 0; i < elements.size(); ++i) {
        EXPECT_EQ(elements[i].as_canonical_u64(), decoded[i].as_canonical_u64());
    }
}

TEST_F(SerializationTest, Bn254PlaceholderTest) {
    // Placeholder for Bn254 since implementation is pending
    std::vector<uint8_t> bytes = {}; 
    ASSERT_TRUE(bytes.empty()); 
}

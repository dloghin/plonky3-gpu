#include <gtest/gtest.h>

#include "baby_bear.hpp"
#include "goldilocks.hpp"
#include "hash.hpp"
#include "keccak.hpp"
#include "serializing_hasher.hpp"
#include "multi_field_sponge.hpp"
#include "compression_from_hasher.hpp"
#include "merkle_tree_mmcs.hpp"

#include <array>
#include <cstdint>
#include <vector>

using namespace p3_field;
using namespace p3_symmetric;
using namespace p3_merkle_tree;

namespace {

struct MockByteHasher {
    std::array<uint8_t, 32> hash_iter(const std::vector<uint8_t>& data) const {
        uint8_t s = 0;
        for (uint8_t b : data) s = static_cast<uint8_t>(s + b);
        std::array<uint8_t, 32> out{};
        for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint8_t>(s + i);
        return out;
    }
};

struct MockU64Perm {
    void permute_mut(std::array<uint64_t, 4>& state) const {
        for (size_t i = 0; i < 4; ++i) state[i] += (i + 1);
    }
};

} // namespace

TEST(SymmetricExtended, SerializingHasherDeterministic) {
    SerializingHasher<BabyBear, MockByteHasher, 32, 8> h(MockByteHasher{});
    std::vector<BabyBear> input = {BabyBear(1u), BabyBear(2u), BabyBear(3u)};
    const auto d1 = h.hash_iter(input);
    const auto d2 = h.hash_iter(input);
    EXPECT_EQ(d1, d2);
}

TEST(SymmetricExtended, SerializingHasherKeccakInnerWorks) {
    SerializingHasher<BabyBear, Keccak256Hash, 32, 8> h(Keccak256Hash{});
    std::vector<BabyBear> input = {BabyBear(10u), BabyBear(20u), BabyBear(30u), BabyBear(40u)};
    const auto digest = h.hash_iter(input);
    bool any_non_zero = false;
    for (const auto& x : digest) {
        if (x != BabyBear::zero_val()) { any_non_zero = true; break; }
    }
    EXPECT_TRUE(any_non_zero);
}

TEST(SymmetricExtended, MultiFieldSpongePacksPairsIntoU64Lanes) {
    MultiField32PaddingFreeSponge<BabyBear, MockU64Perm, 4, 2, 4> sponge(MockU64Perm{});
    std::vector<BabyBear> input = {BabyBear(1u), BabyBear(2u), BabyBear(3u), BabyBear(4u)};
    const auto out = sponge.hash_iter(input);

    // packed lanes:
    // lane0 = (2<<32)|1, lane1=(4<<32)|3, then +1/+2 from perm
    EXPECT_EQ(out[0], BabyBear(static_cast<uint32_t>(2u)));
    EXPECT_EQ(out[1], BabyBear(static_cast<uint32_t>(2u)));
    EXPECT_EQ(out[2], BabyBear(static_cast<uint32_t>(5u)));
    EXPECT_EQ(out[3], BabyBear(static_cast<uint32_t>(4u)));
}

TEST(SymmetricExtended, CompressionFunctionFromHasherCompressesChunks) {
    using H = SerializingHasher<BabyBear, MockByteHasher, 32, 8>;
    CompressionFunctionFromHasher<H, BabyBear, 2, 8> c(H(MockByteHasher{}));
    std::array<std::array<BabyBear, 8>, 2> in{};
    for (size_t i = 0; i < 8; ++i) {
        in[0][i] = BabyBear(static_cast<uint32_t>(i + 1));
        in[1][i] = BabyBear(static_cast<uint32_t>(i + 11));
    }
    const auto out = c.compress(in);
    bool any_non_zero = false;
    for (const auto& x : out) {
        if (x != BabyBear::zero_val()) { any_non_zero = true; break; }
    }
    EXPECT_TRUE(any_non_zero);
}

TEST(SymmetricExtended, IntegratesWithMerkleTreeMmcs) {
    using H = SerializingHasher<BabyBear, MockByteHasher, 32, 8>;
    using C = CompressionFunctionFromHasher<H, BabyBear, 2, 8>;
    MerkleTreeMmcs<BabyBear, BabyBear, H, C, 8> mmcs(H(MockByteHasher{}), C(H(MockByteHasher{})), 0);

    std::vector<BabyBear> vals(16);
    for (size_t i = 0; i < vals.size(); ++i) vals[i] = BabyBear(static_cast<uint32_t>(i + 1));
    std::vector<RowMajorMatrix<BabyBear>> mats;
    mats.emplace_back(vals, 2); // 8x2 matrix

    auto [commitment, prover_data] = mmcs.commit(std::move(mats));
    auto opening = mmcs.open_batch(0, prover_data);
    EXPECT_FALSE(commitment.cap.empty());
    EXPECT_FALSE(opening.opened_values.empty());
}

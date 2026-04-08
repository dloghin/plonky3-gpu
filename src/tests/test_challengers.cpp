#include <gtest/gtest.h>

#include "baby_bear.hpp"
#include "challenger_traits.hpp"
#include "hash_challenger.hpp"
#include "serializing_challenger.hpp"
#include "multi_field_challenger.hpp"

#include <array>
#include <cstdint>
#include <vector>

using namespace p3_field;
using namespace p3_challenger;

namespace {

struct SumLenHasher {
    std::array<BabyBear, 2> hash_iter(const std::vector<BabyBear>& input) const {
        BabyBear sum = BabyBear::zero_val();
        for (const auto& x : input) sum += x;
        return {sum, BabyBear(static_cast<uint32_t>(input.size()))};
    }
};

class ByteQueueChallenger {
public:
    explicit ByteQueueChallenger(std::vector<uint8_t> seed = {}) : queue_(std::move(seed)) {}
    void observe(uint8_t b) { observed_.push_back(b); }
    uint8_t sample() {
        if (idx_ >= queue_.size()) {
            queue_.push_back(static_cast<uint8_t>((idx_ * 17u + 13u) & 0xffu));
        }
        return queue_[idx_++];
    }

    const std::vector<uint8_t>& observed() const { return observed_; }

private:
    std::vector<uint8_t> queue_;
    size_t idx_ = 0;
    std::vector<uint8_t> observed_;
};

class U64QueueChallenger {
public:
    explicit U64QueueChallenger(std::vector<uint64_t> seed = {}) : queue_(std::move(seed)) {}
    void observe(uint64_t x) { observed_.push_back(x); }
    uint64_t sample() {
        if (idx_ >= queue_.size()) queue_.push_back(static_cast<uint64_t>(0x1000 + idx_));
        return queue_[idx_++];
    }
    const std::vector<uint64_t>& observed() const { return observed_; }

private:
    std::vector<uint64_t> queue_;
    size_t idx_ = 0;
    std::vector<uint64_t> observed_;
};

} // namespace

TEST(HashChallenger, DeterministicAndReproducible) {
    HashChallenger<BabyBear, SumLenHasher, 2> c1(SumLenHasher{}, {BabyBear(1u), BabyBear(2u)});
    HashChallenger<BabyBear, SumLenHasher, 2> c2(SumLenHasher{}, {BabyBear(1u), BabyBear(2u)});

    c1.observe(BabyBear(7u));
    c2.observe(BabyBear(7u));

    EXPECT_EQ(c1.sample(), c2.sample());
    EXPECT_EQ(c1.sample(), c2.sample());
}

TEST(HashChallenger, RustReferenceLikeSequence) {
    // Mirrors the simple rust unit-test hasher behavior (sum, len).
    HashChallenger<BabyBear, SumLenHasher, 2> c(SumLenHasher{},
        {BabyBear(1u), BabyBear(2u), BabyBear(3u), BabyBear(4u)});

    // Flush on first sample -> [sum=10, len=4], sample pops len then sum.
    EXPECT_EQ(c.sample(), BabyBear(4u));
    EXPECT_EQ(c.sample(), BabyBear(10u));
}

TEST(SerializingChallenger, ObserveSerializesBytesLittleEndian) {
    SerializingChallenger<BabyBear, ByteQueueChallenger> c(ByteQueueChallenger{});
    c.observe(BabyBear(0x01020304u));

    const auto& obs = c.inner().observed();
    ASSERT_EQ(obs.size(), 4u);
    EXPECT_EQ(obs[0], 0x04u);
    EXPECT_EQ(obs[1], 0x03u);
    EXPECT_EQ(obs[2], 0x02u);
    EXPECT_EQ(obs[3], 0x01u);
}

TEST(SerializingChallenger, SampleRoundTripFromBytes) {
    ByteQueueChallenger inner({0x78u, 0x56u, 0x34u, 0x12u});
    SerializingChallenger<BabyBear, ByteQueueChallenger> c(std::move(inner));
    const BabyBear v = c.sample();
    EXPECT_EQ(v, BabyBear(0x12345678u % BabyBear::PRIME));
}

TEST(MultiFieldChallenger, PacksAndUnpacksDeterministically) {
    U64QueueChallenger inner({0x0004000300020001ULL});
    MultiFieldChallenger<BabyBear, U64QueueChallenger, 4> c(std::move(inner));

    // unpack order is sampled from back of chunk: [1,2,3,4] => returns 4,3,2,1
    EXPECT_EQ(c.sample(), BabyBear(4u));
    EXPECT_EQ(c.sample(), BabyBear(3u));
    EXPECT_EQ(c.sample(), BabyBear(2u));
    EXPECT_EQ(c.sample(), BabyBear(1u));
}

TEST(MultiFieldChallenger, ObserveSameSequenceProducesSameChallenges) {
    MultiFieldChallenger<BabyBear, U64QueueChallenger, 4> c1(U64QueueChallenger({0x0008000700060005ULL}));
    MultiFieldChallenger<BabyBear, U64QueueChallenger, 4> c2(U64QueueChallenger({0x0008000700060005ULL}));

    c1.observe(BabyBear(11u));
    c1.observe(BabyBear(12u));
    c2.observe(BabyBear(11u));
    c2.observe(BabyBear(12u));

    EXPECT_EQ(c1.sample(), c2.sample());
    EXPECT_EQ(c1.sample(), c2.sample());
}

TEST(ChallengerTraits, FieldChallengerTraitChecks) {
    using HC = HashChallenger<BabyBear, SumLenHasher, 2>;
    static_assert(can_observe<HC, BabyBear>::value, "HashChallenger must observe field elements");
    static_assert(can_sample<HC, BabyBear>::value, "HashChallenger must sample field elements");
}

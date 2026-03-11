#include <gtest/gtest.h>
#include "p3_util/util.hpp"

using namespace p3_util;

// ---------------------------------------------------------------------------
// log2_strict_usize
// ---------------------------------------------------------------------------

TEST(Log2StrictTest, PowersOfTwo) {
    EXPECT_EQ(log2_strict_usize(1),    0u);
    EXPECT_EQ(log2_strict_usize(2),    1u);
    EXPECT_EQ(log2_strict_usize(4),    2u);
    EXPECT_EQ(log2_strict_usize(8),    3u);
    EXPECT_EQ(log2_strict_usize(16),   4u);
    EXPECT_EQ(log2_strict_usize(1024), 10u);
    EXPECT_EQ(log2_strict_usize(1u << 20), 20u);
}

TEST(Log2StrictTest, NonPowerThrows) {
    EXPECT_THROW(log2_strict_usize(0), std::invalid_argument);
    EXPECT_THROW(log2_strict_usize(3), std::invalid_argument);
    EXPECT_THROW(log2_strict_usize(5), std::invalid_argument);
    EXPECT_THROW(log2_strict_usize(7), std::invalid_argument);
    EXPECT_THROW(log2_strict_usize(100), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// log2_ceil_usize
// ---------------------------------------------------------------------------

TEST(Log2CeilTest, SmallValues) {
    EXPECT_EQ(log2_ceil_usize(0), 0u);
    EXPECT_EQ(log2_ceil_usize(1), 0u);
    EXPECT_EQ(log2_ceil_usize(2), 1u);
    EXPECT_EQ(log2_ceil_usize(3), 2u);
    EXPECT_EQ(log2_ceil_usize(4), 2u);
    EXPECT_EQ(log2_ceil_usize(5), 3u);
    EXPECT_EQ(log2_ceil_usize(8), 3u);
    EXPECT_EQ(log2_ceil_usize(9), 4u);
    EXPECT_EQ(log2_ceil_usize(16), 4u);
    EXPECT_EQ(log2_ceil_usize(17), 5u);
    EXPECT_EQ(log2_ceil_usize(1024), 10u);
    EXPECT_EQ(log2_ceil_usize(1025), 11u);
}

// ---------------------------------------------------------------------------
// reverse_bits_len
// ---------------------------------------------------------------------------

TEST(ReverseBitsLenTest, RustDocExample) {
    // From task description: reverse_bits_len(0b01011, 5) == 0b11010
    EXPECT_EQ(reverse_bits_len(0b01011u, 5u), 0b11010u);
}

TEST(ReverseBitsLenTest, ThreeBits) {
    EXPECT_EQ(reverse_bits_len(0b000u, 3u), 0b000u);
    EXPECT_EQ(reverse_bits_len(0b001u, 3u), 0b100u);
    EXPECT_EQ(reverse_bits_len(0b010u, 3u), 0b010u);
    EXPECT_EQ(reverse_bits_len(0b011u, 3u), 0b110u);
    EXPECT_EQ(reverse_bits_len(0b100u, 3u), 0b001u);
    EXPECT_EQ(reverse_bits_len(0b101u, 3u), 0b101u);
    EXPECT_EQ(reverse_bits_len(0b110u, 3u), 0b011u);
    EXPECT_EQ(reverse_bits_len(0b111u, 3u), 0b111u);
}

TEST(ReverseBitsLenTest, OneBit) {
    EXPECT_EQ(reverse_bits_len(0u, 1u), 0u);
    EXPECT_EQ(reverse_bits_len(1u, 1u), 1u);
}

TEST(ReverseBitsLenTest, ZeroBits) {
    EXPECT_EQ(reverse_bits_len(0u, 0u), 0u);
    EXPECT_EQ(reverse_bits_len(42u, 0u), 0u);
}

// ---------------------------------------------------------------------------
// reverse_slice_index_bits -- small (log_n < 4)
// ---------------------------------------------------------------------------

TEST(ReverseSliceBitsTest, Size4) {
    // indices: 0b00=0, 0b01=1, 0b10=2, 0b11=3
    // reversed: 0b00=0, 0b10=2, 0b01=1, 0b11=3
    std::vector<int> v = {10, 20, 30, 40};
    reverse_slice_index_bits(v);
    EXPECT_EQ(v, (std::vector<int>{10, 30, 20, 40}));
}

TEST(ReverseSliceBitsTest, Size8) {
    std::vector<int> v = {0, 1, 2, 3, 4, 5, 6, 7};
    reverse_slice_index_bits(v);
    // bit-reversal of 0..7 with 3 bits:
    // 000->000=0, 001->100=4, 010->010=2, 011->110=6,
    // 100->001=1, 101->101=5, 110->011=3, 111->111=7
    EXPECT_EQ(v, (std::vector<int>{0, 4, 2, 6, 1, 5, 3, 7}));
}

TEST(ReverseSliceBitsTest, Size1) {
    std::vector<int> v = {42};
    reverse_slice_index_bits(v);
    EXPECT_EQ(v, (std::vector<int>{42}));
}

TEST(ReverseSliceBitsTest, Empty) {
    std::vector<int> v;
    EXPECT_NO_THROW(reverse_slice_index_bits(v));
    EXPECT_TRUE(v.empty());
}

TEST(ReverseSliceBitsTest, NonPowerOfTwoThrows) {
    std::vector<int> v = {1, 2, 3};
    EXPECT_THROW(reverse_slice_index_bits(v), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// reverse_slice_index_bits -- large (log_n >= 4, uses block algorithm)
// ---------------------------------------------------------------------------

TEST(ReverseSliceBitsTest, Size256) {
    // Build reference: apply simple swap loop
    std::vector<int> ref(256);
    for (int i = 0; i < 256; ++i) ref[i] = i;
    {
        // naive reference permutation
        for (size_t i = 0; i < 256; ++i) {
            size_t j = reverse_bits_len(i, 8);
            if (i < j) std::swap(ref[i], ref[j]);
        }
    }

    std::vector<int> v(256);
    for (int i = 0; i < 256; ++i) v[i] = i;
    reverse_slice_index_bits(v);

    EXPECT_EQ(v, ref);
}

TEST(ReverseSliceBitsTest, Size1024) {
    std::vector<int> ref(1024);
    for (int i = 0; i < 1024; ++i) ref[i] = i;
    for (size_t i = 0; i < 1024; ++i) {
        size_t j = reverse_bits_len(i, 10);
        if (i < j) std::swap(ref[i], ref[j]);
    }

    std::vector<int> v(1024);
    for (int i = 0; i < 1024; ++i) v[i] = i;
    reverse_slice_index_bits(v);

    EXPECT_EQ(v, ref);
}

// ---------------------------------------------------------------------------
// apply_to_chunks
// ---------------------------------------------------------------------------

TEST(ApplyToChunksTest, ExactMultiple) {
    std::vector<int> input = {1, 2, 3, 4, 5, 6};
    std::vector<std::vector<int>> chunks;
    apply_to_chunks<2>(input, [&](const int* ptr, size_t len) {
        chunks.push_back(std::vector<int>(ptr, ptr + len));
    });
    ASSERT_EQ(chunks.size(), 3u);
    EXPECT_EQ(chunks[0], (std::vector<int>{1, 2}));
    EXPECT_EQ(chunks[1], (std::vector<int>{3, 4}));
    EXPECT_EQ(chunks[2], (std::vector<int>{5, 6}));
}

TEST(ApplyToChunksTest, PartialLastChunk) {
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<std::vector<int>> chunks;
    apply_to_chunks<3>(input, [&](const int* ptr, size_t len) {
        chunks.push_back(std::vector<int>(ptr, ptr + len));
    });
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0], (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(chunks[1], (std::vector<int>{4, 5}));
}

TEST(ApplyToChunksTest, Empty) {
    std::vector<int> input;
    int call_count = 0;
    apply_to_chunks<4>(input, [&](const int*, size_t) { ++call_count; });
    EXPECT_EQ(call_count, 0);
}

// ---------------------------------------------------------------------------
// iter_array_chunks_padded
// ---------------------------------------------------------------------------

TEST(IterArrayChunksPaddedTest, ExactMultiple) {
    std::vector<int> input = {1, 2, 3, 4};
    auto result = iter_array_chunks_padded<2>(input, 0);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], (std::array<int, 2>{1, 2}));
    EXPECT_EQ(result[1], (std::array<int, 2>{3, 4}));
}

TEST(IterArrayChunksPaddedTest, PaddedLastChunk) {
    std::vector<int> input = {1, 2, 3};
    auto result = iter_array_chunks_padded<2>(input, -1);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], (std::array<int, 2>{1, 2}));
    EXPECT_EQ(result[1], (std::array<int, 2>{3, -1}));
}

TEST(IterArrayChunksPaddedTest, SingleChunkExact) {
    std::vector<int> input = {7, 8, 9};
    auto result = iter_array_chunks_padded<3>(input, 0);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], (std::array<int, 3>{7, 8, 9}));
}

TEST(IterArrayChunksPaddedTest, Empty) {
    std::vector<int> input;
    auto result = iter_array_chunks_padded<4>(input, 0);
    EXPECT_TRUE(result.empty());
}

/**
 * @file test_keccak.cpp
 * @brief Google Test suite for Keccak-f[1600] and Keccak-256.
 *
 * Keccak-f and PaddingFreeSponge vectors match p3-keccak / tiny-keccak (Rust).
 * Keccak-256 byte digests match Keccak::v256() / PyCryptodome Keccak-256.
 */

#include <gtest/gtest.h>

#include "keccak.hpp"
#include "padding_free_sponge.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace p3_symmetric;

namespace {

void expect_digest_hex(const Keccak256Hash::Digest& d, const char* hex64) {
    ASSERT_EQ(std::strlen(hex64), 64u) << "expected 32-byte hex string";
    for (size_t i = 0; i < 32; ++i) {
        const char* p = hex64 + i * 2;
        const unsigned hi = p[0] >= 'a' ? 10 + (p[0] - 'a') : (p[0] >= 'A' ? 10 + (p[0] - 'A') : (p[0] - '0'));
        const unsigned lo = p[1] >= 'a' ? 10 + (p[1] - 'a') : (p[1] >= 'A' ? 10 + (p[1] - 'A') : (p[1] - '0'));
        const uint8_t expected = static_cast<uint8_t>((hi << 4) | lo);
        EXPECT_EQ(d[i], expected) << "byte " << i;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Keccak-f[1600]
// ---------------------------------------------------------------------------

TEST(KeccakF, ZeroStatePermutation) {
    std::array<uint64_t, KECCAK_STATE_LANES> st{};
    KeccakF f;
    f.permute_mut(st);
    EXPECT_EQ(st[0], 0xf1258f7940e1dde7ULL);
    EXPECT_EQ(st[1], 0x84d5ccf933c0478aULL);
    EXPECT_EQ(st[2], 0xd598261ea65aa9eeULL);
    EXPECT_EQ(st[3], 0xbd1547306f80494dULL);
}

TEST(KeccakF, Deterministic) {
    std::array<uint64_t, KECCAK_STATE_LANES> a{};
    std::array<uint64_t, KECCAK_STATE_LANES> b{};
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<uint64_t>(i * i + 3 * i + 7);
        b[i] = a[i];
    }
    KeccakF f;
    f.permute_mut(a);
    f.permute_mut(b);
    EXPECT_EQ(a, b);
}

// ---------------------------------------------------------------------------
// Keccak-256 (Ethereum / original Keccak)
// ---------------------------------------------------------------------------

TEST(Keccak256Hash, Empty) {
    Keccak256Hash h;
    const auto d = h.hash_iter(std::vector<uint8_t>{});
    expect_digest_hex(d, "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
}

TEST(Keccak256Hash, Abc) {
    Keccak256Hash h;
    const std::vector<uint8_t> msg = {'a', 'b', 'c'};
    const auto d = h.hash_iter(msg);
    expect_digest_hex(d, "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
}

TEST(Keccak256Hash, PointerLenOverload) {
    Keccak256Hash h;
    const uint8_t msg[] = {'a', 'b', 'c'};
    const auto d = h.hash_iter(msg, sizeof(msg));
    expect_digest_hex(d, "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
}

TEST(Keccak256Hash, TwoHundredBytes) {
    Keccak256Hash h;
    std::vector<uint8_t> msg(200, 0xcc);
    const auto d = h.hash_iter(msg);
    expect_digest_hex(d, "3b8a384f5faa725c47bc35682f704f6226e0db6ca583db1a2019dd0817ff6736");
}

TEST(Keccak256Hash, HashIterSlices) {
    Keccak256Hash h;
    const std::vector<std::vector<uint8_t>> slices = {
        {0x01, 0x02},
        {0x03, 0x04, 0x05},
    };
    const auto d_slices = h.hash_iter_slices(slices);
    std::vector<uint8_t> flat = {0x01, 0x02, 0x03, 0x04, 0x05};
    const auto d_flat = h.hash_iter(flat);
    EXPECT_EQ(d_slices, d_flat);
}

// ---------------------------------------------------------------------------
// PaddingFreeSponge<KeccakF, u64, 25, 17, 4> (Plonky3 Merkle / STARK config)
// ---------------------------------------------------------------------------

TEST(KeccakPaddingFreeSponge, EmptyInput) {
    PaddingFreeSponge<KeccakF, uint64_t, KECCAK_STATE_LANES, 17, 4> sponge{KeccakF{}};
    const std::vector<uint64_t> input;
    const auto d = sponge.hash_iter(input);
    EXPECT_EQ(d[0], 0u);
    EXPECT_EQ(d[1], 0u);
    EXPECT_EQ(d[2], 0u);
    EXPECT_EQ(d[3], 0u);
}

TEST(KeccakPaddingFreeSponge, SeventeenOnes) {
    PaddingFreeSponge<KeccakF, uint64_t, KECCAK_STATE_LANES, 17, 4> sponge{KeccakF{}};
    std::vector<uint64_t> input(17, 1u);
    const auto d = sponge.hash_iter(input);
    EXPECT_EQ(d[0], 0x774a62b78b216834ULL);
    EXPECT_EQ(d[1], 0x7d20f65c7acfed24ULL);
    EXPECT_EQ(d[2], 0xf0189c414fc06e86ULL);
    EXPECT_EQ(d[3], 0x179210117072bd81ULL);
}

/**
 * @file test_sha256.cpp
 * @brief Google Test suite for SHA-256 hash and compression.
 */

#include <gtest/gtest.h>

#include "dense_matrix.hpp"
#include "merkle_tree_mmcs.hpp"
#include "sha256.hpp"

#include <array>
#include <cstdint>
#include <vector>

using namespace p3_symmetric;
using namespace p3_matrix;
using namespace p3_merkle_tree;

namespace {

uint8_t from_hex(char c) {
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(10 + (c - 'a'));
    if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(10 + (c - 'A'));
    return 0;
}

std::array<uint8_t, 32> digest_from_hex(const char* hex64) {
    std::array<uint8_t, 32> out{};
    for (size_t i = 0; i < 32; ++i) {
        out[i] = static_cast<uint8_t>((from_hex(hex64[2 * i]) << 4) | from_hex(hex64[2 * i + 1]));
    }
    return out;
}

void expect_digest_hex(const std::array<uint8_t, 32>& got, const char* expected_hex64) {
    EXPECT_EQ(got, digest_from_hex(expected_hex64));
}

} // namespace

TEST(Sha256, EmptyStringVector) {
    Sha256 sha;
    const auto d = sha.hash_iter(std::vector<uint8_t>{});
    expect_digest_hex(d, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(Sha256, Abc) {
    Sha256 sha;
    const std::vector<uint8_t> msg = {'a', 'b', 'c'};
    const auto d = sha.hash_iter(msg);
    expect_digest_hex(d, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(Sha256, PointerLenOverload) {
    Sha256 sha;
    const uint8_t msg[] = {'a', 'b', 'c'};
    const auto d = sha.hash_iter(msg, sizeof(msg));
    expect_digest_hex(d, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(Sha256, HelloWorld) {
    Sha256 sha;
    const uint8_t msg[] = {'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'};
    const auto d = sha.hash_iter(msg, sizeof(msg));
    expect_digest_hex(d, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
}

TEST(Sha256, HashIterSlicesEqualsFlat) {
    Sha256 sha;
    const std::vector<std::vector<uint8_t>> slices = {
        {'a', 'b'},
        {'c'},
    };
    const auto from_slices = sha.hash_iter_slices(slices);
    const std::vector<uint8_t> flat = {'a', 'b', 'c'};
    const auto from_flat = sha.hash_iter(flat);
    EXPECT_EQ(from_slices, from_flat);
}

TEST(Sha256Compress, MatchesSha256ForPaddedSingleBlock) {
    // Same check as Rust reference test:
    // compress([left, right]) equals sha256(left) when right is the SHA-256
    // padding tail for a 32-byte message.
    const std::array<uint8_t, 32> left{};
    std::array<uint8_t, 32> right{};
    right[0] = 0x80u;
    right[30] = 0x01u; // bit length = 256 = 0x000...0100

    const Sha256 sha;
    const Sha256Compress compress;
    const auto expected = sha.hash_iter(left.data(), left.size());
    const auto got = compress.compress({left, right});
    EXPECT_EQ(got, expected);
}

TEST(Sha256Mmcs, CommitOpenVerifySingleMatrix) {
    using Digest = std::array<uint8_t, 32>;
    using Mmcs = MerkleTreeMmcs<uint8_t, uint8_t, Sha256, Sha256Compress, 32>;

    std::vector<uint8_t> vals;
    vals.reserve(8 * 4);
    for (uint16_t i = 0; i < 32; ++i) {
        vals.push_back(static_cast<uint8_t>(i));
    }
    RowMajorMatrix<uint8_t> mat(std::move(vals), 4);
    const std::vector<Dimensions> dims = {mat.dimensions()};

    Mmcs mmcs(Sha256{}, Sha256Compress{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat)});
    ASSERT_EQ(cap.cap.size(), 1u);
    ASSERT_EQ(cap.cap[0].size(), Digest{}.size());

    for (size_t i = 0; i < 8; ++i) {
        auto opening = mmcs.open_batch(i, tree);
        EXPECT_TRUE(mmcs.verify_batch(cap, dims, i, opening));
    }
}

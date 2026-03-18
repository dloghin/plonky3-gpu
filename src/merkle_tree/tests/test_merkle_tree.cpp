/**
 * @file test_merkle_tree.cpp
 * @brief Google Test suite for MerkleTree, MerkleCap, and MerkleTreeMmcs.
 *
 * Tests cover:
 *  1. MerkleCap construction and equality.
 *  2. Single-matrix build and root extraction.
 *  3. Multi-matrix (mixed height) build.
 *  4. open_batch returns correct opened values.
 *  5. verify_batch accepts valid proofs.
 *  6. verify_batch rejects tampered proofs.
 *  7. Round-trip for all valid indices.
 *  8. cap_height > 0 produces correct-size cap.
 *
 * Mock hash/compress functions provide deterministic, easily traceable output.
 */

#include <gtest/gtest.h>

#include "merkle_cap.hpp"
#include "merkle_tree.hpp"
#include "merkle_tree_mmcs.hpp"

#include "padding_free_sponge.hpp"
#include "truncated_permutation.hpp"
#include "poseidon2.hpp"
#include "baby_bear.hpp"
#include "dense_matrix.hpp"

#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace p3_merkle_tree;
using namespace p3_matrix;
using namespace p3_field;
using namespace p3_symmetric;
using namespace poseidon2;

// ---------------------------------------------------------------------------
// Mock primitives
// ---------------------------------------------------------------------------

/**
 * A simple mock hasher: XOR all field element values into the first slot,
 * leaving the rest zero.  Good enough for structural tests.
 */
template <size_t DIGEST_ELEMS = 4>
struct MockHasher {
    using F = uint32_t;
    using W = uint32_t;

    std::array<W, DIGEST_ELEMS> hash_iter_slices(
        const std::vector<std::vector<F>>& slices) const
    {
        std::array<W, DIGEST_ELEMS> out{};
        for (const auto& row : slices) {
            for (auto v : row) out[0] ^= v;
        }
        // Put a second marker so digests are not trivially equal for different inputs
        out[1] = out[0] + 1;
        return out;
    }
};

/**
 * A non-commutative mock compressor:
 *   out[i] = left[i] * 2 + right[i] + (i + 1)
 *
 * Must be non-commutative so that swapping left/right children produces a
 * different output (required for correct Merkle position verification).
 */
template <size_t DIGEST_ELEMS = 4>
struct MockCompressor {
    using W = uint32_t;

    std::array<W, DIGEST_ELEMS> compress(
        const std::array<std::array<W, DIGEST_ELEMS>, 2>& inputs) const
    {
        std::array<W, DIGEST_ELEMS> out{};
        for (size_t i = 0; i < DIGEST_ELEMS; ++i) {
            out[i] = inputs[0][i] * 2u + inputs[1][i] + static_cast<W>(i + 1u);
        }
        return out;
    }
};

// Convenience aliases
using F32  = uint32_t;
using W32  = uint32_t;
constexpr size_t DE = 4;  // DIGEST_ELEMS for mock tests

using MockMmcs = MerkleTreeMmcs<F32, W32, MockHasher<DE>, MockCompressor<DE>, DE>;
using MockTree = MerkleTree<F32, W32, DE>;
using MockCap  = MerkleCap<W32, DE>;
using MockOpen = BatchOpening<F32, W32, DE>;

// ---------------------------------------------------------------------------
// Helper: build a simple RowMajorMatrix<uint32_t> of given size
// ---------------------------------------------------------------------------
static RowMajorMatrix<F32> make_matrix(size_t rows, size_t cols, uint32_t start = 1) {
    std::vector<F32> vals(rows * cols);
    std::iota(vals.begin(), vals.end(), start);
    return RowMajorMatrix<F32>(std::move(vals), cols);
}

// ---------------------------------------------------------------------------
// MerkleCap tests
// ---------------------------------------------------------------------------

TEST(MerkleCap, DefaultConstruction) {
    MockCap cap;
    EXPECT_TRUE(cap.cap.empty());
}

TEST(MerkleCap, Equality) {
    std::vector<std::array<W32, DE>> data = { {1, 2, 3, 4} };
    MockCap a(data), b(data);
    EXPECT_EQ(a, b);
    b.cap[0][0] = 99;
    EXPECT_NE(a, b);
}

// ---------------------------------------------------------------------------
// Single-matrix build tests
// ---------------------------------------------------------------------------

TEST(MerkleTree, SingleMatrixCapSize) {
    // 8 rows, 3 cols; cap_height=0 => single root
    auto mat = make_matrix(8, 3);
    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    EXPECT_EQ(cap.cap.size(), 1u);
    EXPECT_EQ(tree.digest_layers.size(), 4u);  // log2(8)+1 layers
    EXPECT_EQ(tree.digest_layers[0].size(), 8u);
    EXPECT_EQ(tree.digest_layers[1].size(), 4u);
    EXPECT_EQ(tree.digest_layers[2].size(), 2u);
    EXPECT_EQ(tree.digest_layers[3].size(), 1u);
}

TEST(MerkleTree, SingleMatrixCapHeight2) {
    // 8 rows; cap_height=2 => cap of size 4
    auto mat = make_matrix(8, 2);
    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 2);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    EXPECT_EQ(cap.cap.size(), 4u);
    // digest_layers: [8, 4] — one compression step; cap is the second layer
    EXPECT_EQ(tree.digest_layers.size(), 2u);
    EXPECT_EQ(tree.digest_layers[0].size(), 8u);
    EXPECT_EQ(tree.digest_layers[1].size(), 4u);
}

TEST(MerkleTree, SingleRowMatrix) {
    // height=1 with cap_height=0 => cap is the single leaf hash
    auto mat = make_matrix(1, 5);
    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    EXPECT_EQ(cap.cap.size(), 1u);
    EXPECT_EQ(tree.digest_layers.size(), 1u);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

TEST(MerkleTree, Deterministic) {
    auto mat1 = make_matrix(4, 3, 1);
    auto mat2 = make_matrix(4, 3, 1);

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap1, tree1] = mmcs.commit({std::move(mat1)});
    auto [cap2, tree2] = mmcs.commit({std::move(mat2)});

    EXPECT_EQ(cap1, cap2);
}

// ---------------------------------------------------------------------------
// Multi-matrix (mixed height) tests
// ---------------------------------------------------------------------------

TEST(MerkleTree, MultiMatrixSameHeight) {
    auto mat1 = make_matrix(4, 2, 1);
    auto mat2 = make_matrix(4, 3, 100);

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat1), std::move(mat2)});

    EXPECT_EQ(cap.cap.size(), 1u);
    EXPECT_EQ(tree.digest_layers[0].size(), 4u);
}

TEST(MerkleTree, MultiMatrixMixedHeight) {
    // Tall matrix: 8 rows, 2 cols
    // Short matrix: 4 rows, 3 cols
    auto tall  = make_matrix(8, 2, 1);
    auto short_ = make_matrix(4, 3, 100);

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(tall), std::move(short_)});

    EXPECT_EQ(cap.cap.size(), 1u);
    // Leaf layer has 8 nodes (tallest matrix height).
    EXPECT_EQ(tree.digest_layers[0].size(), 8u);
    // After one compression: 4 nodes; here the short matrix is injected.
    EXPECT_EQ(tree.digest_layers[1].size(), 4u);
    // Final (root) layer.
    EXPECT_EQ(tree.digest_layers.back().size(), 1u);
}

TEST(MerkleTree, MultiMatrixMixedHeightRootDiffersFromSingleMatrix) {
    // Committing to two matrices should differ from committing to just one.
    auto tall1  = make_matrix(8, 2, 1);
    auto tall2  = make_matrix(8, 2, 1);
    auto short1 = make_matrix(4, 3, 100);

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap_single, t1] = mmcs.commit({std::move(tall1)});
    auto [cap_mixed,  t2] = mmcs.commit({std::move(tall2), std::move(short1)});

    // The roots should differ (the short matrix contributes additional data).
    EXPECT_NE(cap_single, cap_mixed);
}

// ---------------------------------------------------------------------------
// open_batch tests
// ---------------------------------------------------------------------------

TEST(MerkleTree, OpenBatchValuesCorrect) {
    auto mat = make_matrix(4, 3, 1);
    // Make a copy for reference
    std::vector<F32> expected_row0 = mat.row(0);
    std::vector<F32> expected_row2 = mat.row(2);

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    auto open0 = mmcs.open_batch(0, tree);
    EXPECT_EQ(open0.opened_values.size(), 1u);
    EXPECT_EQ(open0.opened_values[0], expected_row0);

    auto open2 = mmcs.open_batch(2, tree);
    EXPECT_EQ(open2.opened_values[0], expected_row2);
}

TEST(MerkleTree, OpenBatchProofLength) {
    // 8 rows, cap_height=0 => 3 proof elements (log2(8))
    auto mat = make_matrix(8, 3);
    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    for (size_t i = 0; i < 8; ++i) {
        auto opening = mmcs.open_batch(i, tree);
        EXPECT_EQ(opening.proof.size(), 3u)
            << "Expected 3 proof elements for index " << i;
    }
}

TEST(MerkleTree, OpenBatchProofLengthCapHeight2) {
    // 8 rows, cap_height=2 => 1 proof element (log2(8)-2=1)
    auto mat = make_matrix(8, 3);
    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 2);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    auto opening = mmcs.open_batch(0, tree);
    EXPECT_EQ(opening.proof.size(), 1u);
}

// ---------------------------------------------------------------------------
// verify_batch tests
// ---------------------------------------------------------------------------

TEST(MerkleTree, VerifyValidProof) {
    auto mat = make_matrix(4, 3);
    std::vector<Dimensions> dims = { mat.dimensions() };

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    for (size_t i = 0; i < 4; ++i) {
        auto opening = mmcs.open_batch(i, tree);
        EXPECT_TRUE(mmcs.verify_batch(cap, dims, i, opening))
            << "Valid proof should verify for index " << i;
    }
}

TEST(MerkleTree, VerifyAllIndicesSingleMatrix) {
    auto mat = make_matrix(8, 5);
    std::vector<Dimensions> dims = { mat.dimensions() };

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    for (size_t i = 0; i < 8; ++i) {
        auto opening = mmcs.open_batch(i, tree);
        EXPECT_TRUE(mmcs.verify_batch(cap, dims, i, opening))
            << "Valid proof should verify for index " << i;
    }
}

TEST(MerkleTree, VerifyTamperedLeafFails) {
    auto mat = make_matrix(4, 3);
    std::vector<Dimensions> dims = { mat.dimensions() };

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    auto opening = mmcs.open_batch(0, tree);
    // Tamper with the opened value
    opening.opened_values[0][0] ^= 0xDEADBEEF;

    EXPECT_FALSE(mmcs.verify_batch(cap, dims, 0, opening))
        << "Tampered leaf should fail verification";
}

TEST(MerkleTree, VerifyTamperedSiblingFails) {
    auto mat = make_matrix(4, 3);
    std::vector<Dimensions> dims = { mat.dimensions() };

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    auto opening = mmcs.open_batch(1, tree);
    // Tamper with a sibling digest in the proof
    opening.proof[0][0] ^= 0xDEADBEEF;

    EXPECT_FALSE(mmcs.verify_batch(cap, dims, 1, opening))
        << "Tampered sibling should fail verification";
}

TEST(MerkleTree, VerifyWrongIndexFails) {
    auto mat = make_matrix(4, 3);
    std::vector<Dimensions> dims = { mat.dimensions() };

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    // Open at index 0 but verify as if it were index 1
    auto opening = mmcs.open_batch(0, tree);
    EXPECT_FALSE(mmcs.verify_batch(cap, dims, 1, opening))
        << "Proof for index 0 should not verify at index 1";
}

// ---------------------------------------------------------------------------
// Round-trip: multi-matrix mixed height
// ---------------------------------------------------------------------------

TEST(MerkleTree, RoundTripMixedHeight) {
    auto tall  = make_matrix(8, 2, 1);
    auto short_ = make_matrix(4, 3, 100);

    // Store copies of the dimensions before moving
    Dimensions tall_dims  = tall.dimensions();
    Dimensions short_dims = short_.dimensions();
    std::vector<Dimensions> dims = { tall_dims, short_dims };

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(tall), std::move(short_)});

    // Verify at all 8 indices (for the short matrix, row = index / 2).
    for (size_t i = 0; i < 8; ++i) {
        auto opening = mmcs.open_batch(i, tree);
        EXPECT_TRUE(mmcs.verify_batch(cap, dims, i, opening))
            << "Mixed-height round-trip failed at index " << i;
    }
}

TEST(MerkleTree, RoundTripCapHeight2) {
    auto mat = make_matrix(8, 4);
    std::vector<Dimensions> dims = { mat.dimensions() };

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 2);
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    EXPECT_EQ(cap.cap.size(), 4u);

    for (size_t i = 0; i < 8; ++i) {
        auto opening = mmcs.open_batch(i, tree);
        EXPECT_TRUE(mmcs.verify_batch(cap, dims, i, opening))
            << "cap_height=2 round-trip failed at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Round-trip: reversed matrix order (short before tall)
// ---------------------------------------------------------------------------

TEST(MerkleTree, RoundTripMixedHeightReversedOrder) {
    auto short_ = make_matrix(4, 3, 100);
    auto tall   = make_matrix(8, 2, 1);

    Dimensions short_dims = short_.dimensions();
    Dimensions tall_dims  = tall.dimensions();
    std::vector<Dimensions> dims = { short_dims, tall_dims };

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(short_), std::move(tall)});

    for (size_t i = 0; i < 8; ++i) {
        auto opening = mmcs.open_batch(i, tree);
        EXPECT_TRUE(mmcs.verify_batch(cap, dims, i, opening))
            << "Reversed-order mixed-height round-trip failed at index " << i;
    }
}

TEST(MerkleTree, CommitOrderIndependent) {
    auto tall1  = make_matrix(8, 2, 1);
    auto short1 = make_matrix(4, 3, 100);
    auto tall2  = make_matrix(8, 2, 1);
    auto short2 = make_matrix(4, 3, 100);

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap1, t1] = mmcs.commit({std::move(tall1), std::move(short1)});
    auto [cap2, t2] = mmcs.commit({std::move(short2), std::move(tall2)});

    EXPECT_EQ(cap1, cap2)
        << "Commitment should be the same regardless of matrix order";
}

// ---------------------------------------------------------------------------
// Multiple matrices at the same height
// ---------------------------------------------------------------------------

TEST(MerkleTree, RoundTripThreeMatricesSameHeight) {
    auto m1 = make_matrix(4, 2, 1);
    auto m2 = make_matrix(4, 3, 50);
    auto m3 = make_matrix(4, 1, 200);

    std::vector<Dimensions> dims = {
        m1.dimensions(), m2.dimensions(), m3.dimensions()
    };

    MockMmcs mmcs(MockHasher<DE>{}, MockCompressor<DE>{}, 0);
    auto [cap, tree] = mmcs.commit({std::move(m1), std::move(m2), std::move(m3)});

    for (size_t i = 0; i < 4; ++i) {
        auto opening = mmcs.open_batch(i, tree);
        EXPECT_TRUE(mmcs.verify_batch(cap, dims, i, opening))
            << "Three-matrix round-trip failed at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Integration test with Poseidon2 (non-trivial hash/compress)
// ---------------------------------------------------------------------------

static auto make_poseidon2_bb16() {
    constexpr size_t WIDTH = 16;
    constexpr uint64_t D = 7;
    constexpr size_t ROUNDS_F_HALF = 4;

    std::vector<std::array<BabyBear, WIDTH>> init_consts(
        ROUNDS_F_HALF, std::array<BabyBear, WIDTH>{});
    std::vector<std::array<BabyBear, WIDTH>> term_consts(
        ROUNDS_F_HALF, std::array<BabyBear, WIDTH>{});
    ExternalLayerConstants<BabyBear, WIDTH> ext_consts(init_consts, term_consts);
    std::vector<BabyBear> int_consts(22, BabyBear());

    return create_poseidon2<BabyBear, BabyBear, WIDTH, D>(ext_consts, int_consts);
}

struct Poseidon2Wrapper {
    std::shared_ptr<Poseidon2<BabyBear, BabyBear, 16, 7>> perm;
    void permute_mut(std::array<BabyBear, 16>& state) const { perm->permute_mut(state); }
};

TEST(MerkleTreePoseidon2, RoundTripSingleMatrix) {
    constexpr size_t WIDTH = 16, RATE = 8, OUT = 8, N = 2, CHUNK = 8;

    auto perm = make_poseidon2_bb16();
    Poseidon2Wrapper w{perm};

    using MySponge  = PaddingFreeSponge<Poseidon2Wrapper, BabyBear, WIDTH, RATE, OUT>;
    using MyTrunc   = TruncatedPermutation<Poseidon2Wrapper, BabyBear, N, CHUNK, WIDTH>;
    using MyMmcs    = MerkleTreeMmcs<BabyBear, BabyBear, MySponge, MyTrunc, OUT>;

    MySponge sponge{w};
    MyTrunc  trunc{w};
    MyMmcs   mmcs(std::move(sponge), std::move(trunc), 0);

    // Build a 4×3 BabyBear matrix
    std::vector<BabyBear> vals;
    for (uint32_t i = 1; i <= 12; ++i) vals.push_back(BabyBear(i));
    RowMajorMatrix<BabyBear> mat(std::move(vals), 3);

    std::vector<Dimensions> dims = { mat.dimensions() };
    auto [cap, tree] = mmcs.commit({std::move(mat)});

    EXPECT_EQ(cap.cap.size(), 1u);

    for (size_t i = 0; i < 4; ++i) {
        auto opening = mmcs.open_batch(i, tree);
        EXPECT_TRUE(mmcs.verify_batch(cap, dims, i, opening))
            << "Poseidon2 round-trip failed at index " << i;
    }
}

TEST(MerkleTreePoseidon2, RoundTripMixedHeight) {
    constexpr size_t WIDTH = 16, RATE = 8, OUT = 8, N = 2, CHUNK = 8;

    auto perm = make_poseidon2_bb16();
    Poseidon2Wrapper w{perm};

    using MySponge = PaddingFreeSponge<Poseidon2Wrapper, BabyBear, WIDTH, RATE, OUT>;
    using MyTrunc  = TruncatedPermutation<Poseidon2Wrapper, BabyBear, N, CHUNK, WIDTH>;
    using MyMmcs   = MerkleTreeMmcs<BabyBear, BabyBear, MySponge, MyTrunc, OUT>;

    MySponge sponge{w};
    MyTrunc  trunc{w};
    MyMmcs   mmcs(std::move(sponge), std::move(trunc), 0);

    // Tall: 8×2, Short: 4×3
    std::vector<BabyBear> tall_vals, short_vals;
    for (uint32_t i = 1; i <= 16; ++i) tall_vals.push_back(BabyBear(i));
    for (uint32_t i = 1; i <= 12; ++i) short_vals.push_back(BabyBear(i + 100));
    RowMajorMatrix<BabyBear> tall_mat(std::move(tall_vals), 2);
    RowMajorMatrix<BabyBear> short_mat(std::move(short_vals), 3);

    std::vector<Dimensions> dims = {
        tall_mat.dimensions(), short_mat.dimensions()
    };

    auto [cap, tree] = mmcs.commit({std::move(tall_mat), std::move(short_mat)});

    EXPECT_EQ(cap.cap.size(), 1u);

    for (size_t i = 0; i < 8; ++i) {
        auto opening = mmcs.open_batch(i, tree);
        EXPECT_TRUE(mmcs.verify_batch(cap, dims, i, opening))
            << "Poseidon2 mixed-height round-trip failed at index " << i;
    }
}

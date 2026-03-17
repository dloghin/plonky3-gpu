#pragma once

/**
 * @file merkle_tree.hpp
 * @brief MerkleTree data structure and construction algorithm.
 *
 * Mirrors plonky3/merkle-tree/src/merkle_tree.rs.
 *
 * The tree is built over a batch of RowMajorMatrix<F> with potentially
 * different heights (all must be powers of 2).  Shorter matrices are
 * "injected" at the tree level where their height equals the number of
 * nodes on that level.
 *
 * Build algorithm (binary tree, N == 2):
 *  1. Sort matrices by height descending.
 *  2. Hash all rows of the tallest matrices together to form the leaf layer.
 *  3. Compress pairs up the tree level-by-level.
 *  4. When the current level size equals a matrix height, hash those rows
 *     and incorporate them into each node via an additional compress call.
 *  5. Stop when the layer size equals 2^cap_height.
 */

#include "merkle_cap.hpp"
#include "dense_matrix.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <vector>

namespace p3_merkle_tree {

using p3_matrix::RowMajorMatrix;
using p3_matrix::Dimensions;

/**
 * @brief A Merkle tree built over a batch of matrices.
 *
 * @tparam F            Field element type in the leaf matrices.
 * @tparam W            Digest word type (typically the same as F).
 * @tparam DIGEST_ELEMS Number of elements per hash digest.
 */
template <typename F, typename W, size_t DIGEST_ELEMS>
struct MerkleTree {
    /// The original leaf matrices (stored for later opening).
    std::vector<RowMajorMatrix<F>> leaves;

    /**
     * @brief All digest layers, bottom-up.
     *
     * digest_layers[0].size() == tallest matrix height.
     * digest_layers[k].size() == tallest matrix height / 2^k.
     * The last layer is the cap (size == 2^cap_height).
     */
    std::vector<std::vector<std::array<W, DIGEST_ELEMS>>> digest_layers;
};

// ---------------------------------------------------------------------------
// build_merkle_tree
// ---------------------------------------------------------------------------

/**
 * @brief Build a MerkleTree from a batch of leaf matrices.
 *
 * @tparam F            Field element type.
 * @tparam W            Digest word type.
 * @tparam DIGEST_ELEMS Digest length in words.
 * @tparam Hasher       Type with member:
 *                        std::array<W,DIGEST_ELEMS>
 *                        hash_iter_slices(const std::vector<std::vector<F>>&)
 * @tparam Compressor   Type with member:
 *                        std::array<W,DIGEST_ELEMS>
 *                        compress(const std::array<std::array<W,DIGEST_ELEMS>,2>&)
 *
 * @param matrices    Batch of leaf matrices (moved into the tree).
 * @param hasher      Leaf hasher.
 * @param compressor  Internal-node compressor.
 * @param cap_height  Height of the cap layer (0 = single root).
 */
template <typename F, typename W, size_t DIGEST_ELEMS,
          typename Hasher, typename Compressor>
MerkleTree<F, W, DIGEST_ELEMS> build_merkle_tree(
    std::vector<RowMajorMatrix<F>> matrices,
    const Hasher& hasher,
    const Compressor& compressor,
    size_t cap_height)
{
    assert(!matrices.empty() && "need at least one matrix");

    // Step 1: Sort by height descending, keeping original order for same heights.
    std::stable_sort(matrices.begin(), matrices.end(),
        [](const RowMajorMatrix<F>& a, const RowMajorMatrix<F>& b) {
            return a.height() > b.height();
        });

    const size_t max_height = matrices[0].height();
    assert(max_height > 0 && "matrices must be non-empty");
    assert((max_height & (max_height - 1)) == 0 &&
           "matrix height must be a power of two");

    // We process matrices in groups of equal height.
    // 'next_mat' is the index of the next unprocessed matrix.
    size_t next_mat = 0;

    // Helper: collect all matrices whose height equals `h` starting at next_mat,
    // advancing next_mat past them.
    auto collect_at_height = [&](size_t h) -> std::vector<size_t> {
        std::vector<size_t> indices;
        while (next_mat < matrices.size() && matrices[next_mat].height() == h) {
            indices.push_back(next_mat++);
        }
        return indices;
    };

    // Helper: for a given row index `row` and a set of matrix indices,
    // build the slice list [mat.row(row), ...] and call hash_iter_slices.
    auto hash_row_group = [&](size_t row,
                               const std::vector<size_t>& mat_indices)
        -> std::array<W, DIGEST_ELEMS>
    {
        std::vector<std::vector<F>> slices;
        slices.reserve(mat_indices.size());
        for (size_t idx : mat_indices) {
            slices.push_back(matrices[idx].row(row));
        }
        return hasher.hash_iter_slices(slices);
    };

    // Step 2: Build the leaf layer from the tallest matrices.
    std::vector<size_t> tallest_indices = collect_at_height(max_height);
    assert(!tallest_indices.empty());

    std::vector<std::array<W, DIGEST_ELEMS>> current_layer(max_height);
    for (size_t i = 0; i < max_height; ++i) {
        current_layer[i] = hash_row_group(i, tallest_indices);
    }

    MerkleTree<F, W, DIGEST_ELEMS> tree;
    tree.digest_layers.push_back(current_layer);

    // Step 3: Build internal layers until we reach the cap.
    size_t level_size = max_height;
    const size_t cap_size = (size_t(1) << cap_height);

    while (level_size > cap_size) {
        level_size /= 2; // binary tree: each level halves

        std::vector<std::array<W, DIGEST_ELEMS>> next_layer(level_size);

        // Compress pairs from the previous layer.
        for (size_t i = 0; i < level_size; ++i) {
            std::array<std::array<W, DIGEST_ELEMS>, 2> pair = {
                current_layer[2 * i],
                current_layer[2 * i + 1]
            };
            next_layer[i] = compressor.compress(pair);
        }

        // Inject matrices whose height equals the current level size.
        std::vector<size_t> inject_indices = collect_at_height(level_size);
        if (!inject_indices.empty()) {
            for (size_t i = 0; i < level_size; ++i) {
                std::array<W, DIGEST_ELEMS> row_hash =
                    hash_row_group(i, inject_indices);
                std::array<std::array<W, DIGEST_ELEMS>, 2> pair = {
                    next_layer[i],
                    row_hash
                };
                next_layer[i] = compressor.compress(pair);
            }
        }

        tree.digest_layers.push_back(next_layer);
        current_layer = std::move(next_layer);
    }

    // Store leaf matrices.
    tree.leaves = std::move(matrices);

    return tree;
}

// ---------------------------------------------------------------------------
// Accessor helpers
// ---------------------------------------------------------------------------

/**
 * @brief Return the cap (top layer) of a built MerkleTree as a MerkleCap.
 */
template <typename F, typename W, size_t DIGEST_ELEMS>
MerkleCap<W, DIGEST_ELEMS> get_cap(const MerkleTree<F, W, DIGEST_ELEMS>& tree) {
    assert(!tree.digest_layers.empty());
    return MerkleCap<W, DIGEST_ELEMS>(tree.digest_layers.back());
}

} // namespace p3_merkle_tree

#pragma once

/**
 * @file merkle_tree_mmcs.hpp
 * @brief MerkleTreeMmcs: concrete MMCS implementation using a Merkle tree.
 *
 * Mirrors plonky3/merkle-tree/src/mmcs.rs: MerkleTreeMmcs<P,PW,H,C,DIGEST_ELEMS>.
 *
 * Provides commit / open_batch / verify_batch over batches of RowMajorMatrix<F>.
 *
 * For the FRI test:
 *   MerkleTreeMmcs<BabyBear, BabyBear,
 *                  PaddingFreeSponge<Poseidon2,...>,
 *                  TruncatedPermutation<Poseidon2,...>,
 *                  8>
 */

#include "merkle_tree.hpp"
#include "merkle_cap.hpp"
#include "dense_matrix.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_merkle_tree {

using p3_matrix::RowMajorMatrix;
using p3_matrix::Dimensions;

// ---------------------------------------------------------------------------
// BatchOpening
// ---------------------------------------------------------------------------

/**
 * @brief Proof that a set of rows was opened from a committed batch.
 *
 * @tparam F            Leaf field element type.
 * @tparam W            Digest word type.
 * @tparam DIGEST_ELEMS Digest length in words.
 */
template <typename F, typename W, size_t DIGEST_ELEMS>
struct BatchOpening {
    /**
     * Opened row values, one entry per matrix.
     * opened_values[i] is the row of matrix i at the queried index
     * (adjusted modulo matrix height).
     */
    std::vector<std::vector<F>> opened_values;

    /**
     * Merkle path: sibling digests from the leaf level up to (but not
     * including) the cap.  proof[0] is the sibling at the leaf level,
     * proof.back() is the sibling just below the cap.
     */
    std::vector<std::array<W, DIGEST_ELEMS>> proof;
};

// ---------------------------------------------------------------------------
// MerkleTreeMmcs
// ---------------------------------------------------------------------------

/**
 * @brief MMCS (Multi-Matrix Commitment Scheme) backed by a Merkle tree.
 *
 * @tparam F            Leaf field element type.
 * @tparam W            Digest word type (typically the same as F).
 * @tparam Hasher       Leaf hasher satisfying CryptographicHasher concept.
 * @tparam Compressor   Compression function satisfying PseudoCompressionFunction.
 * @tparam DIGEST_ELEMS Number of field elements per digest.
 */
template <typename F, typename W,
          typename Hasher, typename Compressor,
          size_t DIGEST_ELEMS>
class MerkleTreeMmcs {
public:
    using Commitment = MerkleCap<W, DIGEST_ELEMS>;
    using ProverData = MerkleTree<F, W, DIGEST_ELEMS>;
    using Opening    = BatchOpening<F, W, DIGEST_ELEMS>;

    MerkleTreeMmcs(Hasher hasher, Compressor compressor, size_t cap_height = 0)
        : hasher_(std::move(hasher))
        , compressor_(std::move(compressor))
        , cap_height_(cap_height)
    {}

    // ------------------------------------------------------------------
    // commit
    // ------------------------------------------------------------------

    /**
     * @brief Commit to a batch of matrices.
     *
     * @param matrices  Leaf matrices (moved in).
     * @return Pair of (commitment cap, full prover data).
     */
    std::pair<Commitment, ProverData> commit(
        std::vector<RowMajorMatrix<F>> matrices)
    {
        ProverData tree = build_merkle_tree<F, W, DIGEST_ELEMS>(
            std::move(matrices), hasher_, compressor_, cap_height_);
        Commitment cap = get_cap(tree);
        return {std::move(cap), std::move(tree)};
    }

    // ------------------------------------------------------------------
    // open_batch
    // ------------------------------------------------------------------

    /**
     * @brief Open all matrices in the prover data at a given row index.
     *
     * @param index  Global row index (will be reduced modulo each matrix height).
     * @param tree   Prover data from commit().
     * @return BatchOpening containing opened rows and Merkle proof.
     */
    Opening open_batch(size_t index, const ProverData& tree) const {
        const size_t max_height = tree.digest_layers[0].size();
        // Number of proof steps = number of layers below the cap.
        const size_t num_steps  = tree.digest_layers.size() - 1;

        Opening result;

        // Collect opened values for each leaf matrix.
        // For the tallest matrices the row == index (they are at the leaf level).
        // For shorter matrices injected at a higher level, we need the position
        // within that level: row = index / (max_height / mat.height()).
        // (E.g. with max_height=8 and mat.height()=4, level-1 has 4 nodes, so
        //  leaf index `i` maps to level-1 node `i/2`.)
        result.opened_values.reserve(tree.leaves.size());
        for (const auto& mat : tree.leaves) {
            size_t stride = max_height / mat.height(); // always a power of two
            size_t row = index / stride;
            result.opened_values.push_back(mat.row(row));
        }

        // Build the Merkle proof: walk from leaf to just below the cap.
        result.proof.reserve(num_steps);
        size_t current_index = index;
        for (size_t step = 0; step < num_steps; ++step) {
            // At step `step` we are at digest_layers[step].
            const auto& layer = tree.digest_layers[step];
            size_t sibling_index = current_index ^ 1; // XOR 1 flips the last bit
            result.proof.push_back(layer[sibling_index]);
            current_index >>= 1; // move to parent index
        }

        return result;
    }

    // ------------------------------------------------------------------
    // verify_batch
    // ------------------------------------------------------------------

    /**
     * @brief Verify a batch opening against a commitment.
     *
     * @param commitment  The MerkleCap from commit().
     * @param dims        Dimensions of each committed matrix, in the SAME
     *                    order as passed to commit() (before internal sorting).
     * @param index       Row index that was opened.
     * @param opening     BatchOpening from open_batch().
     * @return true if the proof is valid, false otherwise.
     */
    bool verify_batch(
        const Commitment& commitment,
        const std::vector<Dimensions>& dims,
        size_t index,
        const Opening& opening) const
    {
        if (dims.size() != opening.opened_values.size()) return false;
        if (dims.empty()) return false;

        // Sort matrix indices by height descending (mirrors build order).
        std::vector<size_t> order(dims.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return dims[a].height > dims[b].height; });
        const size_t max_height = dims[order[0]].height;
        if (max_height == 0) return false;

        // Number of layers in the tree.
        const size_t cap_size  = commitment.cap.size();
        // num_levels = log2(max_height / cap_size)
        size_t num_levels = 0;
        {
            size_t h = max_height;
            while (h > cap_size) { h /= 2; ++num_levels; }
        }

        if (opening.proof.size() != num_levels) return false;

        // ---- Step 1: hash the tallest matrices' opened rows ----
        size_t sorted_pos = 0;
        auto collect_at_height = [&](size_t h) -> std::vector<size_t> {
            std::vector<size_t> indices;
            while (sorted_pos < order.size() &&
                   dims[order[sorted_pos]].height == h) {
                indices.push_back(order[sorted_pos++]);
            }
            return indices;
        };

        std::vector<size_t> tallest = collect_at_height(max_height);
        // Build slices for the tallest matrices.
        std::array<W, DIGEST_ELEMS> current;
        {
            std::vector<std::vector<F>> slices;
            for (size_t idx : tallest) {
                slices.push_back(opening.opened_values[idx]);
            }
            current = hasher_.hash_iter_slices(slices);
        }

        // ---- Step 2: walk up the proof ----
        size_t current_index = index;
        for (size_t step = 0; step < num_levels; ++step) {
            const auto& sibling = opening.proof[step];
            bool is_left = (current_index & 1) == 0;

            std::array<std::array<W, DIGEST_ELEMS>, 2> pair;
            if (is_left) {
                pair = {current, sibling};
            } else {
                pair = {sibling, current};
            }
            current = compressor_.compress(pair);
            current_index >>= 1;

            // Inject matrices at the new level size.
            size_t new_level_size = max_height >> (step + 1);
            std::vector<size_t> inject = collect_at_height(new_level_size);
            if (!inject.empty()) {
                // Hash their rows. The opener must have provided the correct
                // row for the current level (row = index / stride, where
                // stride = max_height / mat.height()).  We trust the opener.
                std::vector<std::vector<F>> slices;
                for (size_t idx : inject) {
                    slices.push_back(opening.opened_values[idx]);
                }
                std::array<W, DIGEST_ELEMS> row_hash =
                    hasher_.hash_iter_slices(slices);
                std::array<std::array<W, DIGEST_ELEMS>, 2> p2 = {current, row_hash};
                current = compressor_.compress(p2);
            }
        }

        // ---- Step 3: check against the cap ----
        if (current_index >= commitment.cap.size()) return false;
        return current == commitment.cap[current_index];
    }

private:
    Hasher     hasher_;
    Compressor compressor_;
    size_t     cap_height_;
};

} // namespace p3_merkle_tree

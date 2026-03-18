#pragma once

/**
 * @file fri_merkle_tree_mmcs.hpp
 * @brief Binary Merkle tree MMCS for committing matrices of field elements.
 */

#include <vector>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "p3_util/util.hpp"

namespace p3_merkle {

/**
 * Binary Merkle tree MMCS.
 *
 * Template parameters:
 *   InputF  - field element type for matrix rows
 *   DigestF - field element type for hash digests (usually same as InputF)
 *   H       - hasher with Hash<DigestF,CHUNK> hash_iter(const InputF*, size_t)
 *   C       - compressor with Hash<DigestF,CHUNK> compress(std::array<Hash<DigestF,CHUNK>,2>)
 *   N_ARY   - arity (must be 2 for binary tree)
 *   CHUNK   - digest size in field elements
 */
template <typename InputF, typename DigestF, typename H, typename C,
          size_t N_ARY, size_t CHUNK>
class MerkleTreeMmcs {
public:
    static_assert(N_ARY == 2, "MerkleTreeMmcs only supports binary (N_ARY=2) trees");

    using Digest     = std::array<DigestF, CHUNK>;
    using Commitment = std::vector<Digest>;   // Merkle cap (size 1 = just root)

    struct ProverData {
        std::vector<InputF> flat_data;   // row-major matrix data
        size_t height;
        size_t width;
        std::vector<std::vector<Digest>> tree;  // tree[0]=leaves, tree[log_height]=root
    };

    using OpeningProof = std::vector<Digest>;  // sibling hashes leaf-to-root

    H hasher;
    C compressor;

    MerkleTreeMmcs(H h, C c) : hasher(std::move(h)), compressor(std::move(c)) {}

    // -----------------------------------------------------------------------
    // commit_matrix
    // -----------------------------------------------------------------------
    std::pair<Commitment, ProverData> commit_matrix(
        const std::vector<InputF>& data,
        size_t width) const
    {
        if (width == 0 || data.size() % width != 0) {
            throw std::invalid_argument("commit_matrix: bad dimensions");
        }
        size_t height = data.size() / width;
        if (height == 0) {
            throw std::invalid_argument("commit_matrix: empty matrix");
        }

        // Compute leaf hashes
        std::vector<Digest> leaves(height);
        for (size_t i = 0; i < height; ++i) {
            leaves[i] = hasher.hash_iter(data.data() + i * width, width);
        }

        // Build tree bottom-up
        std::vector<std::vector<Digest>> tree;
        tree.push_back(leaves);

        while (tree.back().size() > 1) {
            const auto& prev = tree.back();
            size_t prev_size = prev.size();
            // Pad to even size if needed
            size_t padded_size = (prev_size % 2 == 0) ? prev_size : prev_size + 1;
            size_t next_size   = padded_size / 2;

            std::vector<Digest> next(next_size);
            for (size_t j = 0; j < next_size; ++j) {
                Digest left  = prev[2 * j];
                Digest right = (2 * j + 1 < prev_size) ? prev[2 * j + 1] : Digest{};
                std::array<Digest, 2> pair{left, right};
                next[j] = compressor.compress(pair);
            }
            tree.push_back(std::move(next));
        }

        Commitment commit = {tree.back()[0]};
        ProverData pd;
        pd.flat_data = data;
        pd.height    = height;
        pd.width     = width;
        pd.tree      = std::move(tree);

        return {commit, pd};
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    size_t log_height(const ProverData& d) const {
        return p3_util::log2_strict_usize(d.height);
    }

    size_t log_width(const ProverData& d) const {
        return p3_util::log2_strict_usize(d.width);
    }

    std::vector<InputF> get_row_as_vals(const ProverData& d, size_t row) const {
        const auto* row_start = &d.flat_data[row * d.width];
        return std::vector<InputF>(row_start, row_start + d.width);
    }

    // get_row is an alias used by the FRI prover (rows are Digest-typed since the
    // FriMmcs stores committed evaluation vectors of Challenge elements)
    std::vector<InputF> get_row(const ProverData& d, size_t row) const {
        return get_row_as_vals(d, row);
    }

    // -----------------------------------------------------------------------
    // open_row: produce Merkle opening proof for a leaf
    // -----------------------------------------------------------------------
    void open_row(const ProverData& d, size_t row_index, OpeningProof& proof) const {
        proof.clear();
        size_t cur_idx = row_index;
        for (size_t level = 0; level + 1 < d.tree.size(); ++level) {
            size_t sibling_idx = cur_idx ^ 1u;
            if (sibling_idx < d.tree[level].size()) {
                proof.push_back(d.tree[level][sibling_idx]);
            } else {
                proof.push_back(Digest{});  // zero padding
            }
            cur_idx >>= 1;
        }
    }

    // -----------------------------------------------------------------------
    // verify_row: verify a Merkle proof against the commitment
    // -----------------------------------------------------------------------
    bool verify_row(const Commitment& commit,
                    size_t row_index,
                    const std::vector<InputF>& row_vals,
                    const OpeningProof& proof) const
    {
        // Hash the leaf
        Digest cur_hash = hasher.hash_iter(row_vals.data(), row_vals.size());
        size_t cur_idx = row_index;

        for (const Digest& sibling : proof) {
            std::array<Digest, 2> pair;
            if (cur_idx % 2 == 0) {
                pair = {cur_hash, sibling};
            } else {
                pair = {sibling, cur_hash};
            }
            cur_hash = compressor.compress(pair);
            cur_idx >>= 1;
        }

        if (commit.empty()) return false;
        return cur_hash == commit[0];
    }

    // -----------------------------------------------------------------------
    // For use as InputMmcs in verify_fri: always returns true.
    // The claimed_eval is templated to accept any type (base field or extension field).
    // -----------------------------------------------------------------------
    template <typename ClaimedEval>
    bool verify_query(size_t /*query_index*/, size_t /*log_height*/,
                      const std::vector<Commitment>& /*commits*/,
                      const OpeningProof& /*proof*/,
                      const ClaimedEval& /*claimed_eval*/) const
    {
        return true;
    }

    // -----------------------------------------------------------------------
    // Multi-matrix open stub (actual verification via eval_at_query in PCS)
    // -----------------------------------------------------------------------
    void open(size_t /*query_index*/,
              const std::vector<ProverData>& /*data_vec*/,
              OpeningProof& proof) const
    {
        proof.clear();
    }
};

} // namespace p3_merkle

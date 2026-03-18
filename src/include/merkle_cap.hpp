#pragma once

/**
 * @file merkle_cap.hpp
 * @brief MerkleCap type: the top of a Merkle tree commitment.
 *
 * Mirrors plonky3/merkle-tree/src/merkle_tree.rs: MerkleCap<W, DIGEST_ELEMS>.
 *
 * When cap_height == 0, the cap contains a single root hash.
 * When cap_height == k, the cap contains 2^k digests (the top k levels are
 * excluded from the proof, leaving the cap as the commitment).
 */

#include <array>
#include <cstddef>
#include <vector>

namespace p3_merkle_tree {

/**
 * @brief The commitment cap of a Merkle tree.
 *
 * @tparam W            Digest word type (typically BabyBear).
 * @tparam DIGEST_ELEMS Number of field elements per digest (e.g., 8).
 */
template <typename W, size_t DIGEST_ELEMS>
struct MerkleCap {
    /// The top-level digests. Size == 2^cap_height.
    std::vector<std::array<W, DIGEST_ELEMS>> cap;

    MerkleCap() = default;
    explicit MerkleCap(std::vector<std::array<W, DIGEST_ELEMS>> c)
        : cap(std::move(c)) {}

    bool operator==(const MerkleCap& other) const { return cap == other.cap; }
    bool operator!=(const MerkleCap& other) const { return cap != other.cap; }
};

} // namespace p3_merkle_tree

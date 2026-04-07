#pragma once

#include "cuda_compat.hpp"
#include "merkle_cap.hpp"
#include "merkle_tree.hpp"
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace p3_field {

/**
 * @brief Serialization utilities for FRI proofs and commitments.
 * 
 * Enables saving/loading proofs to/from disk and cross-language compatibility.
 */
class ProofSerialization {
public:
    // Note: Implementations for FriProof, MerkleCap, etc., are placeholders
    // pending the implementation of the underlying data structures.

    /**
     * @brief Serialize MerkleCap to bytes.
     * 
     * Since MerkleCap is templated, we provide specialized or generic encoding.
     * For this task, we'll focus on the structure.
     */
    template <typename W, size_t DIGEST_ELEMS>
    static std::vector<uint8_t> encode_merkle_cap(const p3_merkle_tree::MerkleCap<W, DIGEST_ELEMS>& cap) {
        // Placeholder implementation
        return {};
    }

    /**
     * @brief Deserialize MerkleCap from bytes.
     */
    template <typename W, size_t DIGEST_ELEMS>
    static p3_merkle_tree::MerkleCap<W, DIGEST_ELEMS> decode_merkle_cap(const std::vector<uint8_t>& bytes) {
        // Placeholder implementation
        return p3_merkle_tree::MerkleCap<W, DIGEST_ELEMS>();
    }
};

} // namespace p3_field


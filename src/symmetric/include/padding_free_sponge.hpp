#pragma once

/**
 * @file padding_free_sponge.hpp
 * @brief PaddingFreeSponge hash function wrapping a cryptographic permutation.
 *
 * Mirrors plonky3/symmetric/src/sponge.rs: PaddingFreeSponge<Perm, WIDTH, RATE, OUT>.
 *
 * Algorithm (absorb-then-squeeze, no padding):
 *   1. state = [0; WIDTH]
 *   2. For each chunk of RATE elements from input:
 *        state[0..RATE] = chunk
 *        permute(state)
 *   3. return state[0..OUT]
 *
 * For the FRI test configuration:
 *   PaddingFreeSponge<Poseidon2BabyBear<16>, 16, 8, 8>
 */

#include "hash.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace p3_symmetric {

/**
 * @brief Padding-free sponge hash function.
 *
 * @tparam Perm   Permutation type; must expose
 *                  void permute_mut(std::array<F, WIDTH>&)
 * @tparam F      Field element type
 * @tparam WIDTH  State width (= Perm state width)
 * @tparam RATE   Number of field elements absorbed per permutation call
 * @tparam OUT    Number of field elements in the output digest
 *
 * Constraint: RATE <= WIDTH, OUT <= WIDTH.
 */
template <typename Perm, typename F, size_t WIDTH, size_t RATE, size_t OUT>
class PaddingFreeSponge {
    static_assert(RATE <= WIDTH, "RATE must be <= WIDTH");
    static_assert(OUT  <= WIDTH, "OUT must be <= WIDTH");

    mutable Perm permutation_;

public:
    explicit PaddingFreeSponge(Perm perm) : permutation_(std::move(perm)) {}

    /**
     * @brief Hash a flat sequence of field elements.
     *
     * Corresponds to Rust's hash_iter / hash_slice.
     *
     * @param input  All field elements to hash (any length, including 0).
     * @return       Digest of OUT field elements.
     */
    Hash<F, OUT> hash_iter(const std::vector<F>& input) const {
        std::array<F, WIDTH> state{};

        // Absorb input in RATE-sized chunks, always applying at least one permutation.
        // An empty input absorbs a single all-zero block.
        size_t pos = 0;
        do {
            size_t chunk_size = std::min(RATE, input.size() - pos);
            std::copy_n(input.begin() + pos, chunk_size, state.begin());
            std::fill(state.begin() + chunk_size, state.begin() + RATE, F());
            permutation_.permute_mut(state);
            pos += chunk_size;
        } while (pos < input.size());

        Hash<F, OUT> digest;
        for (size_t i = 0; i < OUT; ++i) {
            digest[i] = state[i];
        }
        return digest;
    }

    /**
     * @brief Hash a sequence of row slices (for Merkle tree leaf hashing).
     *
     * Each row in `slices` is a slice of field elements. The elements are
     * serialised to u32 via their canonical representation and fed through
     * the sponge RATE elements at a time.
     *
     * For a 32-bit field like BabyBear each element maps to exactly one u32,
     * and a u32 maps back to exactly one field element (value < p, so it is
     * already reduced).
     *
     * @param slices  Collection of rows; each row is a vector of F.
     * @return        Digest of OUT field elements.
     */
    Hash<F, OUT> hash_iter_slices(const std::vector<std::vector<F>>& slices) const {
        // Flatten slices into a single stream of field elements, converting
        // via the canonical u32 representation (RawDataSerializable semantics).
        std::vector<F> flat;
        size_t total_elems = 0;
        for (const auto& row : slices) {
            total_elems += row.size();
        }
        flat.reserve(total_elems);
        for (const auto& row : slices) {
            for (const auto& elem : row) {
                // as_canonical_u64() is the canonical representation;
                // reconstruct as a field element so values stay < p.
                flat.push_back(F(static_cast<uint32_t>(elem.as_canonical_u64())));
            }
        }
        return hash_iter(flat);
    }
};

} // namespace p3_symmetric

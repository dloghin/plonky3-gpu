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

#include <array>
#include <cstddef>
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
     * Faithfully mirrors the Rust loop in sponge.rs: only the positions
     * actually consumed from the input are overwritten in `state`;
     * remaining positions keep their values from the previous permutation.
     * An empty input returns the default (all-zero) state without permuting.
     *
     * @param input  All field elements to hash (any length, including 0).
     * @return       Digest of OUT field elements.
     */
    Hash<F, OUT> hash_iter(const F* data, size_t len) const {
        std::array<F, WIDTH> state{};
        size_t input_pos = 0;

        for (;;) {
            size_t i = 0;
            for (; i < RATE && input_pos < len; ++i, ++input_pos) {
                state[i] = data[input_pos];
            }
            if (i == RATE) {
                permutation_.permute_mut(state);
            } else {
                if (i > 0) {
                    permutation_.permute_mut(state);
                }
                break;
            }
        }

        Hash<F, OUT> digest;
        for (size_t i = 0; i < OUT; ++i) {
            digest[i] = state[i];
        }
        return digest;
    }

    Hash<F, OUT> hash_iter(const std::vector<F>& input) const {
        return hash_iter(input.data(), input.size());
    }

    /**
     * @brief Hash a sequence of row slices (for Merkle tree leaf hashing).
     *
     * Mirrors Rust's default CryptographicHasher::hash_iter_slices which
     * simply flattens the slices and delegates to hash_iter.
     *
     * @param slices  Collection of rows; each row is a vector of F.
     * @return        Digest of OUT field elements.
     */
    Hash<F, OUT> hash_iter_slices(const std::vector<std::vector<F>>& slices) const {
        std::vector<F> flat;
        size_t total_elems = 0;
        for (const auto& row : slices) {
            total_elems += row.size();
        }
        flat.reserve(total_elems);
        for (const auto& row : slices) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
        return hash_iter(flat);
    }
};

} // namespace p3_symmetric

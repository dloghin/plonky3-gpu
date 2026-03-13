#pragma once

/**
 * @file truncated_permutation.hpp
 * @brief TruncatedPermutation: N-to-1 compression function for Merkle trees.
 *
 * Mirrors plonky3/symmetric/src/compression.rs:
 *   TruncatedPermutation<Perm, N, CHUNK, WIDTH>
 *
 * Algorithm:
 *   1. state = [0; WIDTH]
 *   2. state[0..CHUNK]         = input[0]
 *      state[CHUNK..2*CHUNK]   = input[1]
 *      ...
 *      state[(N-1)*CHUNK..N*CHUNK] = input[N-1]
 *   3. permute(state)
 *   4. return state[0..CHUNK]
 *
 * For the FRI test configuration:
 *   TruncatedPermutation<Poseidon2BabyBear<16>, 2, 8, 16>
 *   -- compresses 2 x 8 = 16 BabyBear elements, outputs first 8.
 */

#include "hash.hpp"

#include <array>
#include <cstddef>

namespace p3_symmetric {

/**
 * @brief N-to-1 compression function built from a permutation.
 *
 * @tparam Perm   Permutation type; must expose
 *                  void permute_mut(std::array<F, WIDTH>&)
 * @tparam F      Field element type
 * @tparam N      Arity – number of input chunks
 * @tparam CHUNK  Size of each input/output chunk in field elements
 * @tparam WIDTH  State width (must satisfy N * CHUNK <= WIDTH)
 *
 * Satisfies the PseudoCompressionFunction concept:
 *   T = std::array<F, CHUNK>, N inputs -> T output.
 */
template <typename Perm, typename F, size_t N, size_t CHUNK, size_t WIDTH>
class TruncatedPermutation {
    static_assert(N * CHUNK <= WIDTH,
        "N * CHUNK must be <= WIDTH to fit all inputs in the state");

    mutable Perm permutation_;

public:
    explicit TruncatedPermutation(Perm perm) : permutation_(std::move(perm)) {}

    /**
     * @brief Compress N chunks of CHUNK field elements into one chunk.
     *
     * @param input  Array of N arrays, each of CHUNK field elements.
     * @return       First CHUNK elements of the permuted state.
     */
    Hash<F, CHUNK> compress(
        const std::array<std::array<F, CHUNK>, N>& input
    ) const {
        // Build state
        std::array<F, WIDTH> state{};
        // Copy input chunks into state
        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < CHUNK; ++c) {
                state[n * CHUNK + c] = input[n][c];
            }
        }
        // Apply permutation
        permutation_.permute_mut(state);
        // Truncate to first CHUNK elements
        Hash<F, CHUNK> out;
        for (size_t i = 0; i < CHUNK; ++i) {
            out[i] = state[i];
        }
        return out;
    }
};

} // namespace p3_symmetric

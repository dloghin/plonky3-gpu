#pragma once

/**
 * @file hash.hpp
 * @brief Hash type alias and cryptographic hasher/compressor concepts for symmetric primitives.
 *
 * Mirrors plonky3/symmetric/src/hash.rs and plonky3/symmetric/src/lib.rs.
 */

#include <array>
#include <cstddef>
#include <vector>

namespace p3_symmetric {

/**
 * @brief A fixed-size hash output: an array of N field elements of type F.
 *
 * Corresponds to Rust's `Hash<F, N>` type alias: `[F; N]`.
 */
template <typename F, size_t N>
using Hash = std::array<F, N>;

/**
 * @brief Concept: CryptographicHasher<F, OUT_ARRAY>
 *
 * A type satisfying this concept must provide:
 *   - hash_iter(const std::vector<F>&) -> OUT_ARRAY
 *   - hash_iter_slices(const std::vector<std::vector<F>>&) -> OUT_ARRAY
 *
 * Used by Merkle tree leaf hashing.  OUT_ARRAY is typically std::array<F, OUT>.
 */
// Note: C++17 does not have "requires" so we document the concept as a comment.
// Implementors should provide both hash_iter and hash_iter_slices methods.

/**
 * @brief Concept: PseudoCompressionFunction<T, N>
 *
 * A type satisfying this concept must provide:
 *   - compress(std::array<T, N>) -> T
 *
 * TruncatedPermutation satisfies this for T = std::array<F, CHUNK> and N = 2.
 */

} // namespace p3_symmetric

#pragma once

#include <cstdint>
#include <utility>
#include <stdexcept>
#include <string>

namespace poseidon2 {

/**
 * @brief Compute GCD of two 64-bit unsigned integers
 */
constexpr uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/**
 * @brief Check if two 64-bit unsigned integers are relatively prime
 */
constexpr bool relatively_prime_u64(uint64_t a, uint64_t b) {
    return gcd_u64(a, b) == 1;
}

/**
 * @brief Compute log2 of a 64-bit unsigned integer
 */
constexpr uint32_t log2_u64(uint64_t n) {
    uint32_t count = 0;
    while (n > 1) {
        n >>= 1;
        count++;
    }
    return count;
}

/**
 * @brief Get the number of full and partial rounds for Poseidon2 with 128-bit security
 *
 * Returns a pair (rounds_f, rounds_p) for the given width and exponent D.
 *
 * @param width The state width (supported: 8, 12, 16 for 64-bit; 16, 24 for 31-bit)
 * @param d The S-box exponent (typically 3, 5, 7, 9, or 11)
 * @param field_order The order of the field (prime modulus)
 *
 * @return A pair of (rounds_f, rounds_p) or throws if parameters are invalid
 * @throws std::runtime_error if d is not a valid permutation or parameters not supported
 */
inline std::pair<size_t, size_t> poseidon2_round_numbers_128(
    size_t width,
    uint64_t d,
    uint64_t field_order
) {
    // Check that d is a valid permutation
    if (!relatively_prime_u64(d, field_order - 1)) {
        throw std::runtime_error(
            "Invalid permutation: gcd(d, field_order - 1) must be 1"
        );
    }

    // Compute the number of bits in the prime
    uint32_t prime_bit_number = log2_u64(field_order) + 1;

    // Return appropriate round numbers based on field size
    if (prime_bit_number == 31) {
        // 31-bit primes (e.g., Baby Bear, Mersenne-31)
        switch (width) {
            case 16:
                switch (d) {
                    case 3:  return {8, 20};
                    case 5:  return {8, 14};
                    case 7:  return {8, 13};
                    case 9:  return {8, 13};
                    case 11: return {8, 13};
                    default: break;
                }
                break;
            case 24:
                switch (d) {
                    case 3:  return {8, 23};
                    case 5:  return {8, 22};
                    case 7:  return {8, 21};
                    case 9:  return {8, 21};
                    case 11: return {8, 21};
                    default: break;
                }
                break;
        }
    } else if (prime_bit_number == 64) {
        // 64-bit primes (e.g., Goldilocks)
        switch (width) {
            case 8:
                switch (d) {
                    case 3:  return {8, 41};
                    case 5:  return {8, 27};
                    case 7:  return {8, 22};
                    case 9:  return {8, 19};
                    case 11: return {8, 17};
                    default: break;
                }
                break;
            case 12:
                switch (d) {
                    case 3:  return {8, 42};
                    case 5:  return {8, 27};
                    case 7:  return {8, 22};
                    case 9:  return {8, 20};
                    case 11: return {8, 18};
                    default: break;
                }
                break;
            case 16:
                switch (d) {
                    case 3:  return {8, 42};
                    case 5:  return {8, 27};
                    case 7:  return {8, 22};
                    case 9:  return {8, 20};
                    case 11: return {8, 18};
                    default: break;
                }
                break;
        }
    }

    // If we reach here, the parameters are not supported
    throw std::runtime_error(
        "The given pair of width and D has not been checked for these fields, "
        "or the optimal parameters for that size of prime have not been computed."
    );
}

} // namespace poseidon2


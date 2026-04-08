#pragma once

/**
 * @file keccak.hpp
 * @brief Keccak-f[1600] permutation and Keccak-256 (Ethereum) byte hashing.
 *
 * Keccak-f matches tiny-keccak / p3-keccak. Keccak-256 uses rate 136 bytes,
 * delimiter 0x01, and trailing 0x80 padding (Keccak submission, not SHA3-256).
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "cuda_compat.hpp"

namespace p3_symmetric {

namespace keccak_detail {

P3_HOST_DEVICE P3_INLINE uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << (r & 63)) | (x >> ((-r) & 63));
}

P3_HOST_DEVICE P3_INLINE void keccak_f1600(uint64_t* a) {
    // Keep round constants local so the same function compiles for both host and device.
    constexpr uint64_t kRc[24] = {
        1ULL,
        0x8082ULL,
        0x800000000000808aULL,
        0x8000000080008000ULL,
        0x808bULL,
        0x80000001ULL,
        0x8000000080008081ULL,
        0x8000000000008009ULL,
        0x8aULL,
        0x88ULL,
        0x80008009ULL,
        0x8000000aULL,
        0x8000808bULL,
        0x800000000000008bULL,
        0x8000000000008089ULL,
        0x8000000000008003ULL,
        0x8000000000008002ULL,
        0x8000000000000080ULL,
        0x800aULL,
        0x800000008000000aULL,
        0x8000000080008081ULL,
        0x8000000000008080ULL,
        0x80000001ULL,
        0x8000000080008008ULL,
    };

    constexpr unsigned kRho[24] = {
        1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14, 27, 41, 56, 8,
        25, 43, 62, 18, 39, 61, 20, 44,
    };

    constexpr size_t kPi[24] = {
        10, 7,  11, 17, 18, 3,  5,  16, 8,  21, 24, 4,  15, 23, 19, 13,
        12, 2,  20, 14, 22, 9,  6,  1,
    };

    for (unsigned rnd = 0; rnd < 24; ++rnd) {
        uint64_t c[5]{};
        for (unsigned x = 0; x < 5; ++x) {
            for (unsigned yc = 0; yc < 5; ++yc) {
                c[x] ^= a[x + yc * 5];
            }
        }
        for (unsigned x = 0; x < 5; ++x) {
            const uint64_t d = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
            for (unsigned yc = 0; yc < 5; ++yc) {
                a[yc * 5 + x] ^= d;
            }
        }

        uint64_t last = a[1];
        for (unsigned t = 0; t < 24; ++t) {
            const size_t idx = kPi[t];
            const uint64_t tmp = a[idx];
            a[idx] = rotl64(last, kRho[t]);
            last = tmp;
        }

        for (unsigned y_step = 0; y_step < 5; ++y_step) {
            const unsigned y = y_step * 5;
            uint64_t row[5];
            for (unsigned x = 0; x < 5; ++x) {
                row[x] = a[y + x];
            }
            for (unsigned x = 0; x < 5; ++x) {
                a[y + x] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
            }
        }

        a[0] ^= kRc[rnd];
    }
}

} // namespace keccak_detail

constexpr size_t KECCAK_STATE_LANES = 25;

/**
 * @brief Keccak-f[1600] over 25 × 64-bit lanes (Plonky3 / tiny-keccak layout).
 */
struct KeccakF {
    void permute_mut(std::array<uint64_t, KECCAK_STATE_LANES>& state) const {
        keccak_detail::keccak_f1600(state.data());
    }
};

/**
 * @brief Keccak-256 over bytes (Ethereum-compatible), 32-byte digest.
 */
struct Keccak256Hash {
    using Digest = std::array<uint8_t, 32>;

    Digest hash_iter(const std::vector<uint8_t>& data) const {
        return hash_iter(data.data(), data.size());
    }

    Digest hash_iter(const uint8_t* data, size_t len) const {
        std::array<uint64_t, KECCAK_STATE_LANES> words{};
        uint8_t* const bytes = reinterpret_cast<uint8_t*>(words.data());
        constexpr size_t kRateBytes = 136;
        size_t offset = 0;

        while (len >= kRateBytes - offset) {
            const size_t chunk = kRateBytes - offset;
            for (size_t i = 0; i < chunk; ++i) {
                bytes[offset + i] ^= data[i];
            }
            data += chunk;
            len -= chunk;
            keccak_detail::keccak_f1600(words.data());
            offset = 0;
        }
        for (size_t i = 0; i < len; ++i) {
            bytes[offset + i] ^= data[i];
        }
        offset += len;

        bytes[offset] ^= 0x01;
        bytes[kRateBytes - 1] ^= 0x80;
        keccak_detail::keccak_f1600(words.data());

        Digest out{};
        std::memcpy(out.data(), bytes, 32);
        return out;
    }

    Digest hash_iter_slices(const std::vector<std::vector<uint8_t>>& slices) const {
        std::array<uint64_t, KECCAK_STATE_LANES> words{};
        uint8_t* const bytes = reinterpret_cast<uint8_t*>(words.data());
        constexpr size_t kRateBytes = 136;
        size_t offset = 0;

        for (const auto& slice : slices) {
            for (uint8_t byte : slice) {
                bytes[offset++] ^= byte;
                if (offset == kRateBytes) {
                    keccak_detail::keccak_f1600(words.data());
                    offset = 0;
                }
            }
        }

        bytes[offset] ^= 0x01;
        bytes[kRateBytes - 1] ^= 0x80;
        keccak_detail::keccak_f1600(words.data());

        Digest out{};
        std::memcpy(out.data(), bytes, 32);
        return out;
    }
};

} // namespace p3_symmetric

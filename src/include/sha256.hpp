#pragma once

/**
 * @file sha256.hpp
 * @brief SHA-256 hasher and 2-to-1 padding-free SHA-256 compression.
 *
 * Mirrors plonky3/sha256/src/lib.rs:
 *  - Sha256: standard SHA-256 hash with FIPS 180-4 padding.
 *  - Sha256Compress: compression of two 32-byte blocks into one 32-byte digest
 *    (single SHA-256 compression block with IV).
 */

#include "cuda_compat.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace p3_symmetric {

/** FIPS 180-4 round constants K_0..K_63 (matches k_local in compress_block for CUDA). */
inline constexpr uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

namespace sha256_detail {

inline constexpr uint32_t H256_256_RAW[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

P3_HOST_DEVICE P3_INLINE uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

P3_HOST_DEVICE P3_INLINE uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

P3_HOST_DEVICE P3_INLINE uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

P3_HOST_DEVICE P3_INLINE uint32_t big_sigma0(uint32_t x) {
    return rotr32(x, 2u) ^ rotr32(x, 13u) ^ rotr32(x, 22u);
}

P3_HOST_DEVICE P3_INLINE uint32_t big_sigma1(uint32_t x) {
    return rotr32(x, 6u) ^ rotr32(x, 11u) ^ rotr32(x, 25u);
}

P3_HOST_DEVICE P3_INLINE uint32_t small_sigma0(uint32_t x) {
    return rotr32(x, 7u) ^ rotr32(x, 18u) ^ (x >> 3u);
}

P3_HOST_DEVICE P3_INLINE uint32_t small_sigma1(uint32_t x) {
    return rotr32(x, 17u) ^ rotr32(x, 19u) ^ (x >> 10u);
}

P3_HOST_DEVICE P3_INLINE uint32_t load_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24u) |
           (static_cast<uint32_t>(p[1]) << 16u) |
           (static_cast<uint32_t>(p[2]) << 8u) |
            static_cast<uint32_t>(p[3]);
}

P3_HOST_DEVICE P3_INLINE void store_be32(uint32_t x, uint8_t* out) {
    out[0] = static_cast<uint8_t>(x >> 24u);
    out[1] = static_cast<uint8_t>(x >> 16u);
    out[2] = static_cast<uint8_t>(x >> 8u);
    out[3] = static_cast<uint8_t>(x);
}

P3_HOST_DEVICE P3_INLINE void compress_block(uint32_t state[8], const uint8_t block[64]) {
    // Local copy: nvcc does not treat namespace-scope constexpr arrays as device symbols.
    const uint32_t k_local[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    uint32_t w[64];
    for (size_t i = 0; i < 16; ++i) {
        w[i] = load_be32(block + 4 * i);
    }
    for (size_t i = 16; i < 64; ++i) {
        w[i] = small_sigma1(w[i - 2]) + w[i - 7] + small_sigma0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (size_t i = 0; i < 64; ++i) {
        const uint32_t t1 = h + big_sigma1(e) + ch(e, f, g) + k_local[i] + w[i];
        const uint32_t t2 = big_sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

P3_HOST_DEVICE P3_INLINE void compress_two_to_one_raw(
    const uint8_t left[32],
    const uint8_t right[32],
    uint8_t out[32]) {
    const uint32_t h_init[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    uint8_t block[64];
    for (size_t i = 0; i < 32; ++i) {
        block[i] = left[i];
        block[32 + i] = right[i];
    }

    uint32_t state[8];
    for (size_t i = 0; i < 8; ++i) {
        state[i] = h_init[i];
    }
    compress_block(state, block);

    for (size_t i = 0; i < 8; ++i) {
        store_be32(state[i], out + 4 * i);
    }
}

P3_INLINE std::array<uint8_t, 32> compress_two_to_one(
    const std::array<uint8_t, 32>& left,
    const std::array<uint8_t, 32>& right) {
    std::array<uint8_t, 32> out{};
    compress_two_to_one_raw(left.data(), right.data(), out.data());
    return out;
}

class Sha256Ctx {
public:
    P3_INLINE Sha256Ctx() : total_len_(0), buf_len_(0) {
        for (size_t i = 0; i < 8; ++i) {
            state_[i] = H256_256_RAW[i];
        }
    }

    P3_INLINE void update(const uint8_t* data, size_t len) {
        total_len_ += static_cast<uint64_t>(len);
        size_t pos = 0;
        while (pos < len) {
            const size_t take = ((64u - buf_len_) < (len - pos)) ? (64u - buf_len_) : (len - pos);
            for (size_t i = 0; i < take; ++i) {
                buffer_[buf_len_ + i] = data[pos + i];
            }
            buf_len_ += take;
            pos += take;
            if (buf_len_ == 64u) {
                compress_block(state_.data(), buffer_.data());
                buf_len_ = 0;
            }
        }
    }

    P3_INLINE std::array<uint8_t, 32> finalize() {
        buffer_[buf_len_++] = 0x80u;

        if (buf_len_ > 56u) {
            while (buf_len_ < 64u) {
                buffer_[buf_len_++] = 0u;
            }
            compress_block(state_.data(), buffer_.data());
            buf_len_ = 0;
        }

        while (buf_len_ < 56u) {
            buffer_[buf_len_++] = 0u;
        }

        const uint64_t bit_len = total_len_ * 8u;
        for (size_t i = 0; i < 8; ++i) {
            buffer_[56u + i] = static_cast<uint8_t>(bit_len >> (56u - 8u * i));
        }
        compress_block(state_.data(), buffer_.data());

        std::array<uint8_t, 32> out{};
        for (size_t i = 0; i < 8; ++i) {
            store_be32(state_[i], out.data() + 4 * i);
        }
        return out;
    }

private:
    std::array<uint32_t, 8> state_;
    uint64_t total_len_;
    std::array<uint8_t, 64> buffer_{};
    size_t buf_len_;
};

} // namespace sha256_detail

inline constexpr std::array<uint32_t, 8> H256_256 = {
    sha256_detail::H256_256_RAW[0], sha256_detail::H256_256_RAW[1],
    sha256_detail::H256_256_RAW[2], sha256_detail::H256_256_RAW[3],
    sha256_detail::H256_256_RAW[4], sha256_detail::H256_256_RAW[5],
    sha256_detail::H256_256_RAW[6], sha256_detail::H256_256_RAW[7],
};

/**
 * @brief Standard SHA-256 hasher.
 */
struct Sha256 {
    using Digest = std::array<uint8_t, 32>;

    P3_INLINE Digest hash_iter(const uint8_t* data, size_t len) const {
        sha256_detail::Sha256Ctx ctx;
        if (len > 0) {
            ctx.update(data, len);
        }
        return ctx.finalize();
    }

    Digest hash_iter(const std::vector<uint8_t>& data) const {
        return hash_iter(data.data(), data.size());
    }

    Digest hash_iter_slices(const std::vector<std::vector<uint8_t>>& slices) const {
        sha256_detail::Sha256Ctx ctx;
        for (const auto& s : slices) {
            if (!s.empty()) {
                ctx.update(s.data(), s.size());
            }
        }
        return ctx.finalize();
    }
};

/**
 * @brief Padding-free 2-to-1 SHA-256 compression (single 64-byte block).
 */
struct Sha256Compress {
    using Digest = std::array<uint8_t, 32>;

    P3_INLINE Digest compress(
        const std::array<Digest, 2>& input) const {
        return sha256_detail::compress_two_to_one(input[0], input[1]);
    }

    P3_HOST_DEVICE P3_INLINE void compress_raw(
        const uint8_t left[32],
        const uint8_t right[32],
        uint8_t out[32]) const {
        sha256_detail::compress_two_to_one_raw(left, right, out);
    }

};

} // namespace p3_symmetric

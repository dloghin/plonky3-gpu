#pragma once

#include "hash.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace p3_symmetric {

template<typename F, typename Inner, size_t DIGEST_BYTES, size_t OUT>
class SerializingHasher {
    static constexpr size_t byte_width = sizeof(decltype(F::PRIME));
    static_assert(byte_width > 0 && byte_width <= sizeof(uint64_t),
        "SerializingHasher: field byte width must be 1..8 (integral PRIME limb serialization)");

public:
    explicit SerializingHasher(Inner inner) : inner_(std::move(inner)) {}

    /// Serialize field elements to bytes, hash with inner hasher, decode output
    /// back to field elements.  Matches Rust SerializingHasher which simply
    /// delegates F::into_byte_stream → inner.hash_iter.
    Hash<F, OUT> hash_iter(const std::vector<F>& input) const {
        std::vector<uint8_t> bytes;
        bytes.reserve(input.size() * byte_width);
        for (const auto& v : input) {
            const uint64_t x = v.as_canonical_u64();
            for (size_t i = 0; i < byte_width; ++i) {
                bytes.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xffu));
            }
        }
        const auto digest = inner_.hash_iter(bytes);
        return decode_to_field(digest);
    }

    Hash<F, OUT> hash_iter_slices(const std::vector<std::vector<F>>& slices) const {
        std::vector<uint8_t> bytes;
        size_t total_elems = 0;
        for (const auto& s : slices) {
            total_elems += s.size();
        }
        bytes.reserve(total_elems * byte_width);
    
        for (const auto& s : slices) {
            for (const auto& v : s) {
                const uint64_t x = v.as_canonical_u64();
                for (size_t i = 0; i < byte_width; ++i) {
                    bytes.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xffu));
                }
            }
        }
        const auto digest = inner_.hash_iter(bytes);
        return decode_to_field(digest);
    }

private:
    Inner inner_;

    static Hash<F, OUT> decode_to_field(
        const std::array<uint8_t, DIGEST_BYTES>& digest)
    {
        static_assert(DIGEST_BYTES >= OUT * byte_width,
            "DIGEST_BYTES must be >= OUT * field byte width");
        Hash<F, OUT> out{};
        for (size_t i = 0; i < OUT; ++i) {
            const size_t k = i * byte_width;
            uint64_t x = 0;
            for (size_t j = 0; j < byte_width; ++j) {
                x |= static_cast<uint64_t>(digest[k + j]) << (8 * j);
            }
            if constexpr (byte_width <= 4) {
                out[i] = F(static_cast<uint32_t>(x));
            } else {
                out[i] = F(static_cast<uint64_t>(x));
            }
        }
        return out;
    }
};

} // namespace p3_symmetric

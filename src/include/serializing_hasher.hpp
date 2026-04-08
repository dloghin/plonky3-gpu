#pragma once

#include "hash.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace p3_symmetric {

template<typename F, typename Inner, size_t DIGEST_BYTES, size_t OUT>
class SerializingHasher {
public:
    explicit SerializingHasher(Inner inner) : inner_(std::move(inner)) {}

    /// Serialize field elements to bytes, hash with inner hasher, decode output
    /// back to field elements.  Matches Rust SerializingHasher which simply
    /// delegates F::into_byte_stream → inner.hash_iter.
    Hash<F, OUT> hash_iter(const std::vector<F>& input) const {
        std::vector<uint8_t> bytes;
        bytes.reserve(input.size() * 4);
        for (const auto& v : input) {
            const uint64_t x = v.as_canonical_u64();
            for (size_t i = 0; i < sizeof(decltype(F::PRIME)); ++i) {
                bytes.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xffu));
            }
        }
        const auto digest = inner_.hash_iter(bytes);
        return decode_to_field(digest);
    }

    Hash<F, OUT> hash_iter_slices(const std::vector<std::vector<F>>& slices) const {
        std::vector<F> flat;
        size_t total = 0;
        for (const auto& s : slices) total += s.size();
        flat.reserve(total);
        for (const auto& s : slices) flat.insert(flat.end(), s.begin(), s.end());
        return hash_iter(flat);
    }

private:
    Inner inner_;

    static Hash<F, OUT> decode_to_field(
        const std::array<uint8_t, DIGEST_BYTES>& digest)
    {
        static_assert(DIGEST_BYTES >= OUT * 4,
            "DIGEST_BYTES must be >= OUT * sizeof(uint32_t)");
        Hash<F, OUT> out{};
        for (size_t i = 0; i < OUT; ++i) {
            const size_t k = i * 4;
            const uint32_t x =
                static_cast<uint32_t>(digest[k]) |
                (static_cast<uint32_t>(digest[k + 1]) << 8) |
                (static_cast<uint32_t>(digest[k + 2]) << 16) |
                (static_cast<uint32_t>(digest[k + 3]) << 24);
            out[i] = F(x);
        }
        return out;
    }
};

} // namespace p3_symmetric

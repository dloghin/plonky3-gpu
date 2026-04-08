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

    Hash<F, OUT> hash_iter(const std::vector<F>& input) const {
        std::vector<uint8_t> bytes;
        bytes.reserve(input.size() * 4);
        for (const auto& v : input) {
            const uint32_t x = static_cast<uint32_t>(v.as_canonical_u64());
            bytes.push_back(static_cast<uint8_t>(x & 0xffu));
            bytes.push_back(static_cast<uint8_t>((x >> 8) & 0xffu));
            bytes.push_back(static_cast<uint8_t>((x >> 16) & 0xffu));
            bytes.push_back(static_cast<uint8_t>((x >> 24) & 0xffu));
        }
        return decode_to_field(expand_digest(bytes));
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

    std::vector<uint8_t> expand_digest(const std::vector<uint8_t>& input_bytes) const {
        std::vector<uint8_t> stream;
        stream.reserve(OUT * 4);

        std::array<uint8_t, DIGEST_BYTES> block = inner_.hash_iter(input_bytes);
        while (stream.size() < OUT * 4) {
            stream.insert(stream.end(), block.begin(), block.end());
            std::vector<uint8_t> block_vec(block.begin(), block.end());
            block = inner_.hash_iter(block_vec);
        }
        stream.resize(OUT * 4);
        return stream;
    }

    static Hash<F, OUT> decode_to_field(const std::vector<uint8_t>& bytes) {
        Hash<F, OUT> out{};
        for (size_t i = 0; i < OUT; ++i) {
            const size_t k = i * 4;
            const uint32_t x =
                static_cast<uint32_t>(bytes[k]) |
                (static_cast<uint32_t>(bytes[k + 1]) << 8) |
                (static_cast<uint32_t>(bytes[k + 2]) << 16) |
                (static_cast<uint32_t>(bytes[k + 3]) << 24);
            out[i] = F(x);
        }
        return out;
    }
};

} // namespace p3_symmetric

#pragma once

#include "challenger_traits.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace p3_challenger {

template<typename F>
struct FieldBytesCodec {
    static constexpr size_t byte_width = sizeof(decltype(F::PRIME));
    static constexpr uint64_t modulus_u64 = static_cast<uint64_t>(F::PRIME);

    static std::array<uint8_t, byte_width> to_bytes(const F& v) {
        std::array<uint8_t, byte_width> out{};
        uint64_t x = v.as_canonical_u64();
        for (size_t i = 0; i < byte_width; ++i) {
            out[i] = static_cast<uint8_t>((x >> (8 * i)) & 0xffu);
        }
        return out;
    }

    static bool from_bytes(const std::array<uint8_t, byte_width>& bytes, F& out) {
        uint64_t x = 0;
        for (size_t i = 0; i < byte_width; ++i) {
            x |= (static_cast<uint64_t>(bytes[i]) << (8 * i));
        }
        if (x >= modulus_u64) return false;

        if constexpr (byte_width <= 4) {
            out = F(static_cast<uint32_t>(x));
        } else {
            out = F(static_cast<uint64_t>(x));
        }
        return true;
    }
};

template<typename F, typename Inner, typename Codec = FieldBytesCodec<F>>
class SerializingChallenger {
public:
    explicit SerializingChallenger(Inner inner) : inner_(std::move(inner)) {}

    void observe(const F& value) {
        const auto bytes = Codec::to_bytes(value);
        for (uint8_t b : bytes) inner_.observe(b);
    }

    F sample() {
        for (;;) {
            std::array<uint8_t, Codec::byte_width> bytes{};
            for (size_t i = 0; i < Codec::byte_width; ++i) {
                bytes[i] = inner_.sample();
            }
            F out{};
            if (Codec::from_bytes(bytes, out)) return out;
        }
    }

    /**
     * @brief Sample `bits` bits into a `size_t` using the byte-oriented inner RNG.
     *
     * Draws fresh bytes from `inner_.sample()` until `bits` bits are filled.
     * Unlike masking a single `sample()` field element, this does not cap entropy
     * at the field bit-width (e.g. 31 bits for BabyBear when 64 bits are requested).
     */
    size_t sample_bits(size_t bits) {
        assert(bits <= sizeof(size_t) * 8);
        if (bits == 0) return 0;
        size_t result = 0;
        size_t filled = 0;
        while (filled < bits) {
            const uint8_t b = inner_.sample();
            const size_t need = bits - filled;
            const size_t take = need < 8 ? need : 8;
            const size_t mask = (static_cast<size_t>(1) << take) - 1u;
            result |= (static_cast<size_t>(b) & mask) << filled;
            filled += take;
        }
        return result;
    }

    Inner& inner() { return inner_; }
    const Inner& inner() const { return inner_; }

private:
    Inner inner_;
};

} // namespace p3_challenger

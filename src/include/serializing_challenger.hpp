#pragma once

#include "challenger_traits.hpp"

#include <array>
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

    size_t sample_bits(size_t bits) {
        if (bits == 0) return 0;
        const F v = sample();
        const uint64_t raw = v.as_canonical_u64();
        const uint64_t mask = bits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1u);
        return static_cast<size_t>(raw & mask);
    }

    Inner& inner() { return inner_; }
    const Inner& inner() const { return inner_; }

private:
    Inner inner_;
};

} // namespace p3_challenger

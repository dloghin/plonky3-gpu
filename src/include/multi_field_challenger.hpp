#pragma once

#include "challenger_traits.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace p3_challenger {

namespace detail {

template<typename T, typename = void>
struct has_field_bits : std::false_type {};

template<typename T>
struct has_field_bits<T, std::void_t<decltype(T::FIELD_BITS)>> : std::true_type {};

} // namespace detail

template<typename F, typename Inner, size_t WIDTH>
class MultiFieldChallenger {
    static_assert(WIDTH > 0 && WIDTH <= 64, "WIDTH must be in [1, 64]");
    static_assert(detail::has_field_bits<F>::value,
        "MultiFieldChallenger requires F::FIELD_BITS so lane width can be checked");

    static constexpr size_t FIELD_BITS = F::FIELD_BITS;
    static constexpr size_t BITS_PER_ELM = 64 / WIDTH;
    static_assert(BITS_PER_ELM > 0, "WIDTH too large for 64-bit packing");
    static_assert(
        FIELD_BITS <= BITS_PER_ELM,
        "WIDTH too large: 64/WIDTH bits per lane is smaller than F::FIELD_BITS (overlapping or truncated packing)");

    static constexpr uint64_t ELM_MASK = BITS_PER_ELM >= 64
        ? ~uint64_t(0)
        : (uint64_t(1) << BITS_PER_ELM) - 1;

    static_assert(BITS_PER_ELM * WIDTH <= 64, "packed lanes must not exceed 64 bits");

public:
    explicit MultiFieldChallenger(Inner inner) : inner_(std::move(inner)) {}

    void observe(const F& value) {
        unpacked_output_.clear();
        pending_.push_back(value);
        if (pending_.size() == WIDTH) {
            flush_pending_chunk();
        }
    }

    F sample() {
        if (unpacked_output_.empty()) {
            if (!pending_.empty()) {
                while (pending_.size() < WIDTH) pending_.push_back(F::zero_val());
                flush_pending_chunk();
            }
            unpack_from_inner();
        }
        F out = unpacked_output_.back();
        unpacked_output_.pop_back();
        return out;
    }

private:
    Inner inner_;
    std::vector<F> pending_;
    std::vector<F> unpacked_output_;

    static uint64_t pack_chunk(const std::vector<F>& chunk) {
        uint64_t packed = 0;
        for (size_t i = 0; i < WIDTH; ++i) {
            const uint64_t val = chunk[i].as_canonical_u64();
            packed |= ((val & ELM_MASK) << (BITS_PER_ELM * i));
        }
        return packed;
    }

    static std::array<F, WIDTH> unpack_chunk(uint64_t packed) {
        std::array<F, WIDTH> out{};
        for (size_t i = 0; i < WIDTH; ++i) {
            const uint64_t limb = (packed >> (BITS_PER_ELM * i)) & ELM_MASK;
            if constexpr (BITS_PER_ELM <= 32) {
                out[i] = F(static_cast<uint32_t>(limb));
            } else {
                out[i] = F(static_cast<uint64_t>(limb));
            }
        }
        return out;
    }

    void flush_pending_chunk() {
        const uint64_t packed = pack_chunk(pending_);
        inner_.observe(packed);
        pending_.clear();
    }

    void unpack_from_inner() {
        const uint64_t packed = inner_.sample();
        const auto out = unpack_chunk(packed);
        unpacked_output_.assign(out.begin(), out.end());
    }
};

} // namespace p3_challenger

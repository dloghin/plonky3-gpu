#pragma once

#include "challenger_traits.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace p3_challenger {

template<typename F, typename Inner, size_t WIDTH>
class MultiFieldChallenger {
    static_assert(WIDTH > 0, "WIDTH must be > 0");

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

    static constexpr size_t BITS_PER_ELM = 64 / WIDTH;
    static constexpr uint64_t ELM_MASK = BITS_PER_ELM >= 64
        ? ~uint64_t(0)
        : (uint64_t(1) << BITS_PER_ELM) - 1;

    static uint64_t pack_chunk(const std::vector<F>& chunk) {
        uint64_t packed = 0;
        for (size_t i = 0; i < WIDTH; ++i) {
            const uint64_t val = chunk[i].as_canonical_u64();
            packed |= (val << (BITS_PER_ELM * i));
        }
        return packed;
    }

    static std::array<F, WIDTH> unpack_chunk(uint64_t packed) {
        std::array<F, WIDTH> out{};
        for (size_t i = 0; i < WIDTH; ++i) {
            const uint32_t limb = static_cast<uint32_t>((packed >> (BITS_PER_ELM * i)) & ELM_MASK);
            out[i] = F(limb);
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

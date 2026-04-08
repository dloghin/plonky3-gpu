#pragma once

#include "hash.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace p3_symmetric {

template<typename F, typename Perm, size_t WIDTH, size_t RATE, size_t OUT>
class MultiField32PaddingFreeSponge {
    static_assert(RATE < WIDTH, "RATE must be < WIDTH to ensure non-zero capacity");
    static_assert(sizeof(decltype(F::PRIME)) <= 4, "MultiField32PaddingFreeSponge only supports fields with up to 32-bit prime modulus");    
    static_assert(std::is_invocable_v<decltype(&Perm::permute_mut), const Perm&, std::array<uint64_t, WIDTH>&>,
        "Perm::permute_mut must be callable on const Perm with std::array<uint64_t, WIDTH>&");

public:
    explicit MultiField32PaddingFreeSponge(Perm perm) : permutation_(std::move(perm)) {}

    Hash<F, OUT> hash_iter(const std::vector<F>& input) const {
        std::array<uint64_t, WIDTH> state{};
        size_t pos = 0;

        for (;;) {
            size_t consumed = 0;
            for (size_t lane = 0; lane < RATE; ++lane) {
                if (pos >= input.size()) break;
                const uint32_t lo = static_cast<uint32_t>(input[pos++].as_canonical_u64());
                uint32_t hi = 0u;
                ++consumed;
                if (pos < input.size()) {
                    hi = static_cast<uint32_t>(input[pos++].as_canonical_u64());
                    ++consumed;
                }
                state[lane] = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
            }

            if (consumed == 0) break;
            permutation_.permute_mut(state);
            if (pos >= input.size()) break;
        }

        Hash<F, OUT> out{};
        size_t out_idx = 0;
        for (size_t lane = 0; lane < RATE && out_idx < OUT; ++lane) {
            const uint32_t lo = static_cast<uint32_t>(state[lane] & 0xffffffffULL);
            out[out_idx++] = F(lo);
            if (out_idx < OUT) {
                const uint32_t hi = static_cast<uint32_t>(state[lane] >> 32);
                out[out_idx++] = F(hi);
            }
        }
        return out;
    }

private:
    Perm permutation_;
};

} // namespace p3_symmetric

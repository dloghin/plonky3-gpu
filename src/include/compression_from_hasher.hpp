#pragma once

#include "hash.hpp"

#include <array>
#include <vector>

namespace p3_symmetric {

template<typename Hasher, typename F, size_t N, size_t CHUNK>
class CompressionFunctionFromHasher {
public:
    explicit CompressionFunctionFromHasher(Hasher hasher) : hasher_(std::move(hasher)) {}

    std::array<F, CHUNK> compress(const std::array<std::array<F, CHUNK>, N>& input) const {
        std::vector<F> flat;
        flat.reserve(N * CHUNK);
        for (size_t i = 0; i < N; ++i) {
            flat.insert(flat.end(), input[i].begin(), input[i].end());
        }
        return hasher_.hash_iter(flat);
    }

private:
    Hasher hasher_;
};

} // namespace p3_symmetric

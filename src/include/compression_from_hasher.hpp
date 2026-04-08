#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace p3_symmetric {

template<typename Hasher, typename F, size_t N, size_t CHUNK>
class CompressionFunctionFromHasher {
    static_assert(N > 0 && CHUNK > 0, "CompressionFunctionFromHasher requires N > 0 and CHUNK > 0");

public:
    explicit CompressionFunctionFromHasher(Hasher hasher) : hasher_(std::move(hasher)) {}

    std::array<F, CHUNK> compress(const std::array<std::array<F, CHUNK>, N>& input) const {
        const F* flat_ptr = &input[0][0];
        return hash_flat(flat_ptr, N * CHUNK);
    }

private:
    template<typename H = Hasher>
    static auto hash_flat_impl(const H& hasher, const F* data, size_t len, int)
        -> decltype(hasher.hash_iter(data, len))
    {
        return hasher.hash_iter(data, len);
    }

    template<typename H = Hasher>
    static std::array<F, CHUNK> hash_flat_impl(const H& hasher, const F* data, size_t len, long) {
        std::vector<F> flat(data, data + len);
        return hasher.hash_iter(flat);
    }

    std::array<F, CHUNK> hash_flat(const F* data, size_t len) const {
        return hash_flat_impl(hasher_, data, len, 0);
    }

    Hasher hasher_;
};

} // namespace p3_symmetric

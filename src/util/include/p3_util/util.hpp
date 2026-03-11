#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <functional>
#include <array>

namespace p3_util {

// ---------------------------------------------------------------------------
// Bit manipulation
// ---------------------------------------------------------------------------

/// Computes log2(n). Asserts (throws) if n is not a power of two.
/// Mirrors Rust's p3_util::log2_strict_usize.
inline size_t log2_strict_usize(size_t n) {
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("log2_strict_usize: value must be a non-zero power of 2");
    }
    // Use __builtin_ctzll (count trailing zeros) when available (GCC/Clang/NVCC).
    return static_cast<size_t>(__builtin_ctzll(static_cast<unsigned long long>(n)));
}

/// Computes ceil(log2(n)).  log2_ceil_usize(1) == 0, log2_ceil_usize(0) == 0.
/// Mirrors Rust's p3_util::log2_ceil_usize.
inline size_t log2_ceil_usize(size_t n) {
    if (n <= 1) return 0;
    // __builtin_clzll counts leading zeros; for a 64-bit word:
    //   floor(log2(n)) = 63 - clz(n)
    //   ceil(log2(n))  = 64 - clz(n-1)
    return static_cast<size_t>(64 - __builtin_clzll(static_cast<unsigned long long>(n - 1)));
}

/// Reverses the lowest bit_len bits of x.
/// E.g. reverse_bits_len(0b01011, 5) == 0b11010.
/// Mirrors Rust's p3_util::reverse_bits_len.
inline size_t reverse_bits_len(size_t x, size_t bit_len) {
    if (bit_len == 0) return 0;
    size_t result = 0;
    size_t tmp = x;
    for (size_t i = 0; i < bit_len; ++i) {
        result = (result << 1) | (tmp & 1);
        tmp >>= 1;
    }
    return result;
}

// ---------------------------------------------------------------------------
// reverse_slice_index_bits
// ---------------------------------------------------------------------------

namespace detail {

/// In-place square-matrix transpose (used for the cache-friendly block algorithm).
/// `data` is a square block of side `n` stored in row-major order inside a
/// larger buffer with `stride` elements per row.
template<typename T>
void transpose_square_block(T* data, size_t n, size_t stride) {
    for (size_t r = 0; r < n; ++r) {
        for (size_t c = r + 1; c < n; ++c) {
            std::swap(data[r * stride + c], data[c * stride + r]);
        }
    }
}

/// Cache-friendly in-place bit-reversal permutation for large power-of-two arrays.
/// Uses the block-transpose algorithm from Plonky3's transpose.rs.
///
/// The idea: split the index bits into high and low halves.  Bit-reversing an
/// index swaps the high half (now low) and the low half (now high).  This is
/// equivalent to transposing a 2D array of shape [2^hi_bits × 2^lo_bits].
template<typename T>
void reverse_slice_index_bits_large(std::vector<T>& vals) {
    size_t n = vals.size();
    if (n <= 1) return;

    size_t log_n = static_cast<size_t>(__builtin_ctzll(static_cast<unsigned long long>(n)));
    size_t lo_bits = log_n / 2;
    size_t hi_bits = log_n - lo_bits;
    size_t lo_size = static_cast<size_t>(1) << lo_bits; // number of "columns"
    size_t hi_size = static_cast<size_t>(1) << hi_bits; // number of "rows"

    // Step 1: transpose the [hi_size × lo_size] matrix stored in vals.
    //         After this, index bits are reversed within each half.
    // For a non-square matrix we do it with repeated swaps.
    if (hi_size == lo_size) {
        // Square: in-place transpose with a single pass.
        transpose_square_block(vals.data(), hi_size, lo_size);
    } else {
        // Rectangular (hi_size = 2 * lo_size): use out-of-place transpose.
        // Allocate a temporary copy and fill transposed.
        std::vector<T> tmp(n);
        for (size_t r = 0; r < hi_size; ++r) {
            for (size_t c = 0; c < lo_size; ++c) {
                tmp[c * hi_size + r] = vals[r * lo_size + c];
            }
        }
        vals = std::move(tmp);
    }

    // Step 2: bit-reverse the rows (now of length hi_size) using simple swaps.
    for (size_t i = 0; i < lo_size; ++i) {
        size_t j = reverse_bits_len(i, lo_bits);
        if (i < j) {
            T* row_i = vals.data() + i * hi_size;
            T* row_j = vals.data() + j * hi_size;
            for (size_t k = 0; k < hi_size; ++k) {
                std::swap(row_i[k], row_j[k]);
            }
        }
    }

    // Step 3: bit-reverse the columns (now of length lo_size) using simple swaps.
    for (size_t i = 0; i < hi_size; ++i) {
        size_t j = reverse_bits_len(i, hi_bits);
        if (i < j) {
            for (size_t k = 0; k < lo_size; ++k) {
                std::swap(vals[i + k * hi_size], vals[j + k * hi_size]);
            }
        }
    }
}

} // namespace detail

/// Permutes vals so that each index i is mapped to reverse_bits_len(i, log2(n)).
/// vals.size() must be a power of two (or zero).
/// Mirrors Rust's p3_util::reverse_slice_index_bits.
template<typename T>
void reverse_slice_index_bits(std::vector<T>& vals) {
    size_t n = vals.size();
    if (n == 0 || n == 1) return;

    if (n != 0 && (n & (n - 1)) != 0) {
        throw std::invalid_argument("reverse_slice_index_bits: size must be a power of 2");
    }

    size_t log_n = log2_strict_usize(n);

    // For small arrays use the simple O(n) swap loop.
    // Threshold matches Plonky3's approach (use block algo for log_n >= 4).
    if (log_n < 4) {
        for (size_t i = 0; i < n; ++i) {
            size_t j = reverse_bits_len(i, log_n);
            if (i < j) {
                std::swap(vals[i], vals[j]);
            }
        }
    } else {
        detail::reverse_slice_index_bits_large(vals);
    }
}

// ---------------------------------------------------------------------------
// Iterator / chunk utilities
// ---------------------------------------------------------------------------

/// Reads `input` in chunks of BUFLEN and calls func(chunk_span, chunk_len)
/// for each chunk.  The last chunk may be shorter than BUFLEN.
/// Mirrors Rust's p3_util::apply_to_chunks.
template<size_t BUFLEN, typename T, typename Func>
void apply_to_chunks(const std::vector<T>& input, Func func) {
    size_t offset = 0;
    size_t total = input.size();
    while (offset < total) {
        size_t chunk_len = std::min(BUFLEN, total - offset);
        func(input.data() + offset, chunk_len);
        offset += chunk_len;
    }
}

/// Returns a vector of arrays of N elements taken from `input` in order,
/// padding the last array with `pad_val` if necessary.
/// Mirrors Rust's p3_util::iter_array_chunks_padded.
template<size_t N, typename T>
std::vector<std::array<T, N>> iter_array_chunks_padded(const std::vector<T>& input,
                                                        const T& pad_val) {
    std::vector<std::array<T, N>> result;
    size_t total = input.size();
    size_t num_chunks = (total + N - 1) / N;
    result.reserve(num_chunks);

    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        std::array<T, N> arr;
        for (size_t i = 0; i < N; ++i) {
            size_t idx = chunk * N + i;
            arr[i] = (idx < total) ? input[idx] : pad_val;
        }
        result.push_back(arr);
    }
    return result;
}

} // namespace p3_util

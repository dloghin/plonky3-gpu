#pragma once

#include "dense_matrix.hpp"
#include <cmath>
#include <algorithm>

// Pull in the canonical implementations from p3_util.
// Functions reverse_bits_len and log2_strict_usize now live in that library;
// we expose them here under the p3_matrix namespace for backwards compatibility.
// The p3_util include directory is added by the CMake dependency on p3_util.
#include "p3_util/util.hpp"

namespace p3_matrix {

using p3_util::reverse_bits_len;

/// Backwards-compatible alias for p3_util::log2_strict_usize.
inline size_t log2_strict(size_t n) {
    return p3_util::log2_strict_usize(n);
}

/**
 * @brief Reverse the order of matrix rows based on bit-reversal of indices
 *
 * Given a matrix of height h = 2^k, this rearranges rows by reversing
 * the binary representation of each row index. Performed in-place.
 *
 * @tparam T Element type
 * @param mat Matrix to modify (height must be power of 2)
 * @throws std::invalid_argument if height is not a power of 2
 */
template<typename T>
void reverse_matrix_index_bits(RowMajorMatrix<T>& mat) {
    size_t h = mat.height();
    if (h == 0) return;

    size_t log_h = log2_strict(h);

    for (size_t i = 0; i < h; ++i) {
        size_t j = reverse_bits_len(i, log_h);
        if (i < j) {
            mat.swap_rows(i, j);
        }
    }
}

/**
 * @brief Append zeros to the "end" of a bit-reversed matrix
 *
 * The matrix is assumed to be in bit-reversed order. This function
 * interleaves zero rows to effectively pad the matrix.
 *
 * @tparam T Element type (must have zero value)
 * @param mat Input matrix
 * @param added_bits Number of bits to add (doubles height this many times)
 * @return New padded matrix
 */
template<typename T>
RowMajorMatrix<T> bit_reversed_zero_pad(const RowMajorMatrix<T>& mat, size_t added_bits) {
    if (added_bits == 0) {
        return mat;  // Copy constructor
    }

    size_t h = mat.height();
    size_t w = mat.width();
    size_t new_h = h << added_bits;
    size_t chunk_size = 1 << added_bits;

    RowMajorMatrix<T> padded(new_h, w, T());  // Zero-initialized

    // Copy original rows to the first position of each chunk
    for (size_t src_r = 0; src_r < h; ++src_r) {
        size_t dest_r = src_r * chunk_size;
        const T* src_row = mat.row_ptr(src_r);
        T* dest_row = padded.row_mut(dest_r);
        std::copy(src_row, src_row + w, dest_row);
    }

    return padded;
}

/**
 * @brief Compute dot product of two vectors
 * @tparam T Element type
 * @param a First vector
 * @param b Second vector
 * @return Dot product sum(a[i] * b[i])
 * @throws std::invalid_argument if vectors have different sizes
 */
template<typename T>
T dot_product(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have same size");
    }
    T result = T();
    for (size_t i = 0; i < a.size(); ++i) {
        result = result + (a[i] * b[i]);
    }
    return result;
}

/**
 * @brief Compute matrix-vector product (M * v)
 *
 * Each row of the result is the dot product of the corresponding row of M with v.
 *
 * @tparam T Element type
 * @param mat Matrix M
 * @param vec Vector v
 * @return Vector M * v
 * @throws std::invalid_argument if dimensions don't match
 */
template<typename T>
std::vector<T> matrix_vector_mul(const RowMajorMatrix<T>& mat, const std::vector<T>& vec) {
    if (vec.size() != mat.width()) {
        throw std::invalid_argument("Vector size must match matrix width");
    }

    std::vector<T> result;
    result.reserve(mat.height());

    for (size_t r = 0; r < mat.height(); ++r) {
        T sum = T();
        const T* row = mat.row_ptr(r);
        for (size_t c = 0; c < mat.width(); ++c) {
            sum = sum + (row[c] * vec[c]);
        }
        result.push_back(sum);
    }

    return result;
}

/**
 * @brief Compute column-wise dot product (M^T * v)
 *
 * This computes the dot product of each column of M with vector v,
 * equivalent to scaling each row by v[i] and summing across rows.
 *
 * @tparam T Element type
 * @tparam S Scalar type for v
 * @param mat Matrix M
 * @param v Vector of scalars (length must equal mat.height())
 * @return Vector of length mat.width()
 */
template<typename T, typename S>
std::vector<S> columnwise_dot_product(const RowMajorMatrix<T>& mat, const std::vector<S>& v) {
    if (v.size() != mat.height()) {
        throw std::invalid_argument("Vector size must match matrix height");
    }

    std::vector<S> result(mat.width(), S());

    for (size_t r = 0; r < mat.height(); ++r) {
        const T* row = mat.row_ptr(r);
        const S& scale = v[r];

        for (size_t c = 0; c < mat.width(); ++c) {
            result[c] = result[c] + (scale * row[c]);
        }
    }

    return result;
}

/**
 * @brief Matrix multiplication C = A * B
 * @tparam T Element type
 * @param a Left matrix
 * @param b Right matrix
 * @return Product matrix
 * @throws std::invalid_argument if dimensions don't match
 */
template<typename T>
RowMajorMatrix<T> matrix_multiply(const RowMajorMatrix<T>& a, const RowMajorMatrix<T>& b) {
    if (a.width() != b.height()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    size_t m = a.height();
    size_t n = b.width();
    size_t k = a.width();

    RowMajorMatrix<T> result(m, n, T());

    for (size_t i = 0; i < m; ++i) {
        const T* a_row = a.row_ptr(i);
        for (size_t j = 0; j < n; ++j) {
            T sum = T();
            for (size_t p = 0; p < k; ++p) {
                sum = sum + (a_row[p] * b.get_unchecked(p, j));
            }
            result.set_unchecked(i, j, sum);
        }
    }

    return result;
}

} // namespace p3_matrix


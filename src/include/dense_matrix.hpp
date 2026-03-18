#pragma once

#include "matrix.hpp"
#include <vector>
#include <algorithm>
#include <cstring>
#include <memory>
#include <cassert>

namespace p3_matrix {

/**
 * @brief Dense matrix stored in row-major format
 *
 * This is the main matrix implementation storing data as a contiguous
 * vector in row-major order (rows are laid out consecutively).
 *
 * @tparam T Element type
 */
template<typename T>
class RowMajorMatrix : public Matrix<T> {
public:
    std::vector<T> values;  ///< Flat buffer of matrix values in row-major order
    size_t width_;          ///< Number of columns

    /**
     * @brief Construct an empty matrix
     */
    RowMajorMatrix() : width_(0) {}

    /**
     * @brief Construct a matrix from a vector of values
     * @param vals Vector of values in row-major order
     * @param w Number of columns
     * @note vals.size() must be divisible by w
     */
    RowMajorMatrix(std::vector<T> vals, size_t w)
        : values(std::move(vals)), width_(w) {
        assert(w == 0 || values.size() % w == 0);
    }

    /**
     * @brief Construct a matrix filled with a default value
     * @param rows Number of rows
     * @param cols Number of columns
     * @param default_val Value to fill matrix with
     */
    RowMajorMatrix(size_t rows, size_t cols, const T& default_val = T())
        : values(rows * cols, default_val), width_(cols) {}

    /**
     * @brief Create a single-row matrix
     * @param row_vals Vector containing the row values
     * @return RowMajorMatrix with one row
     */
    static RowMajorMatrix new_row(std::vector<T> row_vals) {
        size_t w = row_vals.size();
        return RowMajorMatrix(std::move(row_vals), w);
    }

    /**
     * @brief Create a single-column matrix
     * @param col_vals Vector containing the column values
     * @return RowMajorMatrix with one column
     */
    static RowMajorMatrix new_col(std::vector<T> col_vals) {
        return RowMajorMatrix(std::move(col_vals), 1);
    }

    // Matrix interface implementation
    size_t width() const override { return width_; }

    size_t height() const override {
        return width_ == 0 ? 0 : values.size() / width_;
    }

    T get_unchecked(size_t r, size_t c) const override {
        return values[r * width_ + c];
    }

    const T* row_ptr(size_t r) const override {
        if (r >= height()) return nullptr;
        return &values[r * width_];
    }

    /**
     * @brief Get mutable pointer to a row
     * @param r Row index
     * @return Mutable pointer to the beginning of row r
     */
    T* row_mut(size_t r) {
        if (r >= height()) {
            throw std::out_of_range("Row index out of bounds");
        }
        return &values[r * width_];
    }

    /**
     * @brief Set element at position (r, c)
     * @param r Row index
     * @param c Column index
     * @param value Value to set
     */
    void set(size_t r, size_t c, const T& value) {
        if (r >= height() || c >= width_) {
            throw std::out_of_range("Matrix indices out of bounds");
        }
        values[r * width_ + c] = value;
    }

    /**
     * @brief Set element at position (r, c) without bounds checking
     * @param r Row index
     * @param c Column index
     * @param value Value to set
     */
    void set_unchecked(size_t r, size_t c, const T& value) {
        values[r * width_ + c] = value;
    }

    /**
     * @brief Copy data from another matrix
     * @param source Source matrix to copy from
     * @throws std::invalid_argument if dimensions don't match
     */
    void copy_from(const RowMajorMatrix<T>& source) {
        if (this->dimensions() != source.dimensions()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        values = source.values;
    }

    /**
     * @brief Split matrix into two parts at row r
     * @param r Row to split at
     * @return Pair of matrices: (rows 0..r, rows r..end)
     */
    std::pair<RowMajorMatrix<T>, RowMajorMatrix<T>> split_rows(size_t r) const {
        if (r > height()) {
            throw std::out_of_range("Split index out of bounds");
        }

        auto it = values.begin() + r * width_;
        std::vector<T> top(values.begin(), it);
        std::vector<T> bottom(it, values.end());

        return {
            RowMajorMatrix(std::move(top), width_),
            RowMajorMatrix(std::move(bottom), width_)
        };
    }

    /**
     * @brief Resize matrix to new height, filling with value if growing
     * @param new_height New height
     * @param fill Value to fill new rows with
     */
    void pad_to_height(size_t new_height, const T& fill = T()) {
        if (new_height < height()) {
            throw std::invalid_argument("new_height must be >= current height");
        }
        values.resize(width_ * new_height, fill);
    }

    /**
     * @brief Scale entire matrix by a scalar
     * @param scalar Value to multiply all elements by
     */
    template<typename Scalar>
    void scale(const Scalar& scalar) {
        for (auto& val : values) {
            val = val * scalar;
        }
    }

    /**
     * @brief Scale a single row by a scalar
     * @param r Row index
     * @param scalar Value to multiply row elements by
     */
    template<typename Scalar>
    void scale_row(size_t r, const Scalar& scalar) {
        if (r >= height()) {
            throw std::out_of_range("Row index out of bounds");
        }
        T* row = row_mut(r);
        for (size_t c = 0; c < width_; ++c) {
            row[c] = row[c] * scalar;
        }
    }

    /**
     * @brief Transpose the matrix
     * @return New matrix that is the transpose of this matrix
     */
    RowMajorMatrix<T> transpose() const {
        size_t h = height();
        size_t w = width_;
        std::vector<T> result(values.size());

        for (size_t r = 0; r < h; ++r) {
            for (size_t c = 0; c < w; ++c) {
                result[c * h + r] = values[r * w + c];
            }
        }

        return RowMajorMatrix(std::move(result), h);
    }

    /**
     * @brief Transpose matrix into a pre-allocated destination
     * @param dest Destination matrix (must have dimensions swapped)
     */
    void transpose_into(RowMajorMatrix<T>& dest) const {
        size_t h = height();
        size_t w = width_;

        if (dest.height() != w || dest.width() != h) {
            throw std::invalid_argument("Destination matrix dimensions must be transposed");
        }

        for (size_t r = 0; r < h; ++r) {
            for (size_t c = 0; c < w; ++c) {
                dest.values[c * h + r] = values[r * w + c];
            }
        }
    }

    /**
     * @brief Swap two rows
     * @param i First row index
     * @param j Second row index
     */
    void swap_rows(size_t i, size_t j) {
        if (i >= height() || j >= height()) {
            throw std::out_of_range("Row indices out of bounds");
        }
        if (i == j) return;

        T* row_i = row_mut(i);
        T* row_j = row_mut(j);

        for (size_t c = 0; c < width_; ++c) {
            std::swap(row_i[c], row_j[c]);
        }
    }

    /**
     * @brief Generate a random matrix
     * @tparam RNG Random number generator type
     * @param rng Random number generator
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Random matrix
     */
    template<typename RNG>
    static RowMajorMatrix<T> rand(RNG& rng, size_t rows, size_t cols) {
        std::vector<T> vals(rows * cols);
        for (auto& v : vals) {
            v = T(rng());
        }
        return RowMajorMatrix(std::move(vals), cols);
    }

    /**
     * @brief Get row as a slice (pointer and length)
     * @param r Row index
     * @return Pair of (pointer, length)
     */
    std::pair<const T*, size_t> row_slice(size_t r) const {
        return {row_ptr(r), width_};
    }

    /**
     * @brief Get mutable row as a slice
     * @param r Row index
     * @return Pair of (pointer, length)
     */
    std::pair<T*, size_t> row_slice_mut(size_t r) {
        return {row_mut(r), width_};
    }

    /**
     * @brief Equality comparison
     */
    bool operator==(const RowMajorMatrix<T>& other) const {
        return width_ == other.width_ && values == other.values;
    }

    bool operator!=(const RowMajorMatrix<T>& other) const {
        return !(*this == other);
    }
};

} // namespace p3_matrix


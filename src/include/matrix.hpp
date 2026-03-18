#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>

namespace p3_matrix {

/**
 * @brief Dimensions of a matrix
 *
 * Stores the width (columns) and height (rows) of a matrix.
 */
struct Dimensions {
    size_t width;   ///< Number of columns
    size_t height;  ///< Number of rows

    constexpr Dimensions() : width(0), height(0) {}
    constexpr Dimensions(size_t w, size_t h) : width(w), height(h) {}

    bool operator==(const Dimensions& other) const {
        return width == other.width && height == other.height;
    }

    bool operator!=(const Dimensions& other) const {
        return !(*this == other);
    }
};

/**
 * @brief Base matrix interface providing common operations
 *
 * This is the base class for all matrix implementations. It provides
 * a uniform interface for accessing rows, elements, and performing operations.
 *
 * @tparam T Element type (must be copyable)
 */
template<typename T>
class Matrix {
public:
    virtual ~Matrix() = default;

    /// Get the number of columns in the matrix
    virtual size_t width() const = 0;

    /// Get the number of rows in the matrix
    virtual size_t height() const = 0;

    /// Get the dimensions of the matrix
    Dimensions dimensions() const {
        return Dimensions(width(), height());
    }

    /**
     * @brief Get element at position (r, c)
     * @param r Row index
     * @param c Column index
     * @return Element at (r, c)
     * @throws std::out_of_range if indices are out of bounds
     */
    T get(size_t r, size_t c) const {
        if (r >= height() || c >= width()) {
            throw std::out_of_range("Matrix indices out of bounds");
        }
        return get_unchecked(r, c);
    }

    /**
     * @brief Get element at position (r, c) without bounds checking
     * @param r Row index
     * @param c Column index
     * @return Element at (r, c)
     * @warning Undefined behavior if indices are out of bounds
     */
    virtual T get_unchecked(size_t r, size_t c) const = 0;

    /**
     * @brief Get a pointer to the specified row
     * @param r Row index
     * @return Pointer to the beginning of row r, or nullptr if not available
     * @note Not all matrix implementations can provide direct row access
     */
    virtual const T* row_ptr(size_t /* r */) const {
        return nullptr;  // Default implementation
    }

    /**
     * @brief Get a row as a vector
     * @param r Row index
     * @return Vector containing the elements of row r
     * @throws std::out_of_range if r >= height()
     */
    std::vector<T> row(size_t r) const {
        if (r >= height()) {
            throw std::out_of_range("Row index out of bounds");
        }
        std::vector<T> result;
        result.reserve(width());
        for (size_t c = 0; c < width(); ++c) {
            result.push_back(get_unchecked(r, c));
        }
        return result;
    }

    /**
     * @brief Get the first row
     * @return Vector containing the first row
     * @throws std::out_of_range if matrix is empty
     */
    std::vector<T> first_row() const {
        return row(0);
    }

    /**
     * @brief Get the last row
     * @return Vector containing the last row
     * @throws std::out_of_range if matrix is empty
     */
    std::vector<T> last_row() const {
        if (height() == 0) {
            throw std::out_of_range("Matrix is empty");
        }
        return row(height() - 1);
    }

    /**
     * @brief Get all rows as a vector of vectors
     * @return Vector of vectors, where each inner vector is a row
     */
    std::vector<std::vector<T>> rows() const {
        std::vector<std::vector<T>> result;
        result.reserve(height());
        for (size_t r = 0; r < height(); ++r) {
            result.push_back(row(r));
        }
        return result;
    }

    /**
     * @brief Check if the matrix is empty
     * @return true if height or width is zero
     */
    bool empty() const {
        return height() == 0 || width() == 0;
    }

    /**
     * @brief Get the total number of elements
     * @return width() * height()
     */
    size_t size() const {
        return width() * height();
    }
};

} // namespace p3_matrix


#pragma once

#include "matrix.hpp"
#include "p3_util/util.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_matrix {

template<typename T, typename RowMapper>
class RowIndexMappedView : public Matrix<T> {
public:
    RowIndexMappedView(const Matrix<T>& inner, size_t mapped_height, RowMapper row_mapper)
        : inner_(inner), mapped_height_(mapped_height), row_mapper_(std::move(row_mapper)) {}

    size_t width() const override { return inner_.width(); }
    size_t height() const override { return mapped_height_; }

    T get_unchecked(size_t r, size_t c) const override {
        return inner_.get_unchecked(row_mapper_(r), c);
    }

    const T* row_ptr(size_t r) const override {
        if (r >= mapped_height_) {
            return nullptr;
        }
        return inner_.row_ptr(row_mapper_(r));
    }

private:
    const Matrix<T>& inner_;
    size_t mapped_height_;
    RowMapper row_mapper_;
};

template<typename T>
class VerticallyStridedMatrixView : public Matrix<T> {
public:
    VerticallyStridedMatrixView(const Matrix<T>& inner, size_t stride, size_t offset)
        : inner_(inner), stride_(stride), offset_(offset), height_(compute_height(inner.height(), stride, offset)) {
        if (stride_ == 0) {
            throw std::invalid_argument("VerticallyStridedMatrixView: stride must be > 0");
        }
    }

    size_t width() const override { return inner_.width(); }
    size_t height() const override { return height_; }

    T get_unchecked(size_t r, size_t c) const override {
        return inner_.get_unchecked(r * stride_ + offset_, c);
    }

    const T* row_ptr(size_t r) const override {
        if (r >= height_) {
            return nullptr;
        }
        return inner_.row_ptr(r * stride_ + offset_);
    }

private:
    static size_t compute_height(size_t inner_height, size_t stride, size_t offset) {
        if (stride == 0 || offset >= inner_height) {
            return 0;
        }
        const size_t full_strides = inner_height / stride;
        const size_t remainder = inner_height % stride;
        const bool has_last = offset < remainder;
        return full_strides + static_cast<size_t>(has_last);
    }

    const Matrix<T>& inner_;
    size_t stride_;
    size_t offset_;
    size_t height_;
};

template<typename T>
class BitReversedMatrixView : public Matrix<T> {
public:
    explicit BitReversedMatrixView(const Matrix<T>& inner)
        : inner_(inner), log_height_(p3_util::log2_strict_usize(inner.height())) {}

    size_t width() const override { return inner_.width(); }
    size_t height() const override { return inner_.height(); }

    T get_unchecked(size_t r, size_t c) const override {
        return inner_.get_unchecked(p3_util::reverse_bits_len(r, log_height_), c);
    }

    const T* row_ptr(size_t r) const override {
        if (r >= height()) {
            return nullptr;
        }
        return inner_.row_ptr(p3_util::reverse_bits_len(r, log_height_));
    }

private:
    const Matrix<T>& inner_;
    size_t log_height_;
};

template<typename T>
class HorizontallyTruncatedView : public Matrix<T> {
public:
    HorizontallyTruncatedView(const Matrix<T>& inner, size_t truncated_width)
        : inner_(inner), width_(truncated_width) {
        if (truncated_width > inner.width()) {
            throw std::invalid_argument("HorizontallyTruncatedView: truncated width exceeds inner width");
        }
    }

    size_t width() const override { return width_; }
    size_t height() const override { return inner_.height(); }

    T get_unchecked(size_t r, size_t c) const override {
        return inner_.get_unchecked(r, c);
    }

    const T* row_ptr(size_t r) const override {
        return inner_.row_ptr(r);
    }

private:
    const Matrix<T>& inner_;
    size_t width_;
};

template<typename T>
class VerticallyStackedMatrices : public Matrix<T> {
public:
    explicit VerticallyStackedMatrices(std::vector<std::reference_wrapper<const Matrix<T>>> matrices)
        : matrices_(std::move(matrices)), width_(0), height_(0) {
        if (matrices_.empty()) {
            return;
        }

        width_ = matrices_.front().get().width();
        for (const auto& matrix_ref : matrices_) {
            const Matrix<T>& m = matrix_ref.get();
            if (m.width() != width_) {
                throw std::invalid_argument("VerticallyStackedMatrices: all matrices must have equal width");
            }
            start_rows_.push_back(height_);
            height_ += m.height();
        }
    }

    size_t width() const override { return width_; }
    size_t height() const override { return height_; }

    T get_unchecked(size_t r, size_t c) const override {
        const auto [matrix_idx, local_row] = locate_row(r);
        return matrices_[matrix_idx].get().get_unchecked(local_row, c);
    }

    const T* row_ptr(size_t r) const override {
        if (r >= height_) {
            return nullptr;
        }
        const auto [matrix_idx, local_row] = locate_row(r);
        return matrices_[matrix_idx].get().row_ptr(local_row);
    }

private:
    std::pair<size_t, size_t> locate_row(size_t r) const {
        auto it = std::upper_bound(start_rows_.begin(), start_rows_.end(), r);
        size_t matrix_idx = static_cast<size_t>(std::distance(start_rows_.begin(), it));
        if (matrix_idx > 0) {
            --matrix_idx;
        }
        return {matrix_idx, r - start_rows_[matrix_idx]};
    }

    std::vector<std::reference_wrapper<const Matrix<T>>> matrices_;
    std::vector<size_t> start_rows_;
    size_t width_;
    size_t height_;
};

} // namespace p3_matrix


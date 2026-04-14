#pragma once

#include "dense_matrix.hpp"
#include "extension_field.hpp"
#include "matrix.hpp"

#include <array>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace p3_matrix {

/// Zero-copy view that reinterprets a Matrix<Ext> as a Matrix<F> by
/// expanding each extension element into D base-field columns.
/// Mirrors Plonky3's `as_base_slice` / `flatten_to_base` semantics.
///
/// When the inner matrix provides contiguous row storage (row_ptr),
/// this view reinterprets the memory directly — no element copying.
template<typename F, size_t D, uint32_t W>
class FlattenedMatrixView : public Matrix<F> {
public:
    using Ext = p3_field::BinomialExtensionField<F, D, W>;

    static_assert(
        sizeof(Ext) == sizeof(F) * D,
        "BinomialExtensionField must have contiguous layout of D base elements");
    static_assert(
        alignof(Ext) == alignof(F),
        "BinomialExtensionField alignment must match base field alignment");

    explicit FlattenedMatrixView(const Matrix<Ext>& inner) : inner_(inner) {}

    size_t width() const override { return inner_.width() * D; }
    size_t height() const override { return inner_.height(); }

    F get_unchecked(size_t r, size_t c) const override {
        const size_t ext_col = c / D;
        const size_t coeff_idx = c % D;
        if (const Ext* row = inner_.row_ptr(r)) {
            return row[ext_col].coeffs[coeff_idx];
        }
        return inner_.get_unchecked(r, ext_col).coeffs[coeff_idx];
    }

    const F* row_ptr(size_t r) const override {
        const Ext* inner_row = inner_.row_ptr(r);
        if (!inner_row) return nullptr;
        return reinterpret_cast<const F*>(inner_row);
    }

    RowMajorMatrix<F> to_row_major_matrix() const {
        const size_t h = height();
        const size_t w = width();
        std::vector<F> data;
        data.reserve(h * w);
        for (size_t r = 0; r < h; ++r) {
            const F* rp = row_ptr(r);
            if (rp) {
                data.insert(data.end(), rp, rp + w);
            } else {
                for (size_t c = 0; c < w; ++c) {
                    data.push_back(get_unchecked(r, c));
                }
            }
        }
        return RowMajorMatrix<F>(std::move(data), w);
    }

private:
    const Matrix<Ext>& inner_;
};

/// Zero-copy view that reinterprets a Matrix<F> as a Matrix<Ext> by
/// grouping every D consecutive base-field columns into one extension element.
/// Mirrors Plonky3's `reconstitute_from_base` semantics.
///
/// When the inner matrix provides contiguous row storage (row_ptr),
/// this view reinterprets the memory directly — no element copying.
template<typename F, size_t D, uint32_t W>
class UnflattenedMatrixView : public Matrix<p3_field::BinomialExtensionField<F, D, W>> {
public:
    using Ext = p3_field::BinomialExtensionField<F, D, W>;

    static_assert(
        sizeof(Ext) == sizeof(F) * D,
        "BinomialExtensionField must have contiguous layout of D base elements");
    static_assert(
        alignof(Ext) == alignof(F),
        "BinomialExtensionField alignment must match base field alignment");

    explicit UnflattenedMatrixView(const Matrix<F>& inner) : inner_(inner) {
        if (inner_.width() % D != 0) {
            throw std::invalid_argument(
                "UnflattenedMatrixView: base matrix width must be divisible by extension degree");
        }
    }

    size_t width() const override { return inner_.width() / D; }
    size_t height() const override { return inner_.height(); }

    Ext get_unchecked(size_t r, size_t c) const override {
        if (const F* row = inner_.row_ptr(r)) {
            return reinterpret_cast<const Ext*>(row)[c];
        }
        std::array<F, D> coeffs{};
        for (size_t k = 0; k < D; ++k) {
            coeffs[k] = inner_.get_unchecked(r, c * D + k);
        }
        return Ext(coeffs);
    }

    const Ext* row_ptr(size_t r) const override {
        const F* inner_row = inner_.row_ptr(r);
        if (!inner_row) return nullptr;
        return reinterpret_cast<const Ext*>(inner_row);
    }

    RowMajorMatrix<Ext> to_row_major_matrix() const {
        const size_t h = height();
        const size_t w = width();
        std::vector<Ext> data;
        data.reserve(h * w);
        for (size_t r = 0; r < h; ++r) {
            const Ext* rp = row_ptr(r);
            if (rp) {
                data.insert(data.end(), rp, rp + w);
            } else {
                for (size_t c = 0; c < w; ++c) {
                    data.push_back(get_unchecked(r, c));
                }
            }
        }
        return RowMajorMatrix<Ext>(std::move(data), w);
    }

private:
    const Matrix<F>& inner_;
};

/// Return a zero-copy FlattenedMatrixView over the given extension matrix.
template<typename F, size_t D, uint32_t W>
FlattenedMatrixView<F, D, W>
flatten_to_base(const Matrix<p3_field::BinomialExtensionField<F, D, W>>& ext_matrix) {
    return FlattenedMatrixView<F, D, W>(ext_matrix);
}

/// Return a zero-copy UnflattenedMatrixView over the given base matrix.
template<typename F, size_t D, uint32_t W>
UnflattenedMatrixView<F, D, W>
unflatten_from_base(const Matrix<F>& base_matrix) {
    return UnflattenedMatrixView<F, D, W>(base_matrix);
}

} // namespace p3_matrix

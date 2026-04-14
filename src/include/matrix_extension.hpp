#pragma once

#include "extension_field.hpp"
#include "matrix.hpp"

#include <array>
#include <stdexcept>

namespace p3_matrix {

template<typename F, size_t D, uint32_t W>
class FlattenedMatrixView : public Matrix<F> {
public:
    using Ext = p3_field::BinomialExtensionField<F, D, W>;

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
        const Ext* ptr = inner_.row_ptr(r);
        return ptr ? reinterpret_cast<const F*>(ptr) : nullptr;
    }

private:
    const Matrix<Ext>& inner_;
};

template<typename F, size_t D, uint32_t W>
class UnflattenedMatrixView : public Matrix<p3_field::BinomialExtensionField<F, D, W>> {
public:
    using Ext = p3_field::BinomialExtensionField<F, D, W>;

    explicit UnflattenedMatrixView(const Matrix<F>& inner) : inner_(inner) {
        if (inner_.width() % D != 0) {
            throw std::invalid_argument(
                "UnflattenedMatrixView: base matrix width must be divisible by extension degree");
        }
    }

    size_t width() const override { return inner_.width() / D; }
    size_t height() const override { return inner_.height(); }

    Ext get_unchecked(size_t r, size_t c) const override {
        std::array<F, D> coeffs{};
        for (size_t k = 0; k < D; ++k) {
            coeffs[k] = inner_.get_unchecked(r, c * D + k);
        }
        return Ext(coeffs);
    }

private:
    const Matrix<F>& inner_;
};

template<typename F, size_t D, uint32_t W>
FlattenedMatrixView<F, D, W>
flatten_to_base(const Matrix<p3_field::BinomialExtensionField<F, D, W>>& ext_matrix) {
    return FlattenedMatrixView<F, D, W>(ext_matrix);
}

template<typename F, size_t D, uint32_t W>
UnflattenedMatrixView<F, D, W>
unflatten_from_base(const Matrix<F>& base_matrix) {
    return UnflattenedMatrixView<F, D, W>(base_matrix);
}

} // namespace p3_matrix


#pragma once

#include "dense_matrix.hpp"
#include "extension_field.hpp"
#include "matrix.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace p3_matrix {

template<typename F, size_t D, uint32_t W>
RowMajorMatrix<F> flatten_to_base(const Matrix<p3_field::BinomialExtensionField<F, D, W>>& ext_matrix) {
    const size_t h = ext_matrix.height();
    const size_t ext_w = ext_matrix.width();
    const size_t base_w = ext_w * D;

    std::vector<F> base_values(h * base_w);
    for (size_t r = 0; r < h; ++r) {
        F* row_out = &base_values[r * base_w];
        for (size_t c = 0; c < ext_w; ++c) {
            const auto value = ext_matrix.get_unchecked(r, c);
            for (size_t k = 0; k < D; ++k) {
                row_out[c * D + k] = value.coeffs[k];
            }
        }
    }
    return RowMajorMatrix<F>(std::move(base_values), base_w);
}

template<typename F, size_t D, uint32_t W>
RowMajorMatrix<p3_field::BinomialExtensionField<F, D, W>>
unflatten_from_base(const Matrix<F>& base_matrix) {
    const size_t h = base_matrix.height();
    const size_t base_w = base_matrix.width();
    if (base_w % D != 0) {
        throw std::invalid_argument("unflatten_from_base: base matrix width must be divisible by extension degree");
    }

    const size_t ext_w = base_w / D;
    using Ext = p3_field::BinomialExtensionField<F, D, W>;
    std::vector<Ext> ext_values(h * ext_w);

    for (size_t r = 0; r < h; ++r) {
        Ext* row_out = &ext_values[r * ext_w];
        for (size_t c = 0; c < ext_w; ++c) {
            std::array<F, D> coeffs{};
            for (size_t k = 0; k < D; ++k) {
                coeffs[k] = base_matrix.get_unchecked(r, c * D + k);
            }
            row_out[c] = Ext(coeffs);
        }
    }

    return RowMajorMatrix<Ext>(std::move(ext_values), ext_w);
}

} // namespace p3_matrix


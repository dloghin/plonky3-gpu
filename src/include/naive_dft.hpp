#pragma once

#include "traits.hpp"
#include "dense_matrix.hpp"
#include "p3_util/util.hpp"

#include <cstddef>

namespace p3_dft {

template<typename F>
class NaiveDft : public TwoAdicSubgroupDft<F, NaiveDft<F>> {
public:
    p3_matrix::RowMajorMatrix<F> dft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        const size_t h = mat.height();
        const size_t w = mat.width();
        if (h <= 1) return mat;

        const size_t log_h = p3_util::log2_strict_usize(h);
        const F g = TwoAdicFieldTraits<F>::two_adic_generator(log_h);

        p3_matrix::RowMajorMatrix<F> out(h, w, F::zero_val());
        F point = F::one_val();
        for (size_t r = 0; r < h; ++r) {
            F point_power = F::one_val();
            for (size_t src_r = 0; src_r < h; ++src_r) {
                for (size_t c = 0; c < w; ++c) {
                    out.set_unchecked(r, c,
                        out.get_unchecked(r, c) + point_power * mat.get_unchecked(src_r, c));
                }
                point_power = point_power * point;
            }
            point = point * g;
        }
        return out;
    }

    p3_matrix::RowMajorMatrix<F> idft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        const size_t h = mat.height();
        if (h <= 1) return mat;

        const size_t log_h = p3_util::log2_strict_usize(h);
        mat = dft_batch(std::move(mat));

        size_t lo = 1;
        size_t hi = h - 1;
        while (lo < hi) {
            mat.swap_rows(lo, hi);
            ++lo;
            --hi;
        }

        const F inv_n = compute_inv_n(log_h);
        mat.scale(inv_n);
        return mat;
    }

    p3_matrix::RowMajorMatrix<F> coset_dft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        const size_t h = mat.height();
        F shift_power = shift;
        for (size_t i = 1; i < h; ++i) {
            mat.scale_row(i, shift_power);
            shift_power = shift_power * shift;
        }
        return dft_batch(std::move(mat));
    }

    p3_matrix::RowMajorMatrix<F> coset_idft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        mat = idft_batch(std::move(mat));
        const size_t h = mat.height();
        const F inv_shift = shift.inv();
        F inv_shift_power = inv_shift;
        for (size_t i = 1; i < h; ++i) {
            mat.scale_row(i, inv_shift_power);
            inv_shift_power = inv_shift_power * inv_shift;
        }
        return mat;
    }

private:
    static F compute_inv_n(size_t log_h) {
        F two = F::one_val() + F::one_val();
        F inv_two = two.inv();
        return inv_two.exp_u64(static_cast<uint64_t>(log_h));
    }
};

} // namespace p3_dft

#pragma once

#include "butterflies.hpp"
#include "traits.hpp"
#include "dense_matrix.hpp"
#include "util.hpp"
#include "p3_util/util.hpp"

#include <unordered_map>
#include <vector>
#include <cstddef>

namespace p3_dft {

/**
 * @brief Radix-2 Decimation-In-Time (DIT) NTT over two-adic fields.
 *
 * Implements the Cooley-Tukey FFT algorithm for prime fields with high
 * 2-adicity.  Twiddle factors are memoised by log2(n) to amortise the
 * cost of their generation across multiple DFT calls of the same size.
 *
 * The DFT is defined over the canonical 2^log_h-th subgroup of the
 * multiplicative group of F, whose generator is
 *   g = TwoAdicFieldTraits<F>::two_adic_generator(log_h).
 *
 * @tparam F  Field element type.  Must expose:
 *              - F::zero_val(), F::one_val()
 *              - operator+, operator-, operator*
 *              - F::two_adic_generator(size_t bits) [via TwoAdicFieldTraits]
 *              - inv() (for IDFT scaling)
 *              - powers() iterator (or hand-rolled power loop)
 */
template<typename F>
class Radix2Dit : public TwoAdicSubgroupDft<F, Radix2Dit<F>> {
public:
    // -----------------------------------------------------------------------
    // Twiddle-factor cache
    // -----------------------------------------------------------------------

    /**
     * @brief Return (or compute and cache) twiddle factors for size 2^log_h.
     *
     * Returns n/2 factors: g^0, g^1, ..., g^(n/2-1) where g is the
     * primitive n-th root of unity.
     */
    const std::vector<F>& get_or_compute_twiddles(size_t log_h) {
        auto it = twiddles_.find(log_h);
        if (it != twiddles_.end()) {
            return it->second;
        }

        size_t n = static_cast<size_t>(1) << log_h;
        size_t half = n / 2;

        F root = TwoAdicFieldTraits<F>::two_adic_generator(log_h);

        std::vector<F> tw;
        tw.reserve(half);
        F cur = F::one_val();
        for (size_t i = 0; i < half; ++i) {
            tw.push_back(cur);
            cur = cur * root;
        }

        auto& stored = twiddles_[log_h];
        stored = std::move(tw);
        return stored;
    }

    // -----------------------------------------------------------------------
    // Forward DFT
    // -----------------------------------------------------------------------

    /**
     * @brief Forward DFT on a batch of columns.
     *
     * Input:  RowMajorMatrix<F> of height h = 2^log_h and width w.
     * Output: same shape; column j holds the DFT of the j-th input column.
     *
     * Algorithm: DIT (bit-reverse then butterfly layers).
     */
    p3_matrix::RowMajorMatrix<F> dft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        size_t h = mat.height();
        if (h <= 1) return mat;

        size_t log_h = p3_util::log2_strict_usize(h);
        size_t w = mat.width();

        // Step 1: bit-reverse rows
        p3_matrix::reverse_matrix_index_bits(mat);

        // Step 2: DIT butterfly layers
        const std::vector<F>& tw = get_or_compute_twiddles(log_h);
        size_t half_n = h / 2;  // = n/2, size of twiddle table

        for (size_t l = 0; l < log_h; ++l) {
            size_t half  = static_cast<size_t>(1) << l;       // half-block size
            size_t step  = half * 2;                           // block size
            // Twiddle for position j in this layer is tw[j * stride] where
            // stride = (n/2) / half = half_n >> l.
            size_t stride = half_n >> l;

            for (size_t block = 0; block < h; block += step) {
                // j == 0: twiddle is 1, use twiddle-free butterfly
                {
                    F* a = mat.row_mut(block);
                    F* b = mat.row_mut(block + half);
                    twiddle_free_butterfly(a, b, w);
                }
                // j > 0: use twiddle factor
                for (size_t j = 1; j < half; ++j) {
                    const F& twiddle = tw[j * stride];
                    F* a = mat.row_mut(block + j);
                    F* b = mat.row_mut(block + half + j);
                    dit_butterfly_safe(a, b, twiddle, w);
                }
            }
        }

        return mat;
    }

    // -----------------------------------------------------------------------
    // Inverse DFT
    // -----------------------------------------------------------------------

    /**
     * @brief Inverse DFT on a batch of columns.
     *
     * Recovers polynomial coefficients from evaluations using the identity:
     *   IDFT(A)[k] = (1/n) * DFT(A)[n - k  mod n]
     *
     * That is: apply the forward DFT, then reverse rows [1..n-1] in-place,
     * then scale the whole matrix by 1/n.
     */
    p3_matrix::RowMajorMatrix<F> idft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        size_t h = mat.height();
        if (h <= 1) return mat;

        size_t log_h = p3_util::log2_strict_usize(h);

        // Step 1: Forward DFT
        mat = dft_batch(std::move(mat));

        // Step 2: Reverse rows [1 .. h-1], keeping row 0 fixed.
        // This maps result[k] -> DFT(input)[h - k mod h].
        size_t lo = 1;
        size_t hi = h - 1;
        while (lo < hi) {
            mat.swap_rows(lo, hi);
            ++lo;
            --hi;
        }

        // Step 3: scale by 1/n
        F inv_n = compute_inv_n(log_h);
        mat.scale(inv_n);

        return mat;
    }

    // -----------------------------------------------------------------------
    // Coset DFT / IDFT
    // -----------------------------------------------------------------------

    /**
     * @brief DFT on a coset of the standard subgroup.
     *
     * Equivalent to multiplying row i by shift^i before calling dft_batch.
     * coset_dft_batch(mat, shift) evaluates each polynomial at
     *   shift * omega^0, shift * omega^1, ..., shift * omega^{h-1}.
     */
    p3_matrix::RowMajorMatrix<F> coset_dft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        size_t h = mat.height();

        // Multiply row i by shift^i (row 0 is always multiplied by 1, skip it)
        F shift_power = shift;
        for (size_t i = 1; i < h; ++i) {
            mat.scale_row(i, shift_power);
            shift_power = shift_power * shift;
        }

        return dft_batch(std::move(mat));
    }

    /**
     * @brief Inverse DFT followed by coset un-twisting.
     *
     * Recovers polynomial coefficients evaluated on the coset
     *   shift * {omega^0, ..., omega^{h-1}}.
     */
    p3_matrix::RowMajorMatrix<F> coset_idft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        mat = idft_batch(std::move(mat));

        size_t h = mat.height();

        // Multiply row i by shift^{-i} (row 0 multiplied by 1, skip it)
        F inv_shift = shift.inv();
        F inv_shift_power = inv_shift;
        for (size_t i = 1; i < h; ++i) {
            mat.scale_row(i, inv_shift_power);
            inv_shift_power = inv_shift_power * inv_shift;
        }

        return mat;
    }

    /**
     * @brief Low-degree extension (LDE) on a coset.
     *
     * Takes a matrix of evaluations on the canonical 2^log_h-th subgroup,
     * recovers polynomial coefficients via IDFT, zero-pads to
     * height * 2^added_bits, then evaluates on the coset
     *   shift * {omega'^0, ..., omega'^{h*2^added_bits - 1}}
     * where omega' is the primitive (h * 2^added_bits)-th root of unity.
     */
    p3_matrix::RowMajorMatrix<F> coset_lde_batch(
        p3_matrix::RowMajorMatrix<F> mat, size_t added_bits, const F& shift)
    {
        // 1. IDFT to get polynomial coefficients
        mat = idft_batch(std::move(mat));

        // 2. Zero-pad to new height
        size_t new_h = mat.height() << added_bits;
        mat.pad_to_height(new_h);

        // 3. Coset DFT on the larger domain
        return coset_dft_batch(std::move(mat), shift);
    }

    // -----------------------------------------------------------------------
    // Extension field DFT
    // -----------------------------------------------------------------------

    /**
     * @brief DFT of a matrix whose elements are extension field elements.
     *
     * Decomposes each column of EF elements into D columns of base field F
     * elements, runs dft_batch on each, then reconstitutes EF elements.
     *
     * @tparam EF  Extension field type with EF::DEGREE and EF::coeffs[D].
     */
    template<typename EF>
    p3_matrix::RowMajorMatrix<EF> dft_algebra_batch(
        p3_matrix::RowMajorMatrix<EF> mat)
    {
        return algebra_batch_impl(std::move(mat),
            [this](p3_matrix::RowMajorMatrix<F> m) {
                return dft_batch(std::move(m));
            });
    }

    /**
     * @brief IDFT of a matrix whose elements are extension field elements.
     *
     * Same decomposition strategy as dft_algebra_batch but uses idft_batch.
     *
     * @tparam EF  Extension field type.
     */
    template<typename EF>
    p3_matrix::RowMajorMatrix<EF> idft_algebra_batch(
        p3_matrix::RowMajorMatrix<EF> mat)
    {
        return algebra_batch_impl(std::move(mat),
            [this](p3_matrix::RowMajorMatrix<F> m) {
                return idft_batch(std::move(m));
            });
    }

    /**
     * @brief IDFT on a single column of extension field elements.
     *
     * Convenience wrapper that accepts/returns a std::vector<EF>.
     */
    template<typename EF>
    std::vector<EF> idft_algebra(std::vector<EF> vec) {
        if (vec.empty()) return {};
        // Wrap as single-column matrix, moving the vector's data.
        p3_matrix::RowMajorMatrix<EF> mat(std::move(vec), 1);
        mat = idft_algebra_batch(std::move(mat));
        // Extract column by moving the matrix's data.
        return std::move(mat.values);
    }

private:
    // Memoised twiddle factors: log_h -> [root^0, root^1, ..., root^(n/2-1)]
    std::unordered_map<size_t, std::vector<F>> twiddles_;

    /**
     * @brief Compute 1 / 2^log_h in F using exp_u64 (square-and-multiply).
     */
    static F compute_inv_n(size_t log_h) {
        F two = F::one_val() + F::one_val();
        F inv_two = two.inv();
        return inv_two.exp_u64(static_cast<uint64_t>(log_h));
    }

    /**
     * @brief Common implementation for dft_algebra_batch and idft_algebra_batch.
     *
     * Decomposes EF elements into D base-field matrices, applies dft_fn to
     * each, then reconstitutes the EF matrix.
     */
    template<typename EF, typename DftFn>
    p3_matrix::RowMajorMatrix<EF> algebra_batch_impl(
        p3_matrix::RowMajorMatrix<EF> mat, DftFn&& dft_fn)
    {
        constexpr size_t D = EF::DEGREE;
        size_t h = mat.height();
        size_t w = mat.width();

        // Build D base-field matrices (one per extension coefficient)
        std::vector<p3_matrix::RowMajorMatrix<F>> base_mats(D,
            p3_matrix::RowMajorMatrix<F>(h, w));

        for (size_t r = 0; r < h; ++r) {
            for (size_t c = 0; c < w; ++c) {
                const EF& elem = mat.get_unchecked(r, c);
                for (size_t d = 0; d < D; ++d) {
                    base_mats[d].set_unchecked(r, c, elem[d]);
                }
            }
        }

        // Apply dft_fn to each base-field matrix
        for (size_t d = 0; d < D; ++d) {
            base_mats[d] = dft_fn(std::move(base_mats[d]));
        }

        // Reconstitute EF matrix
        p3_matrix::RowMajorMatrix<EF> result(h, w);
        for (size_t r = 0; r < h; ++r) {
            for (size_t c = 0; c < w; ++c) {
                EF elem;
                for (size_t d = 0; d < D; ++d) {
                    elem[d] = base_mats[d].get_unchecked(r, c);
                }
                result.set_unchecked(r, c, elem);
            }
        }
        return result;
    }
};

} // namespace p3_dft

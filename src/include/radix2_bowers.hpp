#pragma once

#include "traits.hpp"
#include "radix2_dit.hpp"

namespace p3_dft {

/**
 * Bowers-network variant placeholder.
 *
 * Keeps the same public API as the Rust `Radix2Bowers` implementation.
 * For now, it delegates to the existing radix-2 DIT implementation so callers
 * can use a consistent interface across DFT variants.
 */
template<typename F>
class Radix2Bowers : public TwoAdicSubgroupDft<F, Radix2Bowers<F>> {
public:
    p3_matrix::RowMajorMatrix<F> dft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        return dit_.dft_batch(std::move(mat));
    }

    p3_matrix::RowMajorMatrix<F> idft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        return dit_.idft_batch(std::move(mat));
    }

    p3_matrix::RowMajorMatrix<F> coset_dft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        return dit_.coset_dft_batch(std::move(mat), shift);
    }

    p3_matrix::RowMajorMatrix<F> coset_idft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        return dit_.coset_idft_batch(std::move(mat), shift);
    }

    p3_matrix::RowMajorMatrix<F> coset_lde_batch(
        p3_matrix::RowMajorMatrix<F> mat, size_t added_bits, const F& shift)
    {
        return dit_.coset_lde_batch(std::move(mat), added_bits, shift);
    }

private:
    Radix2Dit<F> dit_;
};

} // namespace p3_dft

#pragma once

#include "dense_matrix.hpp"
#include <cstddef>

namespace p3_dft {

/**
 * @brief Traits struct for two-adic field types.
 *
 * Provides the two-adic generator for a field type.  Specialize this for
 * any field type that participates in NTT (e.g. extension fields that store
 * the generator in a separate traits class rather than on the type itself).
 *
 * The primary template simply delegates to F::two_adic_generator(bits),
 * which works for BabyBear, Goldilocks, and Mersenne31 as defined in the
 * existing field headers.
 *
 * @tparam F Field type
 */
template<typename F>
struct TwoAdicFieldTraits {
    /**
     * @brief Return a primitive 2^bits-th root of unity in F.
     * @param bits  log2 of the desired root order
     */
    static F two_adic_generator(size_t bits) {
        return F::two_adic_generator(bits);
    }
};

/**
 * @brief Interface for a DFT over a two-adic subgroup.
 *
 * Concrete implementations (e.g. Radix2Dit<F>) provide these operations.
 * This is a non-virtual base that documents the expected API; callers
 * are free to use the concrete class directly.
 *
 * @tparam F     Base field element type
 * @tparam Impl  Concrete implementation (CRTP)
 */
template<typename F, typename Impl>
struct TwoAdicSubgroupDft {
    /**
     * @brief Forward DFT on a batch of columns.
     *
     * The input matrix has height h = 2^log_h and an arbitrary number of
     * columns.  Each column is treated as an independent polynomial; the
     * output is its evaluation on the canonical 2^log_h-th roots of unity.
     */
    p3_matrix::RowMajorMatrix<F> dft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        return static_cast<Impl*>(this)->dft_batch(std::move(mat));
    }

    /**
     * @brief Inverse DFT on a batch of columns.
     *
     * Recovers polynomial coefficients from evaluations.
     */
    p3_matrix::RowMajorMatrix<F> idft_batch(p3_matrix::RowMajorMatrix<F> mat) {
        return static_cast<Impl*>(this)->idft_batch(std::move(mat));
    }
};

} // namespace p3_dft

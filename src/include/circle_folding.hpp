#pragma once

/**
 * @file circle_folding.hpp
 * @brief Circle FRI folding helpers.
 */

#include "circle_domain.hpp"
#include "dense_matrix.hpp"
#include "p3_util/util.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace p3_circle {

template <typename F>
F halve(const F& x) {
    static const F two_inv = F::two_val().inv();
    return x * two_inv;
}

template <typename F>
std::vector<F> fold_with_twiddles(const p3_matrix::RowMajorMatrix<F>& evals,
                                  const F& beta,
                                  const std::vector<F>& twiddles) {
    if (evals.width() != 2) {
        throw std::invalid_argument("circle fold expects width-2 rows");
    }
    if (evals.height() != twiddles.size()) {
        throw std::invalid_argument("circle fold twiddle count mismatch");
    }
    std::vector<F> out(evals.height());
    for (std::size_t r = 0; r < evals.height(); ++r) {
        const F lo = evals.get_unchecked(r, 0);
        const F hi = evals.get_unchecked(r, 1);
        const F sum = lo + hi;
        const F diff = (lo - hi) * twiddles[r].inv();
        out[r] = halve(sum + beta * diff);
    }
    return out;
}

template <typename F>
std::vector<F> fold_y(const F& beta, const p3_matrix::RowMajorMatrix<F>& evals) {
    const std::size_t log_n = p3_util::log2_strict_usize(evals.height()) + 1;
    return fold_with_twiddles(evals, beta, CircleDomain<F>::standard(log_n).y_twiddles());
}

template <typename F>
F fold_y_row(std::size_t index,
             std::size_t log_folded_height,
             const F& beta,
             const F& lo,
             const F& hi) {
    const F t = CircleDomain<F>::standard(log_folded_height + 1).nth_y_twiddle(index).inv();
    return halve((lo + hi) + beta * ((lo - hi) * t));
}

template <typename F>
std::vector<F> fold_x(const F& beta,
                      std::size_t log_arity,
                      const p3_matrix::RowMajorMatrix<F>& evals) {
    if (log_arity != 1) {
        throw std::invalid_argument("Circle folding currently supports arity 2");
    }
    const std::size_t log_n = p3_util::log2_strict_usize(evals.width() * evals.height());
    return fold_with_twiddles(evals, beta, CircleDomain<F>::standard(log_n + 1).x_twiddles(0));
}

template <typename F>
F fold_x_row(std::size_t index,
             std::size_t log_folded_height,
             std::size_t log_arity,
             const F& beta,
             const F& lo,
             const F& hi) {
    if (log_arity != 1) {
        throw std::invalid_argument("Circle folding currently supports arity 2");
    }
    const F t = CircleDomain<F>::standard(log_folded_height + log_arity + 1)
                    .nth_x_twiddle(p3_util::reverse_bits_len(index, log_folded_height))
                    .inv();
    return halve((lo + hi) + beta * ((lo - hi) * t));
}

template <typename F>
std::vector<F> fold_circle(const std::vector<F>& evals, const F& beta) {
    if (evals.size() < 2 || (evals.size() & 1u) != 0) {
        throw std::invalid_argument("fold_circle expects a non-empty even number of evaluations");
    }
    return fold_y(beta, p3_matrix::RowMajorMatrix<F>(evals, 2));
}

} // namespace p3_circle

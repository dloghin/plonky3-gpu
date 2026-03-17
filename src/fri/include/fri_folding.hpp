#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include "p3_util/util.hpp"

namespace p3_fri {

// ---------------------------------------------------------------------------
// embed_base: lift a Val element into a Challenge element.
//
// When Val == Challenge (base field used as challenge), just return the value.
// When Val != Challenge (extension field), use Challenge::from_base().
// ---------------------------------------------------------------------------
namespace detail {

// Primary template: Val != Challenge → use Challenge::from_base(v)
template <typename Val, typename Challenge,
          bool Same = std::is_same<Val, Challenge>::value>
struct EmbedBase {
    static Challenge embed(const Val& v) {
        return Challenge::from_base(v);
    }
};

// Specialization: Val == Challenge → identity
template <typename Val>
struct EmbedBase<Val, Val, true> {
    static Val embed(const Val& v) { return v; }
};

} // namespace detail

// Convenience wrapper
template <typename Val, typename Challenge>
inline Challenge embed_base(const Val& v) {
    return detail::EmbedBase<Val, Challenge>::embed(v);
}

// ---------------------------------------------------------------------------
// TwoAdicFriFolding
// ---------------------------------------------------------------------------

// TwoAdicFriFolding: folding logic for the FRI protocol over two-adic fields.
//
// Template parameters:
//   Val       - the base (prime) field type; must provide two_adic_generator(bits)
//   Challenge - the extension field type used as the folding challenge
//               (may equal Val for base-field-only tests)
//
// The key operation is fold_row: given arity evaluations of a polynomial at a
// coset {x, x*w, x*w^2, ..., x*w^(a-1)}, compute the unique degree-(a-1)
// polynomial through those points evaluated at `beta` (the folding challenge).
//
// This is Lagrange interpolation: result = sum_i evals[i] * L_i(beta)
// where L_i(beta) = prod_{j != i} (beta - t_j) / (t_i - t_j)
// and   t_i = x * w^i.
template <typename Val, typename Challenge>
struct TwoAdicFriFolding {

    // Fold a single row.
    //
    // index       : index in the *folded* (output) domain, 0 .. height/arity - 1
    // log_height  : log2 of the current (un-folded) vector length
    // log_arity   : log2 of the folding arity (arity = 2^log_arity)
    // beta        : the folding challenge (a Challenge element)
    // evals       : vector of `arity` evaluations f(x*w^0), ..., f(x*w^(a-1))
    //
    // Returns the folded Challenge value for this index.
    static Challenge fold_row(
        size_t index,
        size_t log_height,
        size_t log_arity,
        const Challenge& beta,
        const std::vector<Challenge>& evals
    ) {
        if (log_arity == 0) {
            // No folding: arity = 1, just return the single evaluation.
            if (evals.empty()) throw std::invalid_argument("fold_row: evals is empty");
            return evals[0];
        }

        size_t arity = size_t(1) << log_arity;
        size_t log_folded_height = log_height - log_arity;

        if (evals.size() != arity) {
            throw std::invalid_argument("fold_row: evals.size() != arity");
        }

        // x = omega^(bit_rev(index, log_folded_height))
        // where omega is the 2^log_height primitive root of unity.
        size_t x_pow_idx = p3_util::reverse_bits_len(index, log_folded_height);
        Val omega = Val::two_adic_generator(log_height);
        Val x_val = omega.exp_u64(static_cast<uint64_t>(x_pow_idx));
        Challenge x = embed_base<Val, Challenge>(x_val);

        // w = primitive arity-th root of unity in the base field
        Val w_val = Val::two_adic_generator(log_arity);

        // Build evaluation points t_i = x * w^i for i = 0..arity-1,
        // then bit-reverse to match the bit-reversed data layout.
        // In the committed matrix, column j holds the evaluation at
        // x * w^(bit_reverse(j, log_arity)), so after bit-reversal
        // t[j] correctly maps to evals[j].
        std::vector<Challenge> t(arity);
        {
            Val w_pow = Val::one_val();  // w^0 = 1
            for (size_t i = 0; i < arity; ++i) {
                t[i] = x * embed_base<Val, Challenge>(w_pow);
                w_pow = w_pow * w_val;
            }
        }
        // Bit-reverse to match data ordering (critical for arity > 2)
        {
            for (size_t i = 0; i < arity; ++i) {
                size_t j = p3_util::reverse_bits_len(i, log_arity);
                if (i < j) {
                    std::swap(t[i], t[j]);
                }
            }
        }

        // Lagrange interpolation at beta:
        //   result = sum_i evals[i] * L_i(beta)
        //   L_i(beta) = prod_{j!=i}(beta - t[j]) / prod_{j!=i}(t[i] - t[j])
        //
        // For the denominator we use the closed form:
        //   prod_{j!=i}(t[i] - t[j]) = x^(arity-1) * w^(i*(arity-1)) * arity
        // (Follows from prod_{k=1}^{a-1}(1 - w^k) = a for a primitive a-th root w.)
        //
        // So L_i(beta) = full_num / ((beta - t[i]) * x^(arity-1) * w_inv^i * arity)
        // where full_num = prod_{j=0}^{arity-1}(beta - t[j]).

        // Fast path: if beta equals one of the evaluation points t[i],
        // the interpolating polynomial at beta is just evals[i].
        for (size_t i = 0; i < arity; ++i) {
            if (beta == t[i]) {
                return evals[i];
            }
        }

        // Precompute full numerator product
        Challenge full_num = Challenge::one_val();
        for (size_t j = 0; j < arity; ++j) {
            full_num = full_num * (beta - t[j]);
        }

        // arity_inv = 1/arity (computed as (1/2)^log_arity in the base field)
        Val arity_inv_val = Val::one_val();
        {
            Val two_inv = (Val::one_val() + Val::one_val()).inv();
            for (size_t k = 0; k < log_arity; ++k) {
                arity_inv_val = arity_inv_val * two_inv;
            }
        }
        Challenge arity_inv = embed_base<Val, Challenge>(arity_inv_val);

        // x^(arity-1)
        Val x_pow_am1 = x_val.exp_u64(static_cast<uint64_t>(arity - 1));

        // w_inv = w^(arity-1) = w^(-1)
        Val w_inv = w_val.exp_u64(static_cast<uint64_t>(arity - 1));

        Challenge result = Challenge::zero_val();

        Val w_inv_i = Val::one_val();  // w_inv^i, starts at 1 for i=0
        for (size_t i = 0; i < arity; ++i) {
            // Denominator factor from base field: x^(arity-1) * w_inv^i
            Val base_denom_val = x_pow_am1 * w_inv_i;
            Challenge base_denom = embed_base<Val, Challenge>(base_denom_val);

            // Full denominator: (beta - t[i]) * base_denom * arity
            // We fold arity into the division as: multiply by arity_inv
            Challenge denom_full = (beta - t[i]) * base_denom;

            // L_i = full_num * arity_inv / denom_full
            Challenge L_i = full_num * arity_inv * denom_full.inv();

            result = result + evals[i] * L_i;

            w_inv_i = w_inv_i * w_inv;
        }

        return result;
    }

    // Fold an entire flat evaluation vector by reshaping into [height][arity]
    // and applying fold_row to each row.
    //
    // log_height : log2 of the *unfolded* total length (= height * arity)
    // log_arity  : log2 of arity
    // beta       : folding challenge
    // current    : flat row-major buffer, length = 2^log_height
    //
    // Returns a vector of length 2^(log_height - log_arity).
    static std::vector<Challenge> fold_matrix(
        size_t log_height,
        size_t log_arity,
        const Challenge& beta,
        const std::vector<Challenge>& current
    ) {
        size_t arity = size_t(1) << log_arity;
        size_t total = current.size();
        if (total % arity != 0) {
            throw std::invalid_argument("fold_matrix: current.size() not divisible by arity");
        }
        size_t height = total / arity;  // number of output rows

        std::vector<Challenge> folded;
        folded.reserve(height);

        std::vector<Challenge> row_evals(arity);
        for (size_t i = 0; i < height; ++i) {
            std::copy_n(current.begin() + i * arity, arity, row_evals.begin());
            folded.push_back(fold_row(i, log_height, log_arity, beta, row_evals));
        }
        return folded;
    }
};

} // namespace p3_fri

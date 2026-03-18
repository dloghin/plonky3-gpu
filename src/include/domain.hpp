#pragma once

/**
 * @file domain.hpp
 * @brief Domain types for polynomial commitment schemes.
 *
 * Mirrors plonky3/commit/src/domain.rs and plonky3/field/src/coset.rs.
 *
 * Provides:
 *   - Dimensions      (re-exported from p3_matrix)
 *   - TwoAdicMultiplicativeCoset<F>
 */

#include "matrix.hpp"
#include "p3_util/util.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace p3_commit {

// ---------------------------------------------------------------------------
// Dimensions
// Re-exported from p3_matrix for convenience — commit code can use
// p3_commit::Dimensions without needing to pull in the matrix headers directly.
// ---------------------------------------------------------------------------
using Dimensions = p3_matrix::Dimensions;

// ---------------------------------------------------------------------------
// TwoAdicMultiplicativeCoset
//
// Represents the coset  shift * H  where H is the unique subgroup of order
// 2^log_n in the multiplicative group of a two-adic prime field F.
//
// Template requirement on F:
//   - F::one_val()                        -- multiplicative identity
//   - F::two_adic_generator(size_t bits)  -- primitive 2^bits-th root of unity
//   - F operator*(const F&) const
//
// Mirrors Rust's TwoAdicMultiplicativeCoset<F: TwoAdicField>.
// ---------------------------------------------------------------------------
template<typename F>
struct TwoAdicMultiplicativeCoset {
    size_t log_n;  ///< log2 of the subgroup size
    F shift;       ///< Coset shift (typically F::one_val() or F::generator())

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------
    TwoAdicMultiplicativeCoset() : log_n(0), shift(F::one_val()) {}
    TwoAdicMultiplicativeCoset(size_t log_n_, F shift_)
        : log_n(log_n_), shift(shift_) {}

    // -----------------------------------------------------------------------
    // Size of the domain
    // -----------------------------------------------------------------------
    size_t size() const { return size_t(1) << log_n; }

    // -----------------------------------------------------------------------
    // First element: the coset shift itself (= shift * g^0)
    // -----------------------------------------------------------------------
    F first_point() const { return shift; }

    // -----------------------------------------------------------------------
    // Advance one step in the coset: x -> x * g  where g is the subgroup gen.
    // Returns the next element in the coset after x.
    // -----------------------------------------------------------------------
    F next_point(const F& x) const {
        F g = F::two_adic_generator(log_n);
        return x * g;
    }

    // -----------------------------------------------------------------------
    // Generate all 2^log_n elements of the coset in order.
    //
    // Elements:  shift, shift*g, shift*g^2, ..., shift*g^(n-1)
    // where g = F::two_adic_generator(log_n).
    // -----------------------------------------------------------------------
    std::vector<F> elements() const {
        size_t n = size();
        std::vector<F> result;
        result.reserve(n);

        F g = F::two_adic_generator(log_n);
        F cur = shift;
        for (size_t i = 0; i < n; ++i) {
            result.push_back(cur);
            cur = cur * g;
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Natural domain for a polynomial of the given degree.
    //
    // Returns the smallest power-of-two-sized coset at the identity shift
    // (shift = F::one_val()) that is large enough to uniquely represent a
    // polynomial of that degree.
    //
    // For degree == 0 we return a size-1 domain (log_n = 0).
    // -----------------------------------------------------------------------
    static TwoAdicMultiplicativeCoset natural_domain_for_degree(size_t degree) {
        size_t log_n = p3_util::log2_ceil_usize(degree + 1);
        return TwoAdicMultiplicativeCoset(log_n, F::one_val());
    }

    // -----------------------------------------------------------------------
    // Equality
    // -----------------------------------------------------------------------
    bool operator==(const TwoAdicMultiplicativeCoset& other) const {
        return log_n == other.log_n && shift == other.shift;
    }

    bool operator!=(const TwoAdicMultiplicativeCoset& other) const {
        return !(*this == other);
    }
};

} // namespace p3_commit

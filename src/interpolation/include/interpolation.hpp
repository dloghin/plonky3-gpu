#pragma once

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <cstdint>

/**
 * @file interpolation.hpp
 * @brief Barycentric Lagrange interpolation on multiplicative cosets.
 *
 * Ported from plonky3/interpolation/src/lib.rs.
 *
 * Given evaluations of a polynomial f on a coset gH (where H is a
 * multiplicative subgroup of order n), these routines recover f(z)
 * for an arbitrary point z using the barycentric formula:
 *
 *   f(z) = (z^n - g^n) * sum_i [ w_i * f(g*h_i) / (z - g*h_i) ]
 *
 * where the barycentric weights w_i = diff_invs[i] = 1 / prod_{j!=i}(g*h_i - g*h_j)
 * can be precomputed once per coset and reused for many evaluation points.
 */

namespace p3_interpolation {

/**
 * @brief Batch multiplicative inverse using Montgomery's trick.
 *
 * Computes inv[i] = 1 / values[i] for every i in O(n) multiplications
 * and a single field inversion (instead of n inversions).
 *
 * Algorithm:
 *   1. Prefix products: P[i] = values[0] * ... * values[i]
 *   2. Invert P[n-1]
 *   3. Walk backwards: inv[i] = P[i-1] * running_inverse
 *      then update running_inverse *= values[i]
 *
 * @tparam F  Field type with operator*, inv().
 * @param values  Non-empty vector of non-zero field elements.
 * @return Vector of inverses, same length as values.
 */
template <typename F>
std::vector<F> batch_multiplicative_inverse(const std::vector<F>& values) {
    size_t n = values.size();
    if (n == 0) return {};

    // Step 1: store prefix products directly in result to avoid extra allocation.
    std::vector<F> result(n);
    result[0] = values[0];
    for (size_t i = 1; i < n; ++i) {
        result[i] = result[i - 1] * values[i];
    }

    // Step 2: invert the product of all values
    F running_inv = result[n - 1].inv();

    // Step 3: walk backwards
    for (size_t i = n - 1; i > 0; --i) {
        // inv[i] = prefix[i-1] * running_inv  (= 1 / values[i])
        result[i] = result[i - 1] * running_inv;
        // running_inv = running_inv * values[i]  (= 1 / prefix[i-1])
        running_inv = running_inv * values[i];
    }
    result[0] = running_inv;

    return result;
}

/**
 * @brief Precompute barycentric weights for the coset gH.
 *
 * For a multiplicative subgroup H = {h_0, ..., h_{n-1}} of order n and
 * shift g, the barycentric weight for the i-th coset point g*h_i is:
 *
 *   diff_invs[i] = 1 / prod_{j != i}(g*h_i - g*h_j)
 *
 * Since h_i^n = 1 for all h_i in H, the vanishing polynomial of H is
 * V_H(x) = x^n - 1 with derivative V_H'(x) = n*x^{n-1}.  Evaluating
 * V_H' at h_i gives n*h_i^{n-1} = n/h_i, so:
 *
 *   prod_{j!=i}(h_i - h_j) = n / h_i
 *   prod_{j!=i}(g*h_i - g*h_j) = g^{n-1} * n / h_i
 *   diff_invs[i] = h_i / (n * g^{n-1})
 *
 * This reduces to a single field inversion plus n multiplications.
 *
 * @tparam F  Base field type (e.g. BabyBear).
 * @param subgroup  Elements of H (must all satisfy h^n = 1).
 * @param shift     Coset shift g (non-zero).
 * @return diff_invs vector of length n.
 */
template <typename F>
std::vector<F> compute_diff_invs(const std::vector<F>& subgroup, F shift) {
    size_t n = subgroup.size();
    if (n == 0) return {};

    // n as a field element (safe since n <= 2^27 < p for BabyBear)
    F n_f(static_cast<uint32_t>(n));

    // common_inv = 1 / (n * g^{n-1})
    F shift_pow_nm1 = shift.exp_u64(static_cast<uint64_t>(n) - 1u);
    F common_inv = (n_f * shift_pow_nm1).inv();

    // diff_invs[i] = h_i * common_inv
    std::vector<F> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = subgroup[i] * common_inv;
    }
    return result;
}

/**
 * @brief Barycentric Lagrange interpolation on a coset, using precomputed weights.
 *
 * Given:
 *   - evals[i] = f(shift * subgroup[i])   (polynomial evaluations on coset)
 *   - shift                                (coset shift g)
 *   - point                                (target evaluation point z)
 *   - subgroup                             (elements h_0,...,h_{n-1} of H)
 *   - diff_invs                            (from compute_diff_invs)
 *
 * Returns f(z) using the formula:
 *
 *   f(z) = (z^n - g^n) * sum_i [ diff_invs[i] * f(g*h_i) / (z - g*h_i) ]
 *
 * The point z must not lie on the coset gH (otherwise z - g*h_i = 0).
 * In FRI usage z is always a challenge from the extension field, so this
 * is guaranteed in practice.
 *
 * @tparam F   Base field type (e.g. BabyBear).
 * @tparam EF  Extension field type (e.g. BabyBear4), or same as F.
 *             Must support: zero_val(), exp_u64(), operator+, -, *, inv(),
 *             operator*(const F&), and construction from F.
 * @param evals     Polynomial evaluations on the coset (length n).
 * @param shift     Coset shift g (base field element).
 * @param point     Evaluation point z (extension field element).
 * @param subgroup  Subgroup elements H (length n).
 * @param diff_invs Precomputed barycentric weights (length n).
 * @return f(z) as an extension field element.
 */
template <typename F, typename EF>
EF interpolate_coset_with_precomputation(
    const std::vector<EF>& evals,
    F shift,
    EF point,
    const std::vector<F>& subgroup,
    const std::vector<F>& diff_invs)
{
    size_t n = evals.size();
    if (n == 0) {
        throw std::invalid_argument("interpolate_coset_with_precomputation: empty evals");
    }
    if (subgroup.size() != n) {
        throw std::invalid_argument("interpolate_coset_with_precomputation: subgroup size mismatch");
    }
    if (diff_invs.size() != n) {
        throw std::invalid_argument("interpolate_coset_with_precomputation: diff_invs size mismatch");
    }

    // Compute (z - g*h_i) for each coset point, then batch-invert.
    std::vector<EF> diffs(n);
    for (size_t i = 0; i < n; ++i) {
        F coset_pt = shift * subgroup[i];   // g * h_i  (base field)
        diffs[i] = point - EF(coset_pt);   // z - g*h_i  (extension field)
    }

    std::vector<EF> diffs_inv = batch_multiplicative_inverse(diffs);

    // Accumulate: sum_i [ diff_invs[i] * evals[i] * diffs_inv[i] ]
    EF sum = EF::zero_val();
    for (size_t i = 0; i < n; ++i) {
        // diff_invs[i] is F; multiply EF eval by F scalar first.
        EF term = (evals[i] * diff_invs[i]) * diffs_inv[i];
        sum += term;
    }

    // Vanishing polynomial: z^n - g^n
    EF z_pow_n    = point.exp_u64(static_cast<uint64_t>(n));
    F  g_pow_n    = shift.exp_u64(static_cast<uint64_t>(n));
    EF vanishing  = z_pow_n - EF(g_pow_n);

    return vanishing * sum;
}

} // namespace p3_interpolation

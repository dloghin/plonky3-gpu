#pragma once

/**
 * @file fri_fold_cuda.hpp
 * @brief CUDA-accelerated FRI fold_matrix operation.
 *
 * Provides GPU-backed fold_matrix_cuda<Val, Challenge>() that mirrors the
 * CPU TwoAdicFriFolding::fold_matrix() in fri_folding.hpp.
 *
 * Each fold reduces a vector of length N to N/arity by applying the Lagrange
 * interpolation formula to each row of the evaluation matrix.
 *
 * When compiled without CUDA (P3_CUDA_ENABLED == 0) or when no GPU is present
 * at runtime, all calls transparently fall back to the CPU implementation.
 *
 * Supported arities: 2 (log_arity=1), 4 (log_arity=2), 8 (log_arity=3).
 * Larger arities automatically fall back to CPU.
 */

#include "fri_folding.hpp"
#include "cuda_compat.hpp"
#include "p3_util/util.hpp"

#include <vector>
#include <cstddef>
#include <cstdint>

#if P3_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace p3_fri {

// ============================================================
// Device utility
// ============================================================

#if P3_CUDA_ENABLED

/**
 * @brief Reverse the lowest `bit_len` bits of `x` (device version).
 *
 * Mirrors p3_util::reverse_bits_len for use inside CUDA kernels.
 */
__device__ __forceinline__ size_t fri_reverse_bits(size_t x, size_t bit_len) {
    size_t result = 0;
    for (size_t i = 0; i < bit_len; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// ============================================================
// Arity-2 specialised kernel
// ============================================================

/**
 * @brief Fold kernel specialised for arity 2.
 *
 * For arity 2 the fold formula simplifies to:
 *   output[i] = (ev0 + ev1)/2 + beta * (ev0 - ev1) / (2*x)
 *
 * where:
 *   ev0 = input[2*i],  ev1 = input[2*i+1]
 *   x   = omega^(reverse_bits(i, log_folded_height))
 *   omega = two_adic_generator(log_height)   (log_height = log_folded_height + 1)
 *
 * Each thread handles one output element.
 *
 * @tparam F   Base (prime) field type.
 * @tparam EF  Challenge (extension) field type.
 *
 * @param input       Row-major matrix [n rows × 2 cols], length 2*n.
 * @param output      Result vector of length n.
 * @param beta        Folding challenge (extension field element).
 * @param n           Number of output elements (= input rows).
 * @param log_height  log2 of the unfolded domain size (log_folded_height + 1).
 */
template <typename F, typename EF>
__global__ void fri_fold_arity2_kernel(
    const EF* __restrict__ input,
    EF* __restrict__ output,
    EF   beta,
    size_t n,
    size_t log_height)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    size_t log_folded = log_height - 1;

    // x = omega^(bit_rev(i, log_folded)), where omega is the primitive
    // 2^log_height-th root of unity in F.
    size_t x_pow   = fri_reverse_bits(i, log_folded);
    F      omega   = F::two_adic_generator(log_height);
    F      x_val   = omega.exp_u64(static_cast<uint64_t>(x_pow));
    F      x_inv   = x_val.inv();

    EF ev0 = input[2 * i];
    EF ev1 = input[2 * i + 1];

    // Compute: sum/2 + beta * diff/2 * x^{-1}
    // Using scalar multiplication (EF * F) to avoid full EF-EF multiply for x_inv.
    EF sum_h  = (ev0 + ev1).halve();
    EF diff_h = (ev0 - ev1).halve();

    output[i] = sum_h + beta * (diff_h * x_inv);
}

// ============================================================
// General-arity kernel
// ============================================================

/**
 * @brief General-arity fold kernel using Lagrange interpolation.
 *
 * For arity a = 2^log_arity, each row's `a` evaluations are interpolated
 * at `beta` using the same closed-form Lagrange formula as the CPU
 * TwoAdicFriFolding::fold_row.
 *
 * Algorithm per thread i:
 *  1. Compute x = omega^(bit_rev(i, log_folded_height)).
 *  2. Build evaluation nodes t[j] = x * w^j (j=0..a-1), bit-reversed to
 *     match the committed data ordering.
 *  3. Compute the full numerator product: full_num = prod_j(beta - t[j]).
 *  4. For each j, compute L_j = full_num * arity_inv / ((beta-t[j]) * x^{a-1} * w_inv^j).
 *  5. result = sum_j evals[j] * L_j.
 *
 * @tparam F          Base field type.
 * @tparam EF         Challenge field type.
 * @tparam MAX_ARITY  Compile-time upper bound on arity (must equal 2^log_arity passed at runtime).
 *
 * @param input       Row-major matrix [n × arity], length n*arity.
 * @param output      Result vector of length n.
 * @param beta        Folding challenge.
 * @param n           Number of output rows.
 * @param log_height  log2 of the unfolded domain size.
 * @param log_arity   log2 of the folding arity (must be <= log2(MAX_ARITY)).
 */
template <typename F, typename EF, size_t MAX_ARITY>
__global__ void fri_fold_kernel(
    const EF* __restrict__ input,
    EF* __restrict__ output,
    EF     beta,
    size_t n,
    size_t log_height,
    size_t log_arity)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    size_t arity      = size_t(1) << log_arity;
    size_t log_folded = log_height - log_arity;

    // x = omega^(bit_rev(i, log_folded)), where omega = two_adic_generator(log_height).
    size_t x_pow_idx = fri_reverse_bits(i, log_folded);
    F      omega     = F::two_adic_generator(log_height);
    F      x_val     = omega.exp_u64(static_cast<uint64_t>(x_pow_idx));

    // w = primitive arity-th root of unity.
    F w_val = F::two_adic_generator(log_arity);

    // Build evaluation nodes t[j] = x * w^j (j=0..arity-1).
    EF t[MAX_ARITY];
    {
        EF x_ef = EF::from_base(x_val);
        F w_pow = F::one_val();
        for (size_t j = 0; j < arity; ++j) {
            t[j] = x_ef * w_pow;          // EF * F (scalar mult)
            w_pow = w_pow * w_val;
        }
    }

    // Bit-reverse t to match the committed data layout.
    for (size_t j = 0; j < arity; ++j) {
        size_t k = fri_reverse_bits(j, log_arity);
        if (j < k) {
            EF tmp = t[j]; t[j] = t[k]; t[k] = tmp;
        }
    }

    // Load evaluations from this row.
    EF evals[MAX_ARITY];
    for (size_t j = 0; j < arity; ++j) {
        evals[j] = input[i * arity + j];
    }

    // Fast path: if beta equals one of the evaluation nodes, return that eval directly.
    for (size_t j = 0; j < arity; ++j) {
        if (beta == t[j]) {
            output[i] = evals[j];
            return;
        }
    }

    // Full numerator: prod_{j=0}^{arity-1}(beta - t[j]).
    EF full_num = EF::one_val();
    for (size_t j = 0; j < arity; ++j) {
        full_num = full_num * (beta - t[j]);
    }

    // arity_inv = (1/2)^log_arity (scalar, computed in base field).
    F arity_inv_val = F::one_val();
    {
        F two_inv = (F::one_val() + F::one_val()).inv();
        for (size_t k = 0; k < log_arity; ++k) {
            arity_inv_val = arity_inv_val * two_inv;
        }
    }

    // x^(arity-1): denominator base contribution from x.
    F x_pow_am1 = x_val.exp_u64(static_cast<uint64_t>(arity - 1));

    // w_inv = w^(arity-1) = w^{-1} (since w^arity = 1).
    F w_inv = w_val.exp_u64(static_cast<uint64_t>(arity - 1));

    // Lagrange interpolation:
    //   L_j(beta) = full_num * arity_inv / ((beta - t[j]) * x^{a-1} * w_inv^j)
    //   result    = sum_j evals[j] * L_j(beta)
    EF result  = EF::zero_val();
    F  w_inv_j = F::one_val();   // w_inv^j, starts at 1
    for (size_t j = 0; j < arity; ++j) {
        // base_denom = x^{a-1} * w_inv^j  (base field scalar)
        F base_denom_val = x_pow_am1 * w_inv_j;

        // denom_full = (beta - t[j]) * base_denom  (EF element)
        EF denom_full = (beta - t[j]) * base_denom_val;   // EF * F scalar mult

        // L_j = full_num * arity_inv / denom_full
        //     = full_num * (arity_inv_val / denom_full)  -- keep arity_inv in F
        EF L_j = full_num * arity_inv_val * denom_full.inv();

        result = result + evals[j] * L_j;

        w_inv_j = w_inv_j * w_inv;
    }

    output[i] = result;
}

/**
 * @brief Run the FRI fold kernels on GPU-resident buffers (launch + synchronize only).
 *
 * Does not allocate, copy host data, or free. Use with @p d_input / @p d_output already
 * sized for the fold: @p d_input has @c n * (1<<log_arity) elements, @p d_output has @p n.
 *
 * @pre CUDA is enabled, a device exists, and @p log_arity is in [1,3].
 */
template <typename Val, typename Challenge>
void fold_matrix_cuda_device(
    const Challenge* d_input,
    Challenge*       d_output,
    const Challenge& beta,
    size_t           log_arity,
    size_t           log_height,
    size_t           n)
{
    constexpr size_t BLOCK_SIZE = 256;
    size_t           nblocks     = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (log_arity == 1) {
        fri_fold_arity2_kernel<Val, Challenge>
            <<<nblocks, BLOCK_SIZE>>>(d_input, d_output, beta, n, log_height);
    } else if (log_arity == 2) {
        fri_fold_kernel<Val, Challenge, 4>
            <<<nblocks, BLOCK_SIZE>>>(d_input, d_output, beta, n, log_height, log_arity);
    } else {
        // log_arity == 3
        fri_fold_kernel<Val, Challenge, 8>
            <<<nblocks, BLOCK_SIZE>>>(d_input, d_output, beta, n, log_height, log_arity);
    }

    P3_CUDA_CHECK(cudaGetLastError());
    P3_CUDA_CHECK(cudaDeviceSynchronize());
}

#endif  // P3_CUDA_ENABLED

// ============================================================
// Host-side API
// ============================================================

/**
 * @brief Fold an evaluation matrix, using GPU acceleration when available.
 *
 * Reshapes `current` as a [height × arity] matrix and folds each row using
 * the two-adic FRI Lagrange interpolation formula at challenge `beta`.
 *
 * Falls back to TwoAdicFriFolding::fold_matrix (CPU) when:
 *   - CUDA is not compiled in (P3_CUDA_ENABLED == 0), or
 *   - no CUDA device is found at runtime, or
 *   - log_arity > 3 (arities > 8 not supported on GPU).
 *
 * @tparam Val       Base prime field type (e.g. BabyBear).
 * @tparam Challenge Challenge (extension) field type (e.g. BabyBear4).
 *
 * @param beta        Folding challenge.
 * @param log_arity   log2 of the folding arity (1=arity-2, 2=arity-4, 3=arity-8).
 * @param log_height  log2 of the unfolded domain size (= log(current.size())).
 * @param current     Flat row-major evaluation buffer; length must be divisible by arity.
 *
 * @return Folded vector of length current.size() / arity.
 */
template <typename Val, typename Challenge>
std::vector<Challenge> fold_matrix_cuda(
    const Challenge&              beta,
    size_t                        log_arity,
    size_t                        log_height,
    const std::vector<Challenge>& current)
{
#if P3_CUDA_ENABLED
    if (log_arity >= 1 && log_arity <= 3) {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err == cudaSuccess && device_count > 0) {
            size_t arity  = size_t(1) << log_arity;
            size_t n      = current.size() / arity;  // number of output rows

            std::vector<Challenge> output(n);

            Challenge* d_input  = nullptr;
            Challenge* d_output = nullptr;

            P3_CUDA_CHECK(cudaMalloc(&d_input,  current.size() * sizeof(Challenge)));
            P3_CUDA_CHECK(cudaMalloc(&d_output, n              * sizeof(Challenge)));
            P3_CUDA_CHECK(cudaMemcpy(d_input, current.data(),
                                     current.size() * sizeof(Challenge),
                                     cudaMemcpyHostToDevice));

            fold_matrix_cuda_device<Val, Challenge>(
                d_input, d_output, beta, log_arity, log_height, n);

            P3_CUDA_CHECK(cudaMemcpy(output.data(), d_output,
                                     n * sizeof(Challenge),
                                     cudaMemcpyDeviceToHost));
            P3_CUDA_CHECK(cudaFree(d_input));
            P3_CUDA_CHECK(cudaFree(d_output));
            return output;
        }
    }
#endif

    // CPU fallback.
    return TwoAdicFriFolding<Val, Challenge>::fold_matrix(
        log_height, log_arity, beta, current);
}

}  // namespace p3_fri

#pragma once

/**
 * @file ntt_cuda.hpp
 * @brief CUDA-accelerated NTT (Number Theoretic Transform) for batch polynomial evaluation.
 *
 * Provides a GPU-backed drop-in replacement for the Radix2Dit CPU NTT.
 * When compiled without CUDA (P3_CUDA_ENABLED == 0), all methods fall back
 * transparently to the CPU Radix2Dit implementation.
 *
 * Implements:
 *   - Forward / inverse NTT  (dft_batch / idft_batch)
 *   - Coset DFT / IDFT       (coset_dft_batch / coset_idft_batch)
 *   - Low-degree extension   (coset_lde_batch)
 *
 * Algorithm: iterative Cooley-Tukey DIT (bit-reversal then butterfly layers).
 *
 * Memory layout: row-major matrix, height × width.  Each column is an
 * independent polynomial.  The NTT operates along the height dimension.
 */

#include "cuda_compat.hpp"
#include "radix2_dit.hpp"
#include "dense_matrix.hpp"
#include "p3_util/util.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#if P3_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace p3_dft {

// ============================================================
// Device utility
// ============================================================

#if P3_CUDA_ENABLED

/**
 * @brief Bit-reverse a log_n-bit integer on the device.
 */
__device__ __forceinline__ size_t ntt_bit_reverse(size_t v, size_t log_n) {
    size_t result = 0;
    for (size_t i = 0; i < log_n; ++i) {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    return result;
}

// ============================================================
// CUDA Kernels
// ============================================================

/**
 * @brief Bit-reversal permutation kernel.
 *
 * Permutes rows of a (height × width) matrix according to the bit-reversal
 * of the row index.  Each thread handles one (row, col) pair; swaps are
 * performed only when row < bit_rev(row) to avoid double-swapping.
 */
template<typename F>
__global__ void bit_reverse_kernel(
    F* data,
    size_t height,
    size_t log_n,
    size_t width)
{
    size_t total = height * width;
    size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t col     = idx % width;
    size_t row     = idx / width;
    size_t rev_row = ntt_bit_reverse(row, log_n);

    if (row < rev_row) {
        F tmp                     = data[row     * width + col];
        data[row     * width + col] = data[rev_row * width + col];
        data[rev_row * width + col] = tmp;
    }
}

/**
 * @brief Twiddle-factor precomputation kernel.
 *
 * Computes twiddles[i] = root^i for i in [0, half_n).
 * root must be the primitive 2^log_h-th root of unity.
 */
template<typename F>
__global__ void compute_twiddles_kernel(
    F*     twiddles,
    size_t half_n,
    F      root)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;
    twiddles[idx] = root.exp_u64(static_cast<uint64_t>(idx));
}

/**
 * @brief Single DIT butterfly layer kernel.
 *
 * Performs the l-th butterfly stage of the Cooley-Tukey DIT NTT.
 * For each butterfly pair (upper_row, lower_row) and each column:
 *   a = data[upper * w + col]
 *   b = data[lower * w + col]
 *   data[upper * w + col] = a + twiddle * b
 *   data[lower * w + col] = a - twiddle * b
 *
 * @param data     In-place transform buffer (height × width, row-major)
 * @param twiddles Precomputed twiddle factors (n/2 elements)
 * @param n        Transform height (power of 2)
 * @param width    Batch width (number of independent polynomials)
 * @param layer    Current butterfly layer index (0 .. log_n-1)
 */
template<typename F>
__global__ void ntt_dit_kernel(
    F*           data,
    const F*     twiddles,
    size_t       n,
    size_t       width,
    size_t       layer)
{
    size_t half      = static_cast<size_t>(1) << layer;       // half-block size
    size_t stride_tw = (n >> 1) >> layer;                     // twiddle stride

    size_t n_butterflies = n >> 1;                            // total butterfly pairs
    size_t total         = n_butterflies * width;
    size_t idx           = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t col       = idx % width;
    size_t butterfly = idx / width;

    size_t group = butterfly / half;
    size_t j     = butterfly % half;

    size_t upper = (group * 2 * half) + j;
    size_t lower = upper + half;

    F a  = data[upper * width + col];
    F b  = data[lower * width + col];
    F tw = twiddles[j * stride_tw];
    F tb = tw * b;

    data[upper * width + col] = a + tb;
    data[lower * width + col] = a - tb;
}

/**
 * @brief Element-wise scalar multiplication kernel.
 *
 * Multiplies every element of a flat array by scalar.
 */
template<typename F>
__global__ void scale_kernel(F* data, size_t total, F scalar) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = data[idx] * scalar;
}

/**
 * @brief Reverse rows [1 .. height-1] kernel (in-place).
 *
 * Used as part of the inverse NTT: after the forward DFT, reversing
 * rows 1..h-1 turns DFT[k] into DFT[(n-k) mod n] at position k.
 *
 * Each thread handles one (pair_index, col) and swaps:
 *   row lo = pair_index + 1  <-->  row hi = height - 1 - pair_index
 */
template<typename F>
__global__ void reverse_rows_kernel(F* data, size_t height, size_t width) {
    size_t n_pairs = (height - 1) / 2;
    size_t total   = n_pairs * width;
    size_t idx     = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t col      = idx % width;
    size_t pair_idx = idx / width;

    size_t lo = pair_idx + 1;
    size_t hi = height - 1 - pair_idx;

    if (lo < hi) {
        F tmp                   = data[lo * width + col];
        data[lo * width + col]  = data[hi * width + col];
        data[hi * width + col]  = tmp;
    }
}

/**
 * @brief Coset twist kernel: multiply row i by base_shift^i, i in [1, height).
 *
 * Pass base_shift = shift  for the forward coset twist.
 * Pass base_shift = shift.inv()  for the inverse coset twist.
 * Row 0 is always multiplied by 1, so it is skipped.
 */
template<typename F>
__global__ void coset_twist_kernel(
    F*     data,
    const F* row_powers,
    size_t height,
    size_t width,
    F      /*base_shift*/)
{
    size_t total = (height - 1) * width;
    size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t col = idx % width;
    size_t row = idx / width + 1;    // rows 1 .. height-1

    // row_powers stores base_shift^row at index row (row 1..height-1).
    data[row * width + col] = data[row * width + col] * row_powers[row];
}

/**
 * @brief Precompute row powers for coset twisting: row_powers[row] = base_shift^row.
 *
 * Computes entries for row in [1, height). row_powers[0] is unused (set to 1 by caller if needed).
 */
template<typename F>
__global__ void compute_row_powers_kernel(F* row_powers, size_t height, F base_shift) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row == 0 || row >= height) return;
    row_powers[row] = base_shift.exp_u64(static_cast<uint64_t>(row));
}

/**
 * @brief Zero-fill rows [orig_height .. new_height) of a device buffer.
 */
template<typename F>
__global__ void zero_pad_kernel(
    F*     data,
    size_t orig_height,
    size_t new_height,
    size_t width)
{
    size_t total = (new_height - orig_height) * width;
    size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t col = idx % width;
    size_t row = idx / width + orig_height;

    data[row * width + col] = F::zero_val();
}

#endif  // P3_CUDA_ENABLED

// ============================================================
// NttCuda — host-side GPU-accelerated NTT class
// ============================================================

/**
 * @brief Batched NTT with optional GPU acceleration.
 *
 * Mirrors the Radix2Dit<F> API so it can be used as a drop-in replacement.
 * Falls back to Radix2Dit<F> when CUDA is unavailable or when the GPU is
 * not detected at runtime.
 *
 * @tparam F  Field element type (must have P3_HOST_DEVICE arithmetic and
 *            two_adic_generator).
 */
template<typename F>
class NttCuda {
public:
    explicit NttCuda(bool prefer_gpu = true) : use_gpu_(false) {
#if P3_CUDA_ENABLED
        if (prefer_gpu) {
            int device_count = 0;
            P3_CUDA_CHECK(cudaGetDeviceCount(&device_count));
            use_gpu_ = (device_count > 0);
        }
#else
        (void)prefer_gpu;
#endif
    }

    ~NttCuda() {
#if P3_CUDA_ENABLED
        for (auto& kv : d_twiddles_) {
            // Destructors must not throw; best-effort free.
            (void)cudaFree(kv.second);
        }
#endif
    }

    // Non-copyable; twiddle cache holds device pointers.
    NttCuda(const NttCuda&)            = delete;
    NttCuda& operator=(const NttCuda&) = delete;

    bool using_gpu() const { return use_gpu_; }

    // -----------------------------------------------------------------------
    // Forward DFT
    // -----------------------------------------------------------------------

    p3_matrix::RowMajorMatrix<F> dft_batch(p3_matrix::RowMajorMatrix<F> mat) {
#if P3_CUDA_ENABLED
        if (use_gpu_ && mat.height() > 1) {
            return dft_batch_gpu(std::move(mat));
        }
#endif
        return cpu_dft_.dft_batch(std::move(mat));
    }

    // -----------------------------------------------------------------------
    // Inverse DFT
    // -----------------------------------------------------------------------

    p3_matrix::RowMajorMatrix<F> idft_batch(p3_matrix::RowMajorMatrix<F> mat) {
#if P3_CUDA_ENABLED
        if (use_gpu_ && mat.height() > 1) {
            return idft_batch_gpu(std::move(mat));
        }
#endif
        return cpu_dft_.idft_batch(std::move(mat));
    }

    // -----------------------------------------------------------------------
    // Coset DFT / IDFT
    // -----------------------------------------------------------------------

    p3_matrix::RowMajorMatrix<F> coset_dft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
#if P3_CUDA_ENABLED
        if (use_gpu_ && mat.height() > 1) {
            return coset_dft_batch_gpu(std::move(mat), shift);
        }
#endif
        return cpu_dft_.coset_dft_batch(std::move(mat), shift);
    }

    p3_matrix::RowMajorMatrix<F> coset_idft_batch(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
#if P3_CUDA_ENABLED
        if (use_gpu_ && mat.height() > 1) {
            return coset_idft_batch_gpu(std::move(mat), shift);
        }
#endif
        return cpu_dft_.coset_idft_batch(std::move(mat), shift);
    }

    // -----------------------------------------------------------------------
    // Low-degree extension
    // -----------------------------------------------------------------------

    /**
     * @brief Coset low-degree extension (LDE).
     *
     * Takes evaluations on the canonical 2^log_h-th subgroup, recovers
     * polynomial coefficients via INTT, zero-pads to height × 2^added_bits,
     * then evaluates on the coset  shift × {ω'^0, ..., ω'^{h·2^added_bits-1}}.
     */
    p3_matrix::RowMajorMatrix<F> coset_lde_batch(
        p3_matrix::RowMajorMatrix<F> mat, size_t added_bits, const F& shift)
    {
#if P3_CUDA_ENABLED
        if (use_gpu_ && mat.height() > 1) {
            return coset_lde_batch_gpu(std::move(mat), added_bits, shift);
        }
#endif
        return cpu_dft_.coset_lde_batch(std::move(mat), added_bits, shift);
    }

private:
    bool          use_gpu_;
    Radix2Dit<F>  cpu_dft_;

#if P3_CUDA_ENABLED
    // Cache: log_h -> device pointer to n/2 twiddle factors
    std::unordered_map<size_t, F*> d_twiddles_;

    static constexpr size_t BLOCK_SIZE = 256;

    // ------------------------------------------------------------------
    // Twiddle-factor management
    // ------------------------------------------------------------------

    /**
     * @brief Return (or compute and cache on device) twiddle factors for 2^log_h.
     *
     * Returns a device pointer to n/2 factors: root^0, root^1, ..., root^(n/2-1)
     * where root is the primitive n-th root of unity for F.
     */
    F* get_or_compute_twiddles_gpu(size_t log_h) {
        auto it = d_twiddles_.find(log_h);
        if (it != d_twiddles_.end()) return it->second;

        size_t n    = static_cast<size_t>(1) << log_h;
        size_t half = n >> 1;

        F* d_tw;
        P3_CUDA_CHECK(cudaMalloc(&d_tw, half * sizeof(F)));

        F root = TwoAdicFieldTraits<F>::two_adic_generator(log_h);

        size_t nblocks = (half + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_twiddles_kernel<<<nblocks, BLOCK_SIZE>>>(d_tw, half, root);
        P3_CUDA_CHECK(cudaGetLastError());

        d_twiddles_[log_h] = d_tw;
        return d_tw;
    }

    // ------------------------------------------------------------------
    // Primitive GPU operations (work on device pointers, no H-D copies)
    // ------------------------------------------------------------------

    /**
     * @brief Run the forward NTT in-place on d_data (height × width).
     *
     * Steps: bit-reversal permutation then log_h butterfly layers.
     * Queued into the default CUDA stream; caller must sync before use.
     */
    void do_forward_ntt(F* d_data, size_t height, size_t width) {
        size_t log_h = p3_util::log2_strict_usize(height);
        size_t total = height * width;

        // Bit-reversal permutation
        {
            size_t nblocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            bit_reverse_kernel<<<nblocks, BLOCK_SIZE>>>(d_data, height, log_h, width);
            P3_CUDA_CHECK(cudaGetLastError());
        }

        // Butterfly layers
        F* d_tw = get_or_compute_twiddles_gpu(log_h);
        {
            size_t n_butterflies = height >> 1;
            size_t layer_total   = n_butterflies * width;
            size_t nblocks       = (layer_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (size_t l = 0; l < log_h; ++l) {
                ntt_dit_kernel<<<nblocks, BLOCK_SIZE>>>(d_data, d_tw, height, width, l);
                P3_CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    /**
     * @brief Apply the row-reversal and 1/n scaling steps of the inverse NTT.
     *
     * After do_forward_ntt, this completes the IDFT.
     * Queued into the default CUDA stream; caller must sync before use.
     */
    void do_reverse_scale(F* d_data, size_t height, size_t width, size_t log_h) {
        size_t total   = height * width;
        size_t n_pairs = (height - 1) / 2;

        // Reverse rows [1 .. height-1]
        if (n_pairs > 0) {
            size_t rev_total = n_pairs * width;
            size_t nblocks   = (rev_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            reverse_rows_kernel<<<nblocks, BLOCK_SIZE>>>(d_data, height, width);
            P3_CUDA_CHECK(cudaGetLastError());
        }

        // Scale by 1/n
        F inv_n  = compute_inv_n(log_h);
        {
            size_t nblocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            scale_kernel<<<nblocks, BLOCK_SIZE>>>(d_data, total, inv_n);
            P3_CUDA_CHECK(cudaGetLastError());
        }
    }

    /**
     * @brief Apply the coset twist: multiply row i by base_shift^i, i in [1,h).
     *
     * Queued into the default CUDA stream; caller must sync before use.
     */
    void do_coset_twist(F* d_data, size_t height, size_t width, const F& base_shift) {
        if (height <= 1) return;
        // Precompute base_shift^row once per row, then apply across all columns.
        F* d_row_powers = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&d_row_powers, height * sizeof(F)));

        // row 0 is unused; set it to 1 for safety.
        {
            F one = F::one_val();
            P3_CUDA_CHECK(cudaMemcpy(d_row_powers, &one, sizeof(F), cudaMemcpyHostToDevice));
        }

        {
            size_t nblocks = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
            compute_row_powers_kernel<<<nblocks, BLOCK_SIZE>>>(d_row_powers, height, base_shift);
            P3_CUDA_CHECK(cudaGetLastError());
        }

        {
            size_t twist_total = (height - 1) * width;
            size_t nblocks     = (twist_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            coset_twist_kernel<<<nblocks, BLOCK_SIZE>>>(d_data, d_row_powers, height, width, base_shift);
            P3_CUDA_CHECK(cudaGetLastError());
        }

        P3_CUDA_CHECK(cudaFree(d_row_powers));
    }

    // ------------------------------------------------------------------
    // Public operation implementations (handle H-D-H transfers)
    // ------------------------------------------------------------------

    p3_matrix::RowMajorMatrix<F> dft_batch_gpu(p3_matrix::RowMajorMatrix<F> mat) {
        size_t h     = mat.height();
        size_t w     = mat.width();
        size_t total = h * w;

        F* d_data;
        P3_CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(F)));
        P3_CUDA_CHECK(cudaMemcpy(d_data, mat.values.data(), total * sizeof(F), cudaMemcpyHostToDevice));

        do_forward_ntt(d_data, h, w);

        P3_CUDA_CHECK(cudaDeviceSynchronize());
        P3_CUDA_CHECK(cudaMemcpy(mat.values.data(), d_data, total * sizeof(F), cudaMemcpyDeviceToHost));
        P3_CUDA_CHECK(cudaFree(d_data));
        return mat;
    }

    p3_matrix::RowMajorMatrix<F> idft_batch_gpu(p3_matrix::RowMajorMatrix<F> mat) {
        size_t h     = mat.height();
        size_t w     = mat.width();
        size_t log_h = p3_util::log2_strict_usize(h);
        size_t total = h * w;

        F* d_data;
        P3_CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(F)));
        P3_CUDA_CHECK(cudaMemcpy(d_data, mat.values.data(), total * sizeof(F), cudaMemcpyHostToDevice));

        do_forward_ntt(d_data, h, w);
        do_reverse_scale(d_data, h, w, log_h);

        P3_CUDA_CHECK(cudaDeviceSynchronize());
        P3_CUDA_CHECK(cudaMemcpy(mat.values.data(), d_data, total * sizeof(F), cudaMemcpyDeviceToHost));
        P3_CUDA_CHECK(cudaFree(d_data));
        return mat;
    }

    p3_matrix::RowMajorMatrix<F> coset_dft_batch_gpu(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        size_t h     = mat.height();
        size_t w     = mat.width();
        size_t total = h * w;

        F* d_data;
        P3_CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(F)));
        P3_CUDA_CHECK(cudaMemcpy(d_data, mat.values.data(), total * sizeof(F), cudaMemcpyHostToDevice));

        do_coset_twist(d_data, h, w, shift);
        do_forward_ntt(d_data, h, w);

        P3_CUDA_CHECK(cudaDeviceSynchronize());
        P3_CUDA_CHECK(cudaMemcpy(mat.values.data(), d_data, total * sizeof(F), cudaMemcpyDeviceToHost));
        P3_CUDA_CHECK(cudaFree(d_data));
        return mat;
    }

    p3_matrix::RowMajorMatrix<F> coset_idft_batch_gpu(
        p3_matrix::RowMajorMatrix<F> mat, const F& shift)
    {
        size_t h     = mat.height();
        size_t w     = mat.width();
        size_t log_h = p3_util::log2_strict_usize(h);
        size_t total = h * w;

        F inv_shift = shift.inv();    // Compute on host before GPU operations

        F* d_data;
        P3_CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(F)));
        P3_CUDA_CHECK(cudaMemcpy(d_data, mat.values.data(), total * sizeof(F), cudaMemcpyHostToDevice));

        // IDFT = forward DFT + row reversal + scaling
        do_forward_ntt(d_data, h, w);
        do_reverse_scale(d_data, h, w, log_h);
        // Inverse coset twist: multiply row i by inv_shift^i
        do_coset_twist(d_data, h, w, inv_shift);

        P3_CUDA_CHECK(cudaDeviceSynchronize());
        P3_CUDA_CHECK(cudaMemcpy(mat.values.data(), d_data, total * sizeof(F), cudaMemcpyDeviceToHost));
        P3_CUDA_CHECK(cudaFree(d_data));
        return mat;
    }

    /**
     * @brief GPU LDE: single H→D and D→H transfer covering all three steps.
     *
     * 1. INTT of size orig_h  (first orig_h rows on device)
     * 2. Zero-fill rows [orig_h, new_h)  on device
     * 3. Coset DFT of size new_h  on device
     */
    p3_matrix::RowMajorMatrix<F> coset_lde_batch_gpu(
        p3_matrix::RowMajorMatrix<F> mat, size_t added_bits, const F& shift)
    {
        size_t orig_h    = mat.height();
        size_t w         = mat.width();
        size_t log_orig  = p3_util::log2_strict_usize(orig_h);
        size_t new_h     = orig_h << added_bits;
        size_t total_new = new_h * w;

        // Allocate device buffer large enough for the extended matrix
        F* d_data;
        P3_CUDA_CHECK(cudaMalloc(&d_data, total_new * sizeof(F)));

        // Copy original evaluations to device (first orig_h rows)
        P3_CUDA_CHECK(cudaMemcpy(d_data, mat.values.data(), orig_h * w * sizeof(F),
                                 cudaMemcpyHostToDevice));

        // Step 1: INTT of size orig_h
        do_forward_ntt(d_data, orig_h, w);
        do_reverse_scale(d_data, orig_h, w, log_orig);

        // Step 2: Zero-fill rows [orig_h, new_h)
        {
            size_t pad_total = (new_h - orig_h) * w;
            size_t nblocks   = (pad_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            zero_pad_kernel<<<nblocks, BLOCK_SIZE>>>(d_data, orig_h, new_h, w);
            P3_CUDA_CHECK(cudaGetLastError());
        }

        // Step 3: Coset DFT of size new_h
        //   3a: coset twist (multiply row i by shift^i)
        do_coset_twist(d_data, new_h, w, shift);
        //   3b: forward NTT of size new_h
        do_forward_ntt(d_data, new_h, w);

        // Copy result back to host; resize mat to new_h rows
        mat.values.resize(total_new);
        // (width_ is private; use the public constructor trick via std::vector swap)
        p3_matrix::RowMajorMatrix<F> result(std::move(mat.values), w);
        P3_CUDA_CHECK(cudaDeviceSynchronize());
        P3_CUDA_CHECK(cudaMemcpy(result.values.data(), d_data, total_new * sizeof(F),
                                 cudaMemcpyDeviceToHost));
        P3_CUDA_CHECK(cudaFree(d_data));
        return result;
    }

    // ------------------------------------------------------------------
    // Helper
    // ------------------------------------------------------------------

    /** @brief Compute 1 / 2^log_h in F. */
    static F compute_inv_n(size_t log_h) {
        F two    = F::one_val() + F::one_val();
        F inv2   = two.inv();
        return inv2.exp_u64(static_cast<uint64_t>(log_h));
    }

#endif  // P3_CUDA_ENABLED
};

}  // namespace p3_dft

#pragma once

/**
 * @file merkle_tree_cuda.hpp
 * @brief CUDA-accelerated Merkle tree construction using Poseidon2.
 *
 * Provides GPU-accelerated implementations of:
 *   - Leaf hashing: each row of the input matrix is hashed independently
 *     using the PaddingFreeSponge pattern with Poseidon2.
 *   - Internal node compression: pairs of digests are compressed using
 *     TruncatedPermutation with Poseidon2.
 *   - Full tree construction: host function that orchestrates GPU kernel
 *     launches level-by-level, handling mixed-height matrix injection.
 *
 * When compiled without CUDA (P3_CUDA_ENABLED == 0), a CPU fallback using
 * the same Poseidon2Cuda struct (which has P3_HOST_DEVICE methods) is
 * provided automatically.
 *
 * Target configuration (FRI test):
 *   F = BabyBear, WIDTH = 16, RATE = 8, DIGEST_ELEMS = 8,
 *   ROUNDS_F = 8, ROUNDS_P = 14, D = 5
 */

#include "merkle_tree.hpp"
#include "poseidon2_cuda.hpp"
#include "cuda_compat.hpp"

#if P3_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <vector>

namespace p3_merkle_tree {

using p3_matrix::RowMajorMatrix;

// =============================================================================
// CUDA Kernels (compiled only by nvcc)
// =============================================================================

#if P3_CUDA_ENABLED

/**
 * @brief Hash leaf rows using Poseidon2 sponge (PaddingFreeSponge pattern).
 *
 * Each thread processes one row. The sponge overwrites (not XOR-adds) state
 * positions to match the CPU PaddingFreeSponge::hash_iter implementation.
 *
 * @tparam F            Field element type
 * @tparam WIDTH        Poseidon2 state width
 * @tparam RATE         Elements absorbed per permutation
 * @tparam OUT          Output digest length (== DIGEST_ELEMS)
 * @tparam ROUNDS_F     External rounds
 * @tparam ROUNDS_P     Internal rounds
 * @tparam D            S-box exponent
 *
 * @param poseidon      Device pointer to Poseidon2 constants
 * @param matrix_data   Flat row-major matrix (num_rows * num_cols elements)
 * @param num_rows      Number of rows to hash
 * @param num_cols      Number of columns per row (= total input length per hash)
 * @param digests       Output buffer (num_rows * OUT elements)
 */
template <typename F, size_t WIDTH, size_t RATE, size_t OUT,
          size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
__global__ void hash_leaves_kernel(
    const poseidon2::Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D>* poseidon,
    const F* matrix_data,
    size_t num_rows,
    size_t num_cols,
    F* digests)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    // Initialize sponge state to zero
    F state[WIDTH];
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = F::zero_val();
    }

    const F* row_data = matrix_data + row * num_cols;
    size_t absorbed = 0;

    // PaddingFreeSponge absorb loop: overwrite state[0..i] with each chunk,
    // then permute. Mirrors plonky3/symmetric/src/sponge.rs hash_iter.
    for (;;) {
        size_t i = 0;
        while (i < RATE && absorbed < num_cols) {
            state[i] = row_data[absorbed];
            ++i;
            ++absorbed;
        }
        if (i == RATE) {
            poseidon->permute(state);
            // If we've consumed all input, the next iteration will have i=0
            // and will hit the else-break path.
        } else {
            // Partial or empty chunk: permute only if non-empty, then stop.
            if (i > 0) {
                poseidon->permute(state);
            }
            break;
        }
    }

    // Write OUT elements to digest buffer
    F* digest_out = digests + row * OUT;
    for (size_t i = 0; i < OUT; ++i) {
        digest_out[i] = state[i];
    }
}

/**
 * @brief Compress pairs of digests (binary tree, N=2).
 *
 * Implements TruncatedPermutation::compress for N=2:
 *   state = [left | right | zeros...]
 *   permute(state)
 *   output = state[0..CHUNK]
 *
 * @tparam F        Field element type
 * @tparam WIDTH    Poseidon2 state width (must satisfy 2*CHUNK <= WIDTH)
 * @tparam CHUNK    Digest chunk size (== DIGEST_ELEMS)
 * @tparam ROUNDS_F External rounds
 * @tparam ROUNDS_P Internal rounds
 * @tparam D        S-box exponent
 *
 * @param poseidon      Device pointer to Poseidon2 constants
 * @param input_digests Input: 2 * num_nodes * CHUNK elements (left, right pairs)
 * @param output_digests Output: num_nodes * CHUNK elements
 * @param num_nodes     Number of parent nodes to compute
 */
template <typename F, size_t WIDTH, size_t CHUNK,
          size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
__global__ void compress_layer_kernel(
    const poseidon2::Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D>* poseidon,
    const F* input_digests,
    F* output_digests,
    size_t num_nodes)
{
    size_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    // Initialize state to zero
    F state[WIDTH];
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = F::zero_val();
    }

    // Copy left child into state[0..CHUNK) and right child into state[CHUNK..2*CHUNK)
    const F* left  = input_digests + (2 * node) * CHUNK;
    const F* right = input_digests + (2 * node + 1) * CHUNK;
    for (size_t i = 0; i < CHUNK; ++i) {
        state[i]         = left[i];
        state[CHUNK + i] = right[i];
    }

    // Apply Poseidon2 permutation
    poseidon->permute(state);

    // Write first CHUNK elements as the output digest
    F* out = output_digests + node * CHUNK;
    for (size_t i = 0; i < CHUNK; ++i) {
        out[i] = state[i];
    }
}

/**
 * @brief Hash injected matrix rows and compress with existing node digests.
 *
 * Used when shorter matrices are injected at a tree level above the leaves.
 * For each node i:
 *   1. Hash inject_data[i * inject_cols .. (i+1)*inject_cols) using PaddingFreeSponge.
 *   2. Compress (inout_digests[i], inject_hash) in-place.
 *
 * @tparam F            Field element type
 * @tparam WIDTH        Poseidon2 state width
 * @tparam RATE         Sponge rate
 * @tparam CHUNK        Digest chunk size (== DIGEST_ELEMS)
 * @tparam ROUNDS_F     External rounds
 * @tparam ROUNDS_P     Internal rounds
 * @tparam D            S-box exponent
 *
 * @param poseidon      Device pointer to Poseidon2 constants
 * @param inout_digests Existing node digests (read and overwritten, num_nodes * CHUNK)
 * @param inject_data   Flat inject matrix (num_nodes * inject_cols elements)
 * @param inject_cols   Number of columns in the inject data per node
 * @param num_nodes     Number of nodes at this level
 */
template <typename F, size_t WIDTH, size_t RATE, size_t CHUNK,
          size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
__global__ void inject_and_compress_kernel(
    const poseidon2::Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D>* poseidon,
    F* inout_digests,
    const F* inject_data,
    size_t inject_cols,
    size_t num_nodes)
{
    size_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    // Step 1: Hash inject row using PaddingFreeSponge
    F sponge_state[WIDTH];
    for (size_t i = 0; i < WIDTH; ++i) {
        sponge_state[i] = F::zero_val();
    }

    const F* row_data = inject_data + node * inject_cols;
    size_t absorbed = 0;

    for (;;) {
        size_t i = 0;
        while (i < RATE && absorbed < inject_cols) {
            sponge_state[i] = row_data[absorbed];
            ++i;
            ++absorbed;
        }
        if (i == RATE) {
            poseidon->permute(sponge_state);
        } else {
            if (i > 0) {
                poseidon->permute(sponge_state);
            }
            break;
        }
    }
    // sponge_state[0..CHUNK) now holds the inject hash

    // Step 2: Compress (existing_node_digest, inject_hash) in-place
    F compress_state[WIDTH];
    for (size_t i = 0; i < WIDTH; ++i) {
        compress_state[i] = F::zero_val();
    }

    const F* existing = inout_digests + node * CHUNK;
    for (size_t i = 0; i < CHUNK; ++i) {
        compress_state[i]         = existing[i];
        compress_state[CHUNK + i] = sponge_state[i];
    }

    poseidon->permute(compress_state);

    // Write result back in-place
    F* out = inout_digests + node * CHUNK;
    for (size_t i = 0; i < CHUNK; ++i) {
        out[i] = compress_state[i];
    }
}

#endif // P3_CUDA_ENABLED

// =============================================================================
// CPU Helper Functions (used by fallback and for clarity)
// =============================================================================

namespace detail {

/**
 * @brief Hash a flat array of field elements using PaddingFreeSponge pattern.
 * Runs on CPU using Poseidon2Cuda (which has P3_HOST_DEVICE methods).
 */
template <typename F, size_t WIDTH, size_t RATE, size_t OUT,
          size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
inline std::array<F, OUT> hash_row_cpu(
    const poseidon2::Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D>& poseidon,
    const F* row_data,
    size_t num_cols)
{
    F state[WIDTH];
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = F::zero_val();
    }

    size_t absorbed = 0;
    for (;;) {
        size_t i = 0;
        while (i < RATE && absorbed < num_cols) {
            state[i] = row_data[absorbed];
            ++i;
            ++absorbed;
        }
        if (i == RATE) {
            poseidon.permute(state);
        } else {
            if (i > 0) {
                poseidon.permute(state);
            }
            break;
        }
    }

    std::array<F, OUT> out;
    for (size_t i = 0; i < OUT; ++i) out[i] = state[i];
    return out;
}

/**
 * @brief Compress two digest arrays using TruncatedPermutation pattern.
 * Runs on CPU using Poseidon2Cuda.
 */
template <typename F, size_t WIDTH, size_t CHUNK,
          size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
inline std::array<F, CHUNK> compress_cpu(
    const poseidon2::Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D>& poseidon,
    const std::array<F, CHUNK>& left,
    const std::array<F, CHUNK>& right)
{
    F state[WIDTH];
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = F::zero_val();
    }
    for (size_t i = 0; i < CHUNK; ++i) {
        state[i]         = left[i];
        state[CHUNK + i] = right[i];
    }
    poseidon.permute(state);

    std::array<F, CHUNK> out;
    for (size_t i = 0; i < CHUNK; ++i) out[i] = state[i];
    return out;
}

} // namespace detail

// =============================================================================
// Host-Side Build Function
// =============================================================================

/**
 * @brief Build a Merkle tree using GPU acceleration (or CPU fallback).
 *
 * Algorithm:
 *   1. Sort matrices by height descending.
 *   2. Concatenate rows from all tallest matrices and hash on GPU.
 *   3. Level-by-level: compress pairs, then inject shorter matrices.
 *   4. Copy all digest layers back to CPU and construct MerkleTree.
 *
 * @tparam F            Field element type (e.g., BabyBear)
 * @tparam WIDTH        Poseidon2 state width (must satisfy 2*DIGEST_ELEMS <= WIDTH)
 * @tparam RATE         Sponge rate for leaf hashing
 * @tparam DIGEST_ELEMS Digest size in field elements
 * @tparam ROUNDS_F     Number of external (full) Poseidon2 rounds
 * @tparam ROUNDS_P     Number of internal (partial) Poseidon2 rounds
 * @tparam D            S-box exponent
 *
 * @param matrices      Leaf matrices (moved into the returned tree)
 * @param h_poseidon    Host-side Poseidon2Cuda struct (constants copied to GPU)
 * @param cap_height    Height of the Merkle cap (0 = single root hash)
 * @return              Constructed MerkleTree<F, F, DIGEST_ELEMS>
 */
template <typename F, size_t WIDTH, size_t RATE, size_t DIGEST_ELEMS,
          size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
MerkleTree<F, F, DIGEST_ELEMS> build_merkle_tree_cuda(
    std::vector<RowMajorMatrix<F>> matrices,
    const poseidon2::Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D>& h_poseidon,
    size_t cap_height)
{
    static_assert(2 * DIGEST_ELEMS <= WIDTH,
        "WIDTH must be >= 2*DIGEST_ELEMS to hold two children in the compression state");

    assert(!matrices.empty() && "need at least one matrix");

    // -------------------------------------------------------------------------
    // Step 1: Sort matrix indices by height descending (stable for equal heights)
    // -------------------------------------------------------------------------
    std::vector<size_t> sorted_indices(matrices.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::stable_sort(sorted_indices.begin(), sorted_indices.end(),
        [&](size_t a, size_t b) {
            return matrices[a].height() > matrices[b].height();
        });

    const size_t max_height = matrices[sorted_indices[0]].height();
    assert(max_height > 0 && "matrices must be non-empty");
    assert((max_height & (max_height - 1)) == 0 &&
           "matrix height must be a power of two");

    size_t next_sorted = 0;
    auto collect_at_height = [&](size_t h) -> std::vector<size_t> {
        std::vector<size_t> indices;
        while (next_sorted < sorted_indices.size() &&
               matrices[sorted_indices[next_sorted]].height() == h) {
            indices.push_back(sorted_indices[next_sorted++]);
        }
        return indices;
    };

    // -------------------------------------------------------------------------
    // Helper: build a flat row-major array by concatenating rows from matrices.
    // result[row * total_cols + col_offset + c] = mat[row][c]
    // -------------------------------------------------------------------------
    auto concat_rows = [&](size_t num_rows, const std::vector<size_t>& mat_indices)
        -> std::vector<F>
    {
        size_t total_cols = 0;
        for (size_t idx : mat_indices) {
            total_cols += matrices[idx].width();
        }

        std::vector<F> flat(num_rows * total_cols);
        for (size_t row = 0; row < num_rows; ++row) {
            size_t col_offset = 0;
            for (size_t idx : mat_indices) {
                const size_t w = matrices[idx].width();
                const F* src = matrices[idx].values.data() + row * w;
                F* dst = flat.data() + row * total_cols + col_offset;
                for (size_t c = 0; c < w; ++c) dst[c] = src[c];
                col_offset += w;
            }
        }
        return flat;
    };

    MerkleTree<F, F, DIGEST_ELEMS> tree;

    constexpr size_t BLOCK_SIZE = 256;

#if P3_CUDA_ENABLED
    // =========================================================================
    // GPU Path
    // =========================================================================
    using PoseidonType = poseidon2::Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D>;

    // Copy Poseidon2 constants to device
    PoseidonType* d_poseidon;
    P3_CUDA_CHECK(cudaMalloc(&d_poseidon, sizeof(PoseidonType)));
    P3_CUDA_CHECK(cudaMemcpy(d_poseidon, &h_poseidon, sizeof(PoseidonType),
                             cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Build leaf layer: hash rows of the tallest matrices
    // -------------------------------------------------------------------------
    std::vector<size_t> tallest_indices = collect_at_height(max_height);
    assert(!tallest_indices.empty());

    // Concatenate tallest matrices' rows into a flat CPU array
    size_t total_leaf_cols = 0;
    for (size_t idx : tallest_indices) total_leaf_cols += matrices[idx].width();
    std::vector<F> flat_leaves = concat_rows(max_height, tallest_indices);

    // Transfer leaf data to GPU
    F* d_leaves;
    P3_CUDA_CHECK(cudaMalloc(&d_leaves, flat_leaves.size() * sizeof(F)));
    P3_CUDA_CHECK(cudaMemcpy(d_leaves, flat_leaves.data(),
                             flat_leaves.size() * sizeof(F),
                             cudaMemcpyHostToDevice));
    flat_leaves.clear();
    flat_leaves.shrink_to_fit();

    // Allocate leaf digest buffer
    F* d_current_layer;
    P3_CUDA_CHECK(cudaMalloc(&d_current_layer,
                             max_height * DIGEST_ELEMS * sizeof(F)));

    // Launch leaf hashing kernel (one thread per row)
    {
        int blocks = static_cast<int>((max_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
        hash_leaves_kernel<F, WIDTH, RATE, DIGEST_ELEMS, ROUNDS_F, ROUNDS_P, D>
            <<<blocks, BLOCK_SIZE>>>(d_poseidon, d_leaves, max_height,
                                     total_leaf_cols, d_current_layer);
        P3_CUDA_CHECK(cudaDeviceSynchronize());
        P3_CUDA_CHECK(cudaGetLastError());
    }
    P3_CUDA_CHECK(cudaFree(d_leaves));

    // Copy leaf layer back to CPU
    {
        std::vector<std::array<F, DIGEST_ELEMS>> leaf_layer(max_height);
        P3_CUDA_CHECK(cudaMemcpy(leaf_layer.data(), d_current_layer,
                                 max_height * DIGEST_ELEMS * sizeof(F),
                                 cudaMemcpyDeviceToHost));
        tree.digest_layers.push_back(std::move(leaf_layer));
    }

    // -------------------------------------------------------------------------
    // Build internal layers level-by-level
    // -------------------------------------------------------------------------
    const size_t cap_size = (size_t(1) << cap_height);
    size_t level_size = max_height;

    while (level_size > cap_size) {
        level_size /= 2;

        // Allocate output layer buffer
        F* d_next_layer;
        P3_CUDA_CHECK(cudaMalloc(&d_next_layer,
                                 level_size * DIGEST_ELEMS * sizeof(F)));

        // Compress pairs (one thread per parent node)
        {
            int blocks = static_cast<int>((level_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
            compress_layer_kernel<F, WIDTH, DIGEST_ELEMS, ROUNDS_F, ROUNDS_P, D>
                <<<blocks, BLOCK_SIZE>>>(d_poseidon, d_current_layer,
                                        d_next_layer, level_size);
            P3_CUDA_CHECK(cudaDeviceSynchronize());
            P3_CUDA_CHECK(cudaGetLastError());
        }

        // Free previous layer, advance pointer
        P3_CUDA_CHECK(cudaFree(d_current_layer));
        d_current_layer = d_next_layer;

        // Inject shorter matrices at this level
        std::vector<size_t> inject_indices = collect_at_height(level_size);
        if (!inject_indices.empty()) {
            std::vector<F> flat_inject = concat_rows(level_size, inject_indices);
            const size_t inject_cols = flat_inject.size() / level_size;

            F* d_inject;
            P3_CUDA_CHECK(cudaMalloc(&d_inject, flat_inject.size() * sizeof(F)));
            P3_CUDA_CHECK(cudaMemcpy(d_inject, flat_inject.data(),
                                     flat_inject.size() * sizeof(F),
                                     cudaMemcpyHostToDevice));

            int blocks = static_cast<int>((level_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
            inject_and_compress_kernel<F, WIDTH, RATE, DIGEST_ELEMS, ROUNDS_F, ROUNDS_P, D>
                <<<blocks, BLOCK_SIZE>>>(d_poseidon, d_current_layer, d_inject,
                                        inject_cols, level_size);
            P3_CUDA_CHECK(cudaDeviceSynchronize());
            P3_CUDA_CHECK(cudaGetLastError());

            P3_CUDA_CHECK(cudaFree(d_inject));
        }

        // Copy this layer back to CPU
        {
            std::vector<std::array<F, DIGEST_ELEMS>> layer(level_size);
            P3_CUDA_CHECK(cudaMemcpy(layer.data(), d_current_layer,
                                     level_size * DIGEST_ELEMS * sizeof(F),
                                     cudaMemcpyDeviceToHost));
            tree.digest_layers.push_back(std::move(layer));
        }
    }

    P3_CUDA_CHECK(cudaFree(d_current_layer));
    P3_CUDA_CHECK(cudaFree(d_poseidon));

#else
    // =========================================================================
    // CPU Fallback: same algorithm using Poseidon2Cuda on CPU
    // (Poseidon2Cuda::permute is P3_HOST_DEVICE, so it runs on CPU too)
    // =========================================================================

    // Build leaf layer
    {
        std::vector<size_t> tallest_indices = collect_at_height(max_height);
        assert(!tallest_indices.empty());

        size_t total_leaf_cols = 0;
        for (size_t idx : tallest_indices) total_leaf_cols += matrices[idx].width();
        std::vector<F> flat_leaves = concat_rows(max_height, tallest_indices);

        std::vector<std::array<F, DIGEST_ELEMS>> leaf_layer(max_height);
        for (size_t row = 0; row < max_height; ++row) {
            leaf_layer[row] = detail::hash_row_cpu<F, WIDTH, RATE, DIGEST_ELEMS,
                                                   ROUNDS_F, ROUNDS_P, D>(
                h_poseidon,
                flat_leaves.data() + row * total_leaf_cols,
                total_leaf_cols);
        }
        tree.digest_layers.push_back(std::move(leaf_layer));
    }

    const size_t cap_size = (size_t(1) << cap_height);
    size_t level_size = max_height;

    while (level_size > cap_size) {
        level_size /= 2;

        const auto& prev_layer = tree.digest_layers.back();
        std::vector<std::array<F, DIGEST_ELEMS>> next_layer(level_size);

        // Compress pairs
        for (size_t i = 0; i < level_size; ++i) {
            next_layer[i] = detail::compress_cpu<F, WIDTH, DIGEST_ELEMS,
                                                  ROUNDS_F, ROUNDS_P, D>(
                h_poseidon,
                prev_layer[2 * i],
                prev_layer[2 * i + 1]);
        }

        // Inject shorter matrices
        std::vector<size_t> inject_indices = collect_at_height(level_size);
        if (!inject_indices.empty()) {
            size_t inject_total_cols = 0;
            for (size_t idx : inject_indices) inject_total_cols += matrices[idx].width();
            std::vector<F> flat_inject = concat_rows(level_size, inject_indices);

            for (size_t i = 0; i < level_size; ++i) {
                // Hash the injected row
                std::array<F, DIGEST_ELEMS> inject_hash =
                    detail::hash_row_cpu<F, WIDTH, RATE, DIGEST_ELEMS,
                                        ROUNDS_F, ROUNDS_P, D>(
                        h_poseidon,
                        flat_inject.data() + i * inject_total_cols,
                        inject_total_cols);

                // Compress (node_digest, inject_hash)
                next_layer[i] = detail::compress_cpu<F, WIDTH, DIGEST_ELEMS,
                                                      ROUNDS_F, ROUNDS_P, D>(
                    h_poseidon, next_layer[i], inject_hash);
            }
        }

        tree.digest_layers.push_back(std::move(next_layer));
    }

#endif // P3_CUDA_ENABLED

    // Store leaf matrices in the tree
    tree.leaves = std::move(matrices);

    return tree;
}

} // namespace p3_merkle_tree

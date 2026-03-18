#pragma once

/**
 * @file poseidon2_cuda.hpp
 * @brief CUDA-optimized Poseidon2 permutation
 *
 * This header provides a GPU-compatible implementation of Poseidon2 that
 * doesn't use virtual methods or dynamic memory allocation.
 *
 * The implementation uses template parameters for compile-time optimization
 * and raw pointers for GPU memory access.
 */

#include "cuda_compat.hpp"
#include "external.hpp"
#include "internal.hpp"
#include "generic.hpp"
#include "round_numbers.hpp"

#include <cstdint>
#include <cstddef>

namespace poseidon2 {

/**
 * @brief CUDA-compatible Poseidon2 permutation structure
 *
 * This is a POD (Plain Old Data) structure that can be passed to GPU kernels.
 * All constants are stored as raw arrays.
 *
 * @tparam F The field type
 * @tparam WIDTH The state width
 * @tparam ROUNDS_F Number of full (external) rounds
 * @tparam ROUNDS_P Number of partial (internal) rounds
 * @tparam D The S-box exponent
 */
template<typename F, size_t WIDTH, size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
struct Poseidon2Cuda {
    // External round constants (ROUNDS_F/2 initial + ROUNDS_F/2 terminal)
    // Each round has WIDTH constants
    F initial_external_constants[ROUNDS_F / 2][WIDTH];
    F terminal_external_constants[ROUNDS_F / 2][WIDTH];

    // Internal round constants (one per round)
    F internal_constants[ROUNDS_P];

    /**
     * @brief Apply the full Poseidon2 permutation to a state
     *
     * The permutation consists of:
     * 1. Initial external rounds (ROUNDS_F/2 rounds)
     * 2. Internal rounds (ROUNDS_P rounds)
     * 3. Terminal external rounds (ROUNDS_F/2 rounds)
     *
     * @param state Pointer to the state array of WIDTH elements
     */
    P3_HOST_DEVICE void permute(F* state) const {
        // Initial external rounds
        permute_external_initial(state);

        // Internal rounds
        permute_internal(state);

        // Terminal external rounds
        permute_external_terminal(state);
    }

    /**
     * @brief Apply initial external rounds
     */
    P3_HOST_DEVICE void permute_external_initial(F* state) const {
        // Apply initial MDS layer
        mds_light_permutation_cuda<F, WIDTH>(state);

        // Apply half of external rounds
        for (size_t round = 0; round < ROUNDS_F / 2; ++round) {
            // Add round constants and apply S-box to all elements
            for (size_t i = 0; i < WIDTH; ++i) {
                state[i] += initial_external_constants[round][i];
                state[i] = state[i].template injective_exp_n<D>();
            }

            // Apply MDS
            mds_light_permutation_cuda<F, WIDTH>(state);
        }
    }

    /**
     * @brief Apply internal rounds
     */
    P3_HOST_DEVICE void permute_internal(F* state) const {
        for (size_t round = 0; round < ROUNDS_P; ++round) {
            // Add round constant and apply S-box only to first element
            state[0] += internal_constants[round];
            state[0] = state[0].template injective_exp_n<D>();

            // Apply internal linear layer
            GenericPoseidon2LinearLayers<WIDTH>::template internal_linear_layer_cuda<F>(state);
        }
    }

    /**
     * @brief Apply terminal external rounds
     */
    P3_HOST_DEVICE void permute_external_terminal(F* state) const {
        for (size_t round = 0; round < ROUNDS_F / 2; ++round) {
            // Add round constants and apply S-box to all elements
            for (size_t i = 0; i < WIDTH; ++i) {
                state[i] += terminal_external_constants[round][i];
                state[i] = state[i].template injective_exp_n<D>();
            }

            // Apply MDS
            mds_light_permutation_cuda<F, WIDTH>(state);
        }
    }
};

/**
 * @brief Helper function to initialize Poseidon2Cuda constants on the host
 *
 * @param poseidon The Poseidon2Cuda structure to initialize
 * @param initial_constants Initial external round constants
 * @param terminal_constants Terminal external round constants
 * @param internal_constants Internal round constants
 */
template<typename F, size_t WIDTH, size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
void init_poseidon2_cuda(
    Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D>& poseidon,
    const F initial_constants[ROUNDS_F / 2][WIDTH],
    const F terminal_constants[ROUNDS_F / 2][WIDTH],
    const F internal_constants[ROUNDS_P]
) {
    for (size_t r = 0; r < ROUNDS_F / 2; ++r) {
        for (size_t i = 0; i < WIDTH; ++i) {
            poseidon.initial_external_constants[r][i] = initial_constants[r][i];
            poseidon.terminal_external_constants[r][i] = terminal_constants[r][i];
        }
    }
    for (size_t r = 0; r < ROUNDS_P; ++r) {
        poseidon.internal_constants[r] = internal_constants[r];
    }
}

/**
 * @brief CUDA kernel for batch Poseidon2 permutation
 *
 * Applies Poseidon2 to multiple states in parallel.
 *
 * @param poseidon Pointer to Poseidon2Cuda structure in device memory
 * @param states Array of states (n_states * WIDTH elements)
 * @param n_states Number of states to process
 */
#if P3_CUDA_ENABLED
template<typename F, size_t WIDTH, size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
__global__ void poseidon2_permute_kernel(
    const Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D>* poseidon,
    F* states,
    size_t n_states
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_states) {
        F* state = states + idx * WIDTH;
        poseidon->permute(state);
    }
}
#endif

/**
 * @brief Specialized Poseidon2 for Goldilocks with WIDTH=8, D=7
 */
template<typename Goldilocks>
using Poseidon2CudaGoldilocks8 = Poseidon2Cuda<Goldilocks, 8, 8, 22, 7>;

/**
 * @brief Specialized Poseidon2 for Goldilocks with WIDTH=12, D=7
 */
template<typename Goldilocks>
using Poseidon2CudaGoldilocks12 = Poseidon2Cuda<Goldilocks, 12, 8, 22, 7>;

/**
 * @brief Specialized Poseidon2 for Goldilocks with WIDTH=16, D=7
 */
template<typename Goldilocks>
using Poseidon2CudaGoldilocks16 = Poseidon2Cuda<Goldilocks, 16, 8, 22, 7>;

/**
 * @brief Specialized Poseidon2 for BabyBear with WIDTH=16, D=5
 */
template<typename BabyBear>
using Poseidon2CudaBabyBear16 = Poseidon2Cuda<BabyBear, 16, 8, 14, 5>;

/**
 * @brief Specialized Poseidon2 for BabyBear with WIDTH=24, D=5
 */
template<typename BabyBear>
using Poseidon2CudaBabyBear24 = Poseidon2Cuda<BabyBear, 24, 8, 22, 5>;

/**
 * @brief Specialized Poseidon2 for Mersenne31 with WIDTH=16, D=5
 */
template<typename Mersenne31>
using Poseidon2CudaMersenne31_16 = Poseidon2Cuda<Mersenne31, 16, 8, 14, 5>;

/**
 * @brief Specialized Poseidon2 for Mersenne31 with WIDTH=24, D=5
 */
template<typename Mersenne31>
using Poseidon2CudaMersenne31_24 = Poseidon2Cuda<Mersenne31, 24, 8, 22, 5>;

/**
 * @brief Simple hash function using Poseidon2 (sponge construction)
 *
 * This is a basic sponge-based hash implementation for CUDA.
 *
 * @tparam F Field type
 * @tparam WIDTH State width (rate + capacity)
 * @tparam RATE Number of elements absorbed per permutation
 * @tparam ROUNDS_F Full rounds
 * @tparam ROUNDS_P Partial rounds
 * @tparam D S-box exponent
 */
template<typename F, size_t WIDTH, size_t RATE, size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
struct Poseidon2SpongeCuda {
    Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D> poseidon;

    /**
     * @brief Hash an array of field elements
     *
     * @param input Input elements to hash
     * @param input_len Number of input elements
     * @param output Output buffer (at least RATE elements)
     */
    P3_HOST_DEVICE void hash(const F* input, size_t input_len, F* output) const {
        // Initialize state to zero
        F state[WIDTH];
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] = F::zero_val();
        }

        // Absorb phase
        size_t absorbed = 0;
        while (absorbed < input_len) {
            // Absorb up to RATE elements
            size_t to_absorb = input_len - absorbed;
            if (to_absorb > RATE) to_absorb = RATE;

            for (size_t i = 0; i < to_absorb; ++i) {
                state[i] += input[absorbed + i];
            }
            absorbed += to_absorb;

            // Apply permutation
            poseidon.permute(state);
        }

        // Squeeze phase (output RATE elements)
        for (size_t i = 0; i < RATE; ++i) {
            output[i] = state[i];
        }
    }

    /**
     * @brief Hash two field elements (common case for Merkle trees)
     */
    P3_HOST_DEVICE void hash2(const F& a, const F& b, F* output) const {
        F state[WIDTH];
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] = F::zero_val();
        }
        state[0] = a;
        state[1] = b;

        poseidon.permute(state);

        // Output first element
        output[0] = state[0];
    }
};

/**
 * @brief CUDA kernel for batch hashing with Poseidon2 sponge
 *
 * @param sponge Pointer to sponge structure in device memory
 * @param inputs Array of inputs (n_hashes * input_len elements)
 * @param outputs Array of outputs (n_hashes * RATE elements)
 * @param input_len Number of elements per input
 * @param n_hashes Number of hashes to compute
 */
#if P3_CUDA_ENABLED
template<typename F, size_t WIDTH, size_t RATE, size_t ROUNDS_F, size_t ROUNDS_P, uint64_t D>
__global__ void poseidon2_hash_kernel(
    const Poseidon2SpongeCuda<F, WIDTH, RATE, ROUNDS_F, ROUNDS_P, D>* sponge,
    const F* inputs,
    F* outputs,
    size_t input_len,
    size_t n_hashes
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_hashes) {
        const F* input = inputs + idx * input_len;
        F* output = outputs + idx * RATE;
        sponge->hash(input, input_len, output);
    }
}
#endif

} // namespace poseidon2



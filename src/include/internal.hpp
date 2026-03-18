#pragma once

#include "cuda_compat.hpp"
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>

namespace poseidon2 {

/**
 * @brief Matrix-vector multiplication for internal layers
 *
 * Computes (1 + diag(mat_internal_diag_m_1)) * state
 * where 1 is the constant matrix of ones.
 */
template<typename F, typename A, size_t WIDTH>
void matmul_internal(
    std::array<A, WIDTH>& state,
    const std::array<F, WIDTH>& mat_internal_diag_m_1
) {
    // Compute sum of all state elements
    A sum = A::zero();
    for (size_t i = 0; i < WIDTH; ++i) {
        sum += state[i];
    }

    // Apply transformation: state[i] = state[i] * mat_internal_diag_m_1[i] + sum
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] *= mat_internal_diag_m_1[i];
        state[i] += sum;
    }
}

/**
 * @brief CUDA-compatible matrix-vector multiplication for internal layers
 *
 * Uses raw pointers instead of std::array for GPU compatibility.
 */
template<typename F, typename A, size_t WIDTH>
P3_HOST_DEVICE void matmul_internal_cuda(
    A* state,
    const F* mat_internal_diag_m_1
) {
    // Compute sum of all state elements
    A sum = A::zero_val();
    for (size_t i = 0; i < WIDTH; ++i) {
        sum += state[i];
    }

    // Apply transformation: state[i] = state[i] * mat_internal_diag_m_1[i] + sum
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] *= mat_internal_diag_m_1[i];
        state[i] += sum;
    }
}

/**
 * @brief Trait for constructing internal layers from constants
 */
template<typename F>
class InternalLayerConstructor {
public:
    virtual ~InternalLayerConstructor() = default;
};

/**
 * @brief Trait for internal layer operations
 */
template<typename R, size_t WIDTH, uint64_t D>
class InternalLayer {
public:
    virtual ~InternalLayer() = default;
    virtual void permute_state(std::array<R, WIDTH>& state) = 0;
};

/**
 * @brief Generic internal permutation state implementation
 *
 * This is a helper method for any field to implement internal layer.
 * Should only be used where performance is not critical.
 */
template<typename F, typename A, size_t WIDTH, uint64_t D>
void internal_permute_state(
    std::array<A, WIDTH>& state,
    void (*diffusion_mat)(std::array<A, WIDTH>&),
    const std::vector<F>& internal_constants
) {
    for (const auto& constant : internal_constants) {
        // Apply S-box only to first element with round constant
        state[0] += constant;
        state[0] = state[0].template injective_exp_n<D>();

        // Apply diffusion matrix
        diffusion_mat(state);
    }
}

} // namespace poseidon2

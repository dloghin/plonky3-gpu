#pragma once

#include "cuda_compat.hpp"
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>

#if !P3_CUDA_ENABLED
#include <stdexcept>
#endif

namespace poseidon2 {

/**
 * @brief Multiply a 4-element vector by the Horizon Labs matrix:
 * [ 5 7 1 3 ]
 * [ 4 6 1 1 ]
 * [ 1 3 5 7 ]
 * [ 1 1 4 6 ]
 *
 * This uses the formula from Appendix B in the Poseidon2 paper.
 */
template<typename R>
P3_HOST_DEVICE void apply_hl_mat4(std::array<R, 4>& x) {
    R t0 = x[0] + x[1];
    R t1 = x[2] + x[3];
    R t2 = x[1] + x[1] + t1;
    R t3 = x[3] + x[3] + t0;
    R t4 = t1.double_val().double_val() + t3;
    R t5 = t0.double_val().double_val() + t2;
    R t6 = t3 + t5;
    R t7 = t2 + t4;
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

/**
 * @brief Multiply a 4-element vector by the optimized matrix:
 * [ 2 3 1 1 ]
 * [ 1 2 3 1 ]
 * [ 1 1 2 3 ]
 * [ 3 1 1 2 ]
 *
 * This is more efficient than the HL matrix (7 additions and 2 doubles).
 */
template<typename R>
P3_HOST_DEVICE void apply_mat4(std::array<R, 4>& x) {
    R t01 = x[0] + x[1];
    R t23 = x[2] + x[3];
    R t0123 = t01 + t23;
    R t01123 = t0123 + x[1];
    R t01233 = t0123 + x[3];
    // Order is important: overwrite x[0] and x[2] after x[1] and x[3]
    x[3] = t01233 + x[0].double_val(); // 3*x[0] + x[1] + x[2] + 2*x[3]
    x[1] = t01123 + x[2].double_val(); // x[0] + 2*x[1] + 3*x[2] + x[3]
    x[0] = t01123 + t01;                // 2*x[0] + 3*x[1] + x[2] + x[3]
    x[2] = t01233 + t23;                // x[0] + x[1] + 2*x[2] + 3*x[3]
}

/**
 * @brief Raw pointer version of apply_mat4 for CUDA compatibility
 */
template<typename R>
P3_HOST_DEVICE void apply_mat4_ptr(R* x) {
    R t01 = x[0] + x[1];
    R t23 = x[2] + x[3];
    R t0123 = t01 + t23;
    R t01123 = t0123 + x[1];
    R t01233 = t0123 + x[3];
    x[3] = t01233 + x[0].double_val();
    x[1] = t01123 + x[2].double_val();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

/**
 * @brief Base class for MDS matrix permutations (CPU only)
 */
template<typename R, size_t WIDTH>
class MDSPermutation {
public:
    virtual ~MDSPermutation() = default;
    virtual void permute_mut(std::array<R, WIDTH>& input) = 0;
};

/**
 * @brief Horizon Labs 4x4 MDS matrix
 * Requires 10 additions and 4 doubles
 */
template<typename R>
class HLMDSMat4 : public MDSPermutation<R, 4> {
public:
    void permute_mut(std::array<R, 4>& input) override {
        apply_hl_mat4(input);
    }
};

/**
 * @brief Optimized 4x4 MDS matrix
 * Requires 7 additions and 2 doubles (faster)
 */
template<typename R>
class MDSMat4 : public MDSPermutation<R, 4> {
public:
    void permute_mut(std::array<R, 4>& input) override {
        apply_mat4(input);
    }
};

/**
 * @brief Apply the MDS light permutation for external layers
 *
 * Given a 4x4 MDS matrix M, we multiply by the 4N x 4N matrix
 * [[2M M  ... M], [M  2M ... M], ..., [M  M ... 2M]]
 *
 * Supported widths: 2, 3, 4, 8, 12, 16, 20, 24
 */
template<typename R, size_t WIDTH>
void mds_light_permutation(std::array<R, WIDTH>& state, MDSPermutation<R, 4>* mdsmat) {
    if constexpr (WIDTH == 2) {
        R sum = state[0] + state[1];
        state[0] += sum;
        state[1] += sum;
    } else if constexpr (WIDTH == 3) {
        R sum = state[0] + state[1] + state[2];
        state[0] += sum;
        state[1] += sum;
        state[2] += sum;
    } else if constexpr (WIDTH == 4 || WIDTH == 8 || WIDTH == 12 ||
                         WIDTH == 16 || WIDTH == 20 || WIDTH == 24) {
        // Apply M_4 to each consecutive four elements
        for (size_t i = 0; i < WIDTH; i += 4) {
            std::array<R, 4> chunk = {state[i], state[i+1], state[i+2], state[i+3]};
            mdsmat->permute_mut(chunk);
            state[i] = chunk[0];
            state[i+1] = chunk[1];
            state[i+2] = chunk[2];
            state[i+3] = chunk[3];
        }

        // Compute sums of every four elements
        std::array<R, 4> sums;
        for (size_t k = 0; k < 4; ++k) {
            sums[k] = R::zero();
            for (size_t j = k; j < WIDTH; j += 4) {
                sums[k] += state[j];
            }
        }

        // Add the appropriate sum to each element
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] += sums[i % 4];
        }
#if !P3_CUDA_ENABLED
    } else {
        throw std::runtime_error("Unsupported width");
#endif
    }
}

/**
 * @brief CUDA-compatible MDS light permutation (no virtual methods)
 *
 * Uses apply_mat4 directly instead of virtual dispatch.
 */
template<typename R, size_t WIDTH>
P3_HOST_DEVICE void mds_light_permutation_cuda(R* state) {
    if constexpr (WIDTH == 2) {
        R sum = state[0] + state[1];
        state[0] += sum;
        state[1] += sum;
    } else if constexpr (WIDTH == 3) {
        R sum = state[0] + state[1] + state[2];
        state[0] += sum;
        state[1] += sum;
        state[2] += sum;
    } else if constexpr (WIDTH == 4 || WIDTH == 8 || WIDTH == 12 ||
                         WIDTH == 16 || WIDTH == 20 || WIDTH == 24) {
        // Apply M_4 to each consecutive four elements
        for (size_t i = 0; i < WIDTH; i += 4) {
            apply_mat4_ptr<R>(&state[i]);
        }

        // Compute sums of every four elements
        R sums[4];
        for (size_t k = 0; k < 4; ++k) {
            sums[k] = R::zero_val();
            for (size_t j = k; j < WIDTH; j += 4) {
                sums[k] += state[j];
            }
        }

        // Add the appropriate sum to each element
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] += sums[i % 4];
        }
    }
}

/**
 * @brief Container for external layer constants
 */
template<typename T, size_t WIDTH>
class ExternalLayerConstants {
private:
    std::vector<std::array<T, WIDTH>> initial_;
    std::vector<std::array<T, WIDTH>> terminal_;

public:
    ExternalLayerConstants(
        std::vector<std::array<T, WIDTH>> initial,
        std::vector<std::array<T, WIDTH>> terminal
    ) : initial_(std::move(initial)), terminal_(std::move(terminal)) {
#if !P3_CUDA_ENABLED
        if (initial_.size() != terminal_.size()) {
            throw std::runtime_error(
                "The number of initial and terminal external rounds should be equal."
            );
        }
#endif
    }

    const std::vector<std::array<T, WIDTH>>& get_initial_constants() const {
        return initial_;
    }

    const std::vector<std::array<T, WIDTH>>& get_terminal_constants() const {
        return terminal_;
    }
};

/**
 * @brief Apply terminal external rounds
 *
 * Each external round consists of:
 * 1. Adding round constants
 * 2. Applying S-box
 * 3. Applying external linear layer
 */
template<typename R, typename CT, size_t WIDTH>
void external_terminal_permute_state(
    std::array<R, WIDTH>& state,
    const std::vector<std::array<CT, WIDTH>>& terminal_external_constants,
    void (*add_rc_and_sbox)(R&, const CT&),
    MDSPermutation<R, 4>* mat4
) {
    for (const auto& round_constants : terminal_external_constants) {
        for (size_t i = 0; i < WIDTH; ++i) {
            add_rc_and_sbox(state[i], round_constants[i]);
        }
        mds_light_permutation<R, WIDTH>(state, mat4);
    }
}

/**
 * @brief Apply initial external rounds
 *
 * Starts with an external linear layer, then applies standard external rounds.
 */
template<typename R, typename CT, size_t WIDTH>
void external_initial_permute_state(
    std::array<R, WIDTH>& state,
    const std::vector<std::array<CT, WIDTH>>& initial_external_constants,
    void (*add_rc_and_sbox)(R&, const CT&),
    MDSPermutation<R, 4>* mat4
) {
    mds_light_permutation<R, WIDTH>(state, mat4);
    // After initial MDS, remaining layers are identical to terminal
    external_terminal_permute_state<R, CT, WIDTH>(
        state, initial_external_constants, add_rc_and_sbox, mat4
    );
}

/**
 * @brief Trait for constructing external layers from constants
 */
template<typename F, size_t WIDTH>
class ExternalLayerConstructor {
public:
    virtual ~ExternalLayerConstructor() = default;
};

/**
 * @brief Trait for external layer operations
 */
template<typename R, size_t WIDTH, uint64_t D>
class ExternalLayer {
public:
    virtual ~ExternalLayer() = default;
    virtual void permute_state_initial(std::array<R, WIDTH>& state) = 0;
    virtual void permute_state_terminal(std::array<R, WIDTH>& state) = 0;
};

} // namespace poseidon2

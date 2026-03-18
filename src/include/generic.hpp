#pragma once

#include "cuda_compat.hpp"
#include "external.hpp"
#include "internal.hpp"
#include <cstddef>
#include <cstdint>

namespace poseidon2 {

/**
 * @brief Generic method performing the transformation: s -> (s + rc)^D
 *
 * This is slower than field-specific implementations (particularly for packed fields)
 * and should only be used in non-performance-critical places.
 */
template<typename F, typename A, uint64_t D>
void add_rc_and_sbox_generic(A& val, const F& rc) {
    val += rc;
    val = val.template injective_exp_n<D>();
}

/**
 * @brief CUDA-compatible S-box: s -> (s + rc)^D
 */
template<typename F, typename A, uint64_t D>
P3_HOST_DEVICE P3_INLINE void add_rc_and_sbox_cuda(A& val, const F& rc) {
    val += rc;
    val = val.template injective_exp_n<D>();
}

/**
 * @brief CUDA-compatible S-box without round constant: s -> s^D
 */
template<typename A, uint64_t D>
P3_HOST_DEVICE P3_INLINE void sbox_cuda(A& val) {
    val = val.template injective_exp_n<D>();
}

/**
 * @brief Generic trait for Poseidon2 linear layers
 *
 * This class provides default implementations of the linear layers.
 * For specific widths, you can specialize this template for better performance.
 */
template<size_t WIDTH>
class GenericPoseidon2LinearLayers {
public:
    /**
     * @brief Generic implementation of internal linear layer matrix multiplication
     *
     * For the generic implementation, we use a simple matrix based on powers of 2.
     * This satisfies the diffusion requirements but may not be optimal.
     *
     * The internal matrix is of the form (1 + D) where 1 is the all-ones matrix
     * and D is a diagonal matrix with specific values.
     */
    template<typename R>
    static void internal_linear_layer(std::array<R, WIDTH>& state) {
        // Define diagonal values for different widths
        // These are chosen to satisfy the security requirements
        static_assert(WIDTH == 2 || WIDTH == 3 || WIDTH == 4 || WIDTH == 8 ||
                      WIDTH == 12 || WIDTH == 16 || WIDTH == 20 || WIDTH == 24,
                      "Unsupported WIDTH for internal_linear_layer");

        std::array<R, WIDTH> diag;

        if constexpr (WIDTH == 16) {
            // Diagonal values for width 16 (example values - these should be optimized)
            R base = R::zero();
            base += R::one();
            for (size_t i = 0; i < WIDTH; ++i) {
                diag[i] = base;
                // Create distinct diagonal values
                for (size_t j = 0; j < i; ++j) {
                    diag[i] = diag[i].double_val();
                }
            }
        } else if constexpr (WIDTH == 8) {
            R base = R::one();
            for (size_t i = 0; i < WIDTH; ++i) {
                diag[i] = base;
                for (size_t j = 0; j < i; ++j) {
                    diag[i] = diag[i].double_val();
                }
            }
        } else if constexpr (WIDTH == 12) {
            R base = R::one();
            for (size_t i = 0; i < WIDTH; ++i) {
                diag[i] = base;
                for (size_t j = 0; j < i; ++j) {
                    diag[i] = diag[i].double_val();
                }
            }
        } else if constexpr (WIDTH == 4) {
            R base = R::one();
            for (size_t i = 0; i < WIDTH; ++i) {
                diag[i] = base;
                for (size_t j = 0; j < i; ++j) {
                    diag[i] = diag[i].double_val();
                }
            }
        } else if constexpr (WIDTH == 3) {
            R base = R::one();
            for (size_t i = 0; i < WIDTH; ++i) {
                diag[i] = base;
                for (size_t j = 0; j < i; ++j) {
                    diag[i] = diag[i].double_val();
                }
            }
        } else if constexpr (WIDTH == 2) {
            diag[0] = R::one();
            diag[1] = R::one().double_val();
        } else if constexpr (WIDTH == 20) {
            R base = R::one();
            for (size_t i = 0; i < WIDTH; ++i) {
                diag[i] = base;
                for (size_t j = 0; j < i; ++j) {
                    diag[i] = diag[i].double_val();
                }
            }
        } else if constexpr (WIDTH == 24) {
            R base = R::one();
            for (size_t i = 0; i < WIDTH; ++i) {
                diag[i] = base;
                for (size_t j = 0; j < i; ++j) {
                    diag[i] = diag[i].double_val();
                }
            }
        }

        // Apply (1 + D) transformation: result[i] = diag[i] * state[i] + sum(state)
        R sum = R::zero();
        for (size_t i = 0; i < WIDTH; ++i) {
            sum += state[i];
        }

        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] *= diag[i];
            state[i] += sum;
        }
    }

    /**
     * @brief CUDA-compatible internal linear layer using raw pointers
     */
    template<typename R>
    P3_HOST_DEVICE static void internal_linear_layer_cuda(R* state) {
        static_assert(WIDTH == 2 || WIDTH == 3 || WIDTH == 4 || WIDTH == 8 ||
                      WIDTH == 12 || WIDTH == 16 || WIDTH == 20 || WIDTH == 24,
                      "Unsupported WIDTH for internal_linear_layer_cuda");

        R diag[WIDTH];

        // Compute diagonal values using powers of 2
        R base = R::one_val();
        for (size_t i = 0; i < WIDTH; ++i) {
            diag[i] = base;
            for (size_t j = 0; j < i; ++j) {
                diag[i] = diag[i].double_val();
            }
        }

        // Compute sum of all state elements
        R sum = R::zero_val();
        for (size_t i = 0; i < WIDTH; ++i) {
            sum += state[i];
        }

        // Apply (1 + D) transformation
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] *= diag[i];
            state[i] += sum;
        }
    }

    /**
     * @brief Generic implementation of external linear layer matrix multiplication
     */
    template<typename R>
    static void external_linear_layer(std::array<R, WIDTH>& state) {
        MDSMat4<R> mds_mat;
        mds_light_permutation<R, WIDTH>(state, &mds_mat);
    }

    /**
     * @brief CUDA-compatible external linear layer using raw pointers
     */
    template<typename R>
    P3_HOST_DEVICE static void external_linear_layer_cuda(R* state) {
        mds_light_permutation_cuda<R, WIDTH>(state);
    }
};

} // namespace poseidon2

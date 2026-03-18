#pragma once

/**
 * @file poseidon2.hpp
 * @brief The Poseidon2 cryptographic permutation
 *
 * This implementation is based upon the following resources:
 * - https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2.rs
 * - https://eprint.iacr.org/2023/323.pdf
 */

#include "external.hpp"
#include "internal.hpp"
#include "generic.hpp"
#include "round_numbers.hpp"

#include <array>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

namespace poseidon2 {

/**
 * @brief Supported state widths for Poseidon2
 */
constexpr size_t SUPPORTED_WIDTHS[] = {2, 3, 4, 8, 12, 16, 20, 24};

/**
 * @brief Check if a width is supported
 */
inline bool is_width_supported(size_t width) {
    return std::find(std::begin(SUPPORTED_WIDTHS), std::end(SUPPORTED_WIDTHS), width)
           != std::end(SUPPORTED_WIDTHS);
}

/**
 * @brief Base trait for permutation operations
 */
template<typename T>
class Permutation {
public:
    virtual ~Permutation() = default;
    virtual void permute_mut(T& state) = 0;

    T permute(const T& state) const {
        T result = state;
        const_cast<Permutation*>(this)->permute_mut(result);
        return result;
    }
};

/**
 * @brief Marker trait for cryptographic permutations
 *
 * Indicates that a permutation is suitable for cryptographic use.
 */
template<typename T>
class CryptographicPermutation : public Permutation<T> {
public:
    virtual ~CryptographicPermutation() = default;
};

/**
 * @brief The Poseidon2 permutation
 *
 * @tparam F The base field type
 * @tparam A The algebra type (typically F or a packed field)
 * @tparam WIDTH The state width (must be in SUPPORTED_WIDTHS)
 * @tparam D The S-box exponent
 */
template<typename F, typename A, size_t WIDTH, uint64_t D>
class Poseidon2 : public CryptographicPermutation<std::array<A, WIDTH>> {
private:
    /// External layer implementation
    std::shared_ptr<ExternalLayer<A, WIDTH, D>> external_layer_;

    /// Internal layer implementation
    std::shared_ptr<InternalLayer<A, WIDTH, D>> internal_layer_;

public:
    /**
     * @brief Construct a new Poseidon2 instance with given constants
     *
     * @param external_constants Constants for external rounds
     * @param internal_constants Constants for internal rounds
     */
    Poseidon2(
        std::shared_ptr<ExternalLayer<A, WIDTH, D>> external_layer,
        std::shared_ptr<InternalLayer<A, WIDTH, D>> internal_layer
    ) : external_layer_(std::move(external_layer)),
        internal_layer_(std::move(internal_layer)) {

        if (!is_width_supported(WIDTH)) {
            throw std::runtime_error("Unsupported width for Poseidon2");
        }
    }

    /**
     * @brief Apply the Poseidon2 permutation to the given state
     *
     * The permutation consists of:
     * 1. Initial external rounds
     * 2. Internal rounds
     * 3. Terminal external rounds
     */
    void permute_mut(std::array<A, WIDTH>& state) override {
        external_layer_->permute_state_initial(state);
        internal_layer_->permute_state(state);
        external_layer_->permute_state_terminal(state);
    }

    /**
     * @brief Get a reference to the external layer
     */
    const ExternalLayer<A, WIDTH, D>& get_external_layer() const {
        return *external_layer_;
    }

    /**
     * @brief Get a reference to the internal layer
     */
    const InternalLayer<A, WIDTH, D>& get_internal_layer() const {
        return *internal_layer_;
    }
};

/**
 * @brief Generic Poseidon2 implementation
 *
 * This is a concrete implementation that can be used with any field type
 * that satisfies the required traits. For performance-critical applications,
 * consider using field-specific optimized implementations.
 */
template<typename F, typename A, size_t WIDTH, uint64_t D>
class GenericPoseidon2External : public ExternalLayer<A, WIDTH, D> {
private:
    ExternalLayerConstants<F, WIDTH> constants_;
    MDSMat4<A> mds_mat_;

public:
    explicit GenericPoseidon2External(ExternalLayerConstants<F, WIDTH> constants)
        : constants_(std::move(constants)) {}

    void permute_state_initial(std::array<A, WIDTH>& state) override {
        external_initial_permute_state<A, F, WIDTH>(
            state,
            constants_.get_initial_constants(),
            [](A& val, const F& rc) { add_rc_and_sbox_generic<F, A, D>(val, rc); },
            &mds_mat_
        );
    }

    void permute_state_terminal(std::array<A, WIDTH>& state) override {
        external_terminal_permute_state<A, F, WIDTH>(
            state,
            constants_.get_terminal_constants(),
            [](A& val, const F& rc) { add_rc_and_sbox_generic<F, A, D>(val, rc); },
            &mds_mat_
        );
    }
};

/**
 * @brief Generic internal layer implementation
 */
template<typename F, typename A, size_t WIDTH, uint64_t D>
class GenericPoseidon2Internal : public InternalLayer<A, WIDTH, D> {
private:
    std::vector<F> constants_;

public:
    explicit GenericPoseidon2Internal(std::vector<F> constants)
        : constants_(std::move(constants)) {}

    void permute_state(std::array<A, WIDTH>& state) override {
        for (const auto& constant : constants_) {
            add_rc_and_sbox_generic<F, A, D>(state[0], constant);
            GenericPoseidon2LinearLayers<WIDTH>::template internal_linear_layer<A>(state);
        }
    }
};

/**
 * @brief Factory function to create a Poseidon2 instance with specific round numbers
 *
 * @param rounds_f Number of full (external) rounds
 * @param rounds_p Number of partial (internal) rounds
 * @param external_constants External round constants
 * @param internal_constants Internal round constants
 */
template<typename F, typename A, size_t WIDTH, uint64_t D>
std::shared_ptr<Poseidon2<F, A, WIDTH, D>> create_poseidon2(
    const ExternalLayerConstants<F, WIDTH>& external_constants,
    const std::vector<F>& internal_constants
) {
    auto external_layer = std::make_shared<GenericPoseidon2External<F, A, WIDTH, D>>(
        external_constants
    );
    auto internal_layer = std::make_shared<GenericPoseidon2Internal<F, A, WIDTH, D>>(
        internal_constants
    );

    return std::make_shared<Poseidon2<F, A, WIDTH, D>>(
        external_layer,
        internal_layer
    );
}

/**
 * @brief Factory function to create a Poseidon2 instance with 128-bit security
 *
 * Automatically determines the appropriate number of rounds based on the field
 * characteristics and the state width.
 *
 * @param field_order The order of the field (prime modulus)
 * @param external_constants External round constants (must match computed rounds_f)
 * @param internal_constants Internal round constants (must match computed rounds_p)
 */
template<typename F, typename A, size_t WIDTH, uint64_t D>
std::shared_ptr<Poseidon2<F, A, WIDTH, D>> create_poseidon2_128(
    uint64_t field_order,
    const ExternalLayerConstants<F, WIDTH>& external_constants,
    const std::vector<F>& internal_constants
) {
    auto [rounds_f, rounds_p] = poseidon2_round_numbers_128(WIDTH, D, field_order);

    // Verify that the provided constants match the required rounds
    if (external_constants.get_initial_constants().size() != rounds_f / 2) {
        throw std::runtime_error("External constants size mismatch");
    }
    if (internal_constants.size() != rounds_p) {
        throw std::runtime_error("Internal constants size mismatch");
    }

    return create_poseidon2<F, A, WIDTH, D>(external_constants, internal_constants);
}

} // namespace poseidon2


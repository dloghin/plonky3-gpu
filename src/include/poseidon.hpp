#pragma once

#include <array>
#include <vector>
#include <cassert>
#include <memory>

namespace p3_poseidon {

/**
 * @brief Trait for MDS (Maximum Distance Separable) matrix permutations
 *
 * This trait represents permutations that operate on fixed-size arrays
 * and provide MDS properties required for cryptographic security.
 *
 * @tparam T Element type
 * @tparam WIDTH Size of the state array
 */
template<typename T, size_t WIDTH>
class MdsPermutation {
public:
    virtual ~MdsPermutation() = default;

    /**
     * @brief Permute the state array
     * @param state The state to permute
     * @return The permuted state
     */
    virtual std::array<T, WIDTH> permute(const std::array<T, WIDTH>& state) const = 0;

    /**
     * @brief Permute the state array in-place
     * @param state The state to permute (modified in-place)
     */
    virtual void permute_mut(std::array<T, WIDTH>& state) const = 0;
};

/**
 * @brief The Poseidon permutation
 *
 * This is a cryptographic permutation designed for zero-knowledge proof systems.
 * It uses a substitution-permutation network with:
 * - Full rounds at the beginning and end (all elements undergo S-box)
 * - Partial rounds in the middle (only first element undergoes S-box)
 * - MDS matrix multiplication after each round
 *
 * @tparam F Base field type (must be a prime field)
 * @tparam A Algebra type (typically F or a packed field)
 * @tparam Mds MDS matrix permutation type
 * @tparam WIDTH State width
 * @tparam ALPHA S-box exponent (must be coprime to field order - 1)
 */
template<typename F, typename A, typename Mds, size_t WIDTH, uint64_t ALPHA>
class Poseidon {
private:
    size_t half_num_full_rounds;
    size_t num_partial_rounds;
    std::vector<F> constants;
    std::shared_ptr<Mds> mds;

public:
    /**
     * @brief Construct a new Poseidon permutation
     *
     * @param half_num_full_rounds Half the number of full rounds (rounds before + after partial)
     * @param num_partial_rounds Number of partial rounds
     * @param constants Round constants (must have WIDTH * num_rounds elements)
     * @param mds_matrix MDS matrix for the linear layer
     *
     * The total number of rounds is 2 * half_num_full_rounds + num_partial_rounds.
     * The constants vector must have exactly WIDTH * num_rounds elements.
     */
    Poseidon(
        size_t half_num_full_rounds,
        size_t num_partial_rounds,
        std::vector<F> constants,
        std::shared_ptr<Mds> mds_matrix
    ) : half_num_full_rounds(half_num_full_rounds),
        num_partial_rounds(num_partial_rounds),
        constants(std::move(constants)),
        mds(mds_matrix)
    {
        size_t num_rounds = 2 * half_num_full_rounds + num_partial_rounds;
        assert(this->constants.size() == WIDTH * num_rounds &&
               "Constants vector must have WIDTH * num_rounds elements");
    }

    /**
     * @brief Permute the state
     * @param state Input state
     * @return Permuted state
     */
    std::array<A, WIDTH> permute(std::array<A, WIDTH> state) const {
        permute_mut(state);
        return state;
    }

    /**
     * @brief Permute the state in-place
     * @param state State to permute (modified in-place)
     */
    void permute_mut(std::array<A, WIDTH>& state) const {
        size_t round_ctr = 0;

        // First half of full rounds
        half_full_rounds(state, round_ctr);

        // Partial rounds
        partial_rounds(state, round_ctr);

        // Second half of full rounds
        half_full_rounds(state, round_ctr);
    }

    /**
     * @brief Get the number of half full rounds
     */
    size_t get_half_num_full_rounds() const {
        return half_num_full_rounds;
    }

    /**
     * @brief Get the number of partial rounds
     */
    size_t get_num_partial_rounds() const {
        return num_partial_rounds;
    }

    /**
     * @brief Get total number of rounds
     */
    size_t get_total_rounds() const {
        return 2 * half_num_full_rounds + num_partial_rounds;
    }

private:
    /**
     * @brief Execute half of the full rounds
     * @param state State array
     * @param round_ctr Current round counter (modified)
     */
    void half_full_rounds(std::array<A, WIDTH>& state, size_t& round_ctr) const {
        for (size_t i = 0; i < half_num_full_rounds; ++i) {
            constant_layer(state, round_ctr);
            full_sbox_layer(state);
            mds->permute_mut(state);
            ++round_ctr;
        }
    }

    /**
     * @brief Execute the partial rounds
     * @param state State array
     * @param round_ctr Current round counter (modified)
     */
    void partial_rounds(std::array<A, WIDTH>& state, size_t& round_ctr) const {
        for (size_t i = 0; i < num_partial_rounds; ++i) {
            constant_layer(state, round_ctr);
            partial_sbox_layer(state);
            mds->permute_mut(state);
            ++round_ctr;
        }
    }

    /**
     * @brief Apply S-box to all elements (full round)
     * @param state State array
     */
    void full_sbox_layer(std::array<A, WIDTH>& state) const {
        for (auto& x : state) {
            x = injective_exp_n<A, ALPHA>(x);
        }
    }

    /**
     * @brief Apply S-box to first element only (partial round)
     * @param state State array
     */
    void partial_sbox_layer(std::array<A, WIDTH>& state) const {
        state[0] = injective_exp_n<A, ALPHA>(state[0]);
    }

    /**
     * @brief Add round constants to state
     * @param state State array
     * @param round Current round number
     */
    void constant_layer(std::array<A, WIDTH>& state, size_t round) const {
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] += constants[round * WIDTH + i];
        }
    }

    /**
     * @brief Apply the injective monomial x^ALPHA
     *
     * This requires that ALPHA is coprime to the field order - 1,
     * which makes the map x -> x^ALPHA injective.
     *
     * @tparam T Type (must have injective_exp_n method)
     * @tparam EXP Exponent
     * @param x Input value
     * @return x^EXP
     */
    template<typename T, uint64_t EXP>
    static T injective_exp_n(const T& x) {
        // For fields that support template-based exponentiation
        return x.template injective_exp_n<EXP>();
    }
};

/**
 * @brief Factory function to create a Poseidon instance
 *
 * @tparam F Base field type
 * @tparam A Algebra type
 * @tparam Mds MDS matrix type
 * @tparam WIDTH State width
 * @tparam ALPHA S-box exponent
 *
 * @param half_num_full_rounds Half the number of full rounds
 * @param num_partial_rounds Number of partial rounds
 * @param constants Round constants
 * @param mds MDS matrix
 * @return Shared pointer to Poseidon instance
 */
template<typename F, typename A, typename Mds, size_t WIDTH, uint64_t ALPHA>
std::shared_ptr<Poseidon<F, A, Mds, WIDTH, ALPHA>> create_poseidon(
    size_t half_num_full_rounds,
    size_t num_partial_rounds,
    std::vector<F> constants,
    std::shared_ptr<Mds> mds
) {
    return std::make_shared<Poseidon<F, A, Mds, WIDTH, ALPHA>>(
        half_num_full_rounds,
        num_partial_rounds,
        std::move(constants),
        mds
    );
}

} // namespace p3_poseidon


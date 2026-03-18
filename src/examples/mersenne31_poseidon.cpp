#include "poseidon.hpp"
#include "mds_matrix.hpp"
#include "mersenne31.hpp"
#include <iostream>
#include <iomanip>
#include <random>

using namespace p3_field;
using namespace p3_poseidon;

/**
 * @brief Generate random constants for testing
 */
std::vector<Mersenne31> generate_constants(size_t count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Mersenne31> constants;
    constants.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        uint64_t val = rng() % Mersenne31::PRIME;
        constants.push_back(Mersenne31(val));
    }

    return constants;
}

/**
 * @brief Example: Poseidon with Mersenne31 field, WIDTH=16, ALPHA=5
 */
void example_mersenne31_16() {
    std::cout << "=== Poseidon with Mersenne31, WIDTH=16, ALPHA=5 ===" << std::endl;

    constexpr size_t WIDTH = 16;
    constexpr uint64_t ALPHA = 5;  // Note: Mersenne31 uses ALPHA=5

    const size_t half_num_full_rounds = 4;
    const size_t num_partial_rounds = 22;
    const size_t total_rounds = 2 * half_num_full_rounds + num_partial_rounds;

    auto constants = generate_constants(WIDTH * total_rounds, 1);
    auto mds = std::make_shared<MdsMatrixMersenne3116<Mersenne31>>();

    auto poseidon = create_poseidon<Mersenne31, Mersenne31, MdsPermutation<Mersenne31, WIDTH>, WIDTH, ALPHA>(
        half_num_full_rounds,
        num_partial_rounds,
        constants,
        mds
    );

    std::array<Mersenne31, WIDTH> state;
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = Mersenne31::zero();
    }

    std::cout << "Input state (first 8): [";
    for (size_t i = 0; i < 8; ++i) {
        std::cout << state[i].value();
        if (i < 7) std::cout << ", ";
    }
    std::cout << ", ...]" << std::endl;

    poseidon->permute_mut(state);

    std::cout << "Output state (first 8): [";
    for (size_t i = 0; i < 8; ++i) {
        std::cout << state[i].value();
        if (i < 7) std::cout << ", ";
    }
    std::cout << ", ...]" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Example: Poseidon with Mersenne31 field, WIDTH=32, ALPHA=5
 */
void example_mersenne31_32() {
    std::cout << "=== Poseidon with Mersenne31, WIDTH=32, ALPHA=5 ===" << std::endl;

    constexpr size_t WIDTH = 32;
    constexpr uint64_t ALPHA = 5;

    const size_t half_num_full_rounds = 4;
    const size_t num_partial_rounds = 22;
    const size_t total_rounds = 2 * half_num_full_rounds + num_partial_rounds;

    auto constants = generate_constants(WIDTH * total_rounds, 2);
    auto mds = std::make_shared<MdsMatrixMersenne3132<Mersenne31>>();

    auto poseidon = create_poseidon<Mersenne31, Mersenne31, MdsPermutation<Mersenne31, WIDTH>, WIDTH, ALPHA>(
        half_num_full_rounds,
        num_partial_rounds,
        constants,
        mds
    );

    std::array<Mersenne31, WIDTH> state;
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = Mersenne31(i + 1);
    }

    std::cout << "Input state (first 8): [";
    for (size_t i = 0; i < 8; ++i) {
        std::cout << state[i].value();
        if (i < 7) std::cout << ", ";
    }
    std::cout << ", ...]" << std::endl;

    poseidon->permute_mut(state);

    std::cout << "Output state (first 8): [";
    for (size_t i = 0; i < 8; ++i) {
        std::cout << state[i].value();
        if (i < 7) std::cout << ", ";
    }
    std::cout << ", ...]" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Compare performance characteristics
 */
void compare_widths() {
    std::cout << "=== Comparing Different Widths (Mersenne31, ALPHA=5) ===" << std::endl;

    const size_t half_num_full_rounds = 4;
    const size_t num_partial_rounds = 22;

    std::cout << "WIDTH | Total Rounds | Full Rounds | Partial Rounds" << std::endl;
    std::cout << "------|--------------|-------------|---------------" << std::endl;
    std::cout << "  16  |      " << (2 * half_num_full_rounds + num_partial_rounds)
              << "      |      " << (2 * half_num_full_rounds)
              << "     |       " << num_partial_rounds << std::endl;
    std::cout << "  32  |      " << (2 * half_num_full_rounds + num_partial_rounds)
              << "      |      " << (2 * half_num_full_rounds)
              << "     |       " << num_partial_rounds << std::endl;
    std::cout << std::endl;

    std::cout << "Note: Partial rounds are more efficient as they only apply" << std::endl;
    std::cout << "      the S-box to the first element." << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "Poseidon Permutation Examples with Mersenne31 Field" << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << std::endl;

    example_mersenne31_16();
    example_mersenne31_32();
    compare_widths();

    std::cout << "All examples completed successfully!" << std::endl;

    return 0;
}


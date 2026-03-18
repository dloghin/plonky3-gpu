#include "poseidon.hpp"
#include "mds_matrix.hpp"
#include "goldilocks.hpp"
#include <iostream>
#include <iomanip>
#include <random>

using namespace p3_field;
using namespace p3_poseidon;

/**
 * @brief Generate random constants for testing
 */
std::vector<Goldilocks> generate_constants(size_t count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Goldilocks> constants;
    constants.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        uint64_t val = rng() % Goldilocks::PRIME;
        constants.push_back(Goldilocks(val));
    }

    return constants;
}

/**
 * @brief Example: Poseidon with Goldilocks field, WIDTH=8, ALPHA=7
 */
void example_goldilocks_8() {
    std::cout << "=== Poseidon with Goldilocks, WIDTH=8, ALPHA=7 ===" << std::endl;

    constexpr size_t WIDTH = 8;
    constexpr uint64_t ALPHA = 7;

    const size_t half_num_full_rounds = 4;
    const size_t num_partial_rounds = 22;
    const size_t total_rounds = 2 * half_num_full_rounds + num_partial_rounds;

    auto constants = generate_constants(WIDTH * total_rounds, 1);
    auto mds = std::make_shared<MdsMatrixGoldilocks8<Goldilocks>>();

    auto poseidon = create_poseidon<Goldilocks, Goldilocks, MdsPermutation<Goldilocks, WIDTH>, WIDTH, ALPHA>(
        half_num_full_rounds,
        num_partial_rounds,
        constants,
        mds
    );

    std::array<Goldilocks, WIDTH> state;
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = Goldilocks::zero();
    }

    std::cout << "Input state: [";
    for (size_t i = 0; i < WIDTH; ++i) {
        std::cout << state[i].value();
        if (i < WIDTH - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    poseidon->permute_mut(state);

    std::cout << "Output state: [";
    for (size_t i = 0; i < WIDTH; ++i) {
        std::cout << state[i].value();
        if (i < WIDTH - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Example: Poseidon with Goldilocks field, WIDTH=12, ALPHA=7
 */
void example_goldilocks_12() {
    std::cout << "=== Poseidon with Goldilocks, WIDTH=12, ALPHA=7 ===" << std::endl;

    constexpr size_t WIDTH = 12;
    constexpr uint64_t ALPHA = 7;

    const size_t half_num_full_rounds = 4;
    const size_t num_partial_rounds = 22;
    const size_t total_rounds = 2 * half_num_full_rounds + num_partial_rounds;

    auto constants = generate_constants(WIDTH * total_rounds, 2);
    auto mds = std::make_shared<MdsMatrixGoldilocks12<Goldilocks>>();

    auto poseidon = create_poseidon<Goldilocks, Goldilocks, MdsPermutation<Goldilocks, WIDTH>, WIDTH, ALPHA>(
        half_num_full_rounds,
        num_partial_rounds,
        constants,
        mds
    );

    std::array<Goldilocks, WIDTH> state;
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = Goldilocks(i + 1);
    }

    std::cout << "Input state: [";
    for (size_t i = 0; i < WIDTH; ++i) {
        std::cout << state[i].value();
        if (i < WIDTH - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    poseidon->permute_mut(state);

    std::cout << "Output state: [";
    for (size_t i = 0; i < WIDTH; ++i) {
        std::cout << state[i].value();
        if (i < WIDTH - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Example: Poseidon with Goldilocks field, WIDTH=16, ALPHA=7
 */
void example_goldilocks_16() {
    std::cout << "=== Poseidon with Goldilocks, WIDTH=16, ALPHA=7 ===" << std::endl;

    constexpr size_t WIDTH = 16;
    constexpr uint64_t ALPHA = 7;

    const size_t half_num_full_rounds = 4;
    const size_t num_partial_rounds = 22;
    const size_t total_rounds = 2 * half_num_full_rounds + num_partial_rounds;

    auto constants = generate_constants(WIDTH * total_rounds, 3);
    auto mds = std::make_shared<MdsMatrixGoldilocks16<Goldilocks>>();

    auto poseidon = create_poseidon<Goldilocks, Goldilocks, MdsPermutation<Goldilocks, WIDTH>, WIDTH, ALPHA>(
        half_num_full_rounds,
        num_partial_rounds,
        constants,
        mds
    );

    std::array<Goldilocks, WIDTH> state;
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = Goldilocks((i * 12345) % Goldilocks::PRIME);
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

int main() {
    std::cout << "Poseidon Permutation Examples with Goldilocks Field" << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << std::endl;

    example_goldilocks_8();
    example_goldilocks_12();
    example_goldilocks_16();

    std::cout << "All examples completed successfully!" << std::endl;

    return 0;
}


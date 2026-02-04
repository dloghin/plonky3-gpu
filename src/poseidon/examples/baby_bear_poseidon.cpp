#include "poseidon.hpp"
#include "mds_matrix.hpp"
#include "baby_bear.hpp"
#include <iostream>
#include <iomanip>
#include <random>

using namespace p3_field;
using namespace p3_poseidon;

/**
 * @brief Generate random constants for testing
 */
std::vector<BabyBear> generate_constants(size_t count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<BabyBear> constants;
    constants.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        uint64_t val = rng() % BabyBear::PRIME;
        constants.push_back(BabyBear(val));
    }

    return constants;
}

/**
 * @brief Example: Poseidon with BabyBear field, WIDTH=16, ALPHA=7
 */
void example_baby_bear_16() {
    std::cout << "=== Poseidon with BabyBear, WIDTH=16, ALPHA=7 ===" << std::endl;

    constexpr size_t WIDTH = 16;
    constexpr uint64_t ALPHA = 7;

    // Round configuration (these should be calculated for security)
    // For demonstration, using typical values
    const size_t half_num_full_rounds = 4;
    const size_t num_partial_rounds = 22;
    const size_t total_rounds = 2 * half_num_full_rounds + num_partial_rounds;

    // Generate random constants
    auto constants = generate_constants(WIDTH * total_rounds, 1);

    // Create MDS matrix
    auto mds = std::make_shared<MdsMatrixBabyBear16<BabyBear>>();

    // Create Poseidon instance
    auto poseidon = create_poseidon<BabyBear, BabyBear, MdsPermutation<BabyBear, WIDTH>, WIDTH, ALPHA>(
        half_num_full_rounds,
        num_partial_rounds,
        constants,
        mds
    );

    // Create input state (all zeros)
    std::array<BabyBear, WIDTH> state;
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = BabyBear::zero();
    }

    std::cout << "Input state: [";
    for (size_t i = 0; i < WIDTH; ++i) {
        std::cout << state[i].value();
        if (i < WIDTH - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Apply permutation
    poseidon->permute_mut(state);

    std::cout << "Output state: [";
    for (size_t i = 0; i < WIDTH; ++i) {
        std::cout << state[i].value();
        if (i < WIDTH - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Total rounds: " << poseidon->get_total_rounds() << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Example: Poseidon with BabyBear field, WIDTH=24, ALPHA=7
 */
void example_baby_bear_24() {
    std::cout << "=== Poseidon with BabyBear, WIDTH=24, ALPHA=7 ===" << std::endl;

    constexpr size_t WIDTH = 24;
    constexpr uint64_t ALPHA = 7;

    const size_t half_num_full_rounds = 4;
    const size_t num_partial_rounds = 22;
    const size_t total_rounds = 2 * half_num_full_rounds + num_partial_rounds;

    auto constants = generate_constants(WIDTH * total_rounds, 2);
    auto mds = std::make_shared<MdsMatrixBabyBear24<BabyBear>>();

    auto poseidon = create_poseidon<BabyBear, BabyBear, MdsPermutation<BabyBear, WIDTH>, WIDTH, ALPHA>(
        half_num_full_rounds,
        num_partial_rounds,
        constants,
        mds
    );

    // Create input state with some non-zero values
    std::array<BabyBear, WIDTH> state;
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = BabyBear(i + 1);
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
 * @brief Test MDS matrix properties
 */
void test_mds_matrix() {
    std::cout << "=== Testing MDS Matrix (BabyBear, WIDTH=8) ===" << std::endl;

    MdsMatrixBabyBear8<BabyBear> mds;

    // Test with known input
    std::array<BabyBear, 8> input = {
        BabyBear(391474477),
        BabyBear(1174409341),
        BabyBear(666967492),
        BabyBear(1852498830),
        BabyBear(1801235316),
        BabyBear(820595865),
        BabyBear(585587525),
        BabyBear(1348326858)
    };

    std::array<BabyBear, 8> expected = {
        BabyBear(1752937716),
        BabyBear(1801468855),
        BabyBear(1102954394),
        BabyBear(284747746),
        BabyBear(1636355768),
        BabyBear(205443234),
        BabyBear(1235359747),
        BabyBear(1159982032)
    };

    auto output = mds.permute(input);

    bool success = true;
    for (size_t i = 0; i < 8; ++i) {
        if (output[i].value() != expected[i].value()) {
            success = false;
            std::cout << "Mismatch at index " << i << ": "
                      << "expected " << expected[i].value()
                      << ", got " << output[i].value() << std::endl;
        }
    }

    if (success) {
        std::cout << "✓ MDS matrix test passed!" << std::endl;
    } else {
        std::cout << "✗ MDS matrix test failed!" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Poseidon Permutation Examples with BabyBear Field" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << std::endl;

    // Test MDS matrix first
    test_mds_matrix();

    // Run examples
    example_baby_bear_16();
    example_baby_bear_24();

    std::cout << "All examples completed successfully!" << std::endl;

    return 0;
}


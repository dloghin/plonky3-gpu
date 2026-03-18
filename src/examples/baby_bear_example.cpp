/**
 * @file baby_bear_example.cpp
 * @brief Poseidon2 example using the Baby Bear field
 */

#include "poseidon2.hpp"
#include "baby_bear.hpp"
#include <iostream>
#include <random>
#include <iomanip>

using namespace p3_field;
using namespace poseidon2;

// Generate random constants for demonstration
template<size_t WIDTH>
std::array<BabyBear, WIDTH> generate_random_array(std::mt19937_64& rng) {
    std::array<BabyBear, WIDTH> result;
    std::uniform_int_distribution<uint32_t> dist(0, BabyBear::PRIME - 1);
    for (size_t i = 0; i < WIDTH; ++i) {
        result[i] = BabyBear(dist(rng));
    }
    return result;
}

int main() {
    std::cout << "Poseidon2 with Baby Bear Field - Example\n";
    std::cout << "========================================\n\n";

    // Configuration
    constexpr size_t WIDTH = 16;
    constexpr uint64_t D = 7; // D=7 works with Baby Bear
    const uint64_t FIELD_ORDER = BabyBear::PRIME_U64;

    std::cout << "Configuration:\n";
    std::cout << "  Field: Baby Bear (2^31 - 2^27 + 1 = " << FIELD_ORDER << ")\n";
    std::cout << "  State width: " << WIDTH << "\n";
    std::cout << "  S-box exponent (D): " << D << "\n\n";

    // Get round numbers
    try {
        auto [rounds_f, rounds_p] = poseidon2_round_numbers_128(
            WIDTH, D, FIELD_ORDER
        );

        std::cout << "Round numbers for 128-bit security:\n";
        std::cout << "  Full rounds (external): " << rounds_f << "\n";
        std::cout << "  Partial rounds (internal): " << rounds_p << "\n";
        std::cout << "  Initial external: " << rounds_f / 2 << "\n";
        std::cout << "  Terminal external: " << rounds_f / 2 << "\n\n";

        // Generate random constants
        std::mt19937_64 rng(12345);

        std::vector<std::array<BabyBear, WIDTH>> initial_constants;
        std::vector<std::array<BabyBear, WIDTH>> terminal_constants;

        for (size_t i = 0; i < rounds_f / 2; ++i) {
            initial_constants.push_back(generate_random_array<WIDTH>(rng));
            terminal_constants.push_back(generate_random_array<WIDTH>(rng));
        }

        std::vector<BabyBear> internal_constants;
        for (size_t i = 0; i < rounds_p; ++i) {
            internal_constants.push_back(BabyBear(rng() % FIELD_ORDER));
        }

        std::cout << "Generated " << initial_constants.size() << " initial round constants\n";
        std::cout << "Generated " << terminal_constants.size() << " terminal round constants\n";
        std::cout << "Generated " << internal_constants.size() << " internal round constants\n\n";

        // Create external constants
        ExternalLayerConstants<BabyBear, WIDTH> external_constants(
            initial_constants,
            terminal_constants
        );

        // Create Poseidon2 instance
        auto poseidon = create_poseidon2<BabyBear, BabyBear, WIDTH, D>(
            external_constants,
            internal_constants
        );

        std::cout << "Successfully created Poseidon2 instance!\n\n";

        // Test the permutation
        std::array<BabyBear, WIDTH> state;
        std::cout << "Initial state:\n  ";
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] = BabyBear(i + 1);
            std::cout << std::setw(10) << state[i] << " ";
            if ((i + 1) % 8 == 0) std::cout << "\n  ";
        }
        std::cout << "\n\n";

        // Apply permutation
        std::cout << "Applying Poseidon2 permutation...\n";
        poseidon->permute_mut(state);

        std::cout << "\nPermuted state:\n  ";
        for (size_t i = 0; i < WIDTH; ++i) {
            std::cout << std::setw(10) << state[i] << " ";
            if ((i + 1) % 8 == 0) std::cout << "\n  ";
        }
        std::cout << "\n\n";

        // Verify it's different from input
        bool all_different = true;
        for (size_t i = 0; i < WIDTH; ++i) {
            if (state[i] == BabyBear(i + 1)) {
                all_different = false;
                break;
            }
        }

        if (all_different) {
            std::cout << "✓ Permutation successfully transformed the state\n";
        } else {
            std::cout << "✗ Warning: Some elements unchanged\n";
        }

        // Test matrix operations
        std::cout << "\nTesting matrix operations:\n";
        std::array<BabyBear, 4> test_vec = {
            BabyBear(1),
            BabyBear(2),
            BabyBear(3),
            BabyBear(4)
        };

        std::cout << "  Original 4-vector: ";
        for (const auto& v : test_vec) std::cout << v << " ";
        std::cout << "\n";

        apply_mat4(test_vec);
        std::cout << "  After MDSMat4:     ";
        for (const auto& v : test_vec) std::cout << v << " ";
        std::cout << "\n\n";

        std::cout << "All tests completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}


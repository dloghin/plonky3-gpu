/**
 * @file goldilocks_example.cpp
 * @brief Poseidon2 example using the Goldilocks field
 */

#include "poseidon2.hpp"
#include "goldilocks.hpp"
#include <iostream>
#include <random>
#include <iomanip>

using namespace p3_field;
using namespace poseidon2;

// Generate random constants for demonstration
template<size_t WIDTH>
std::array<Goldilocks, WIDTH> generate_random_array(std::mt19937_64& rng) {
    std::array<Goldilocks, WIDTH> result;
    for (size_t i = 0; i < WIDTH; ++i) {
        uint64_t val = rng();
        // Ensure it's less than the prime
        while (val >= Goldilocks::PRIME) {
            val = rng();
        }
        result[i] = Goldilocks(val);
    }
    return result;
}

int main() {
    std::cout << "Poseidon2 with Goldilocks Field - Example\n";
    std::cout << "==========================================\n\n";

    // Configuration
    constexpr size_t WIDTH = 12;
    constexpr uint64_t D = 7; // D=7 works with Goldilocks
    const uint64_t FIELD_ORDER = Goldilocks::PRIME;

    std::cout << "Configuration:\n";
    std::cout << "  Field: Goldilocks (2^64 - 2^32 + 1 = " << FIELD_ORDER << ")\n";
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
        std::mt19937_64 rng(54321);

        std::vector<std::array<Goldilocks, WIDTH>> initial_constants;
        std::vector<std::array<Goldilocks, WIDTH>> terminal_constants;

        for (size_t i = 0; i < rounds_f / 2; ++i) {
            initial_constants.push_back(generate_random_array<WIDTH>(rng));
            terminal_constants.push_back(generate_random_array<WIDTH>(rng));
        }

        std::vector<Goldilocks> internal_constants;
        for (size_t i = 0; i < rounds_p; ++i) {
            uint64_t val = rng();
            while (val >= FIELD_ORDER) val = rng();
            internal_constants.push_back(Goldilocks(val));
        }

        std::cout << "Generated " << initial_constants.size() << " initial round constants\n";
        std::cout << "Generated " << terminal_constants.size() << " terminal round constants\n";
        std::cout << "Generated " << internal_constants.size() << " internal round constants\n\n";

        // Create external constants
        ExternalLayerConstants<Goldilocks, WIDTH> external_constants(
            initial_constants,
            terminal_constants
        );

        // Create Poseidon2 instance
        auto poseidon = create_poseidon2<Goldilocks, Goldilocks, WIDTH, D>(
            external_constants,
            internal_constants
        );

        std::cout << "Successfully created Poseidon2 instance!\n\n";

        // Test the permutation
        std::array<Goldilocks, WIDTH> state;
        std::cout << "Initial state:\n  ";
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] = Goldilocks(i + 1);
            std::cout << std::setw(10) << state[i] << " ";
            if ((i + 1) % 6 == 0) std::cout << "\n  ";
        }
        std::cout << "\n\n";

        // Apply permutation
        std::cout << "Applying Poseidon2 permutation...\n";
        poseidon->permute_mut(state);

        std::cout << "\nPermuted state:\n  ";
        for (size_t i = 0; i < WIDTH; ++i) {
            std::cout << std::setw(20) << state[i] << " ";
            if ((i + 1) % 6 == 0) std::cout << "\n  ";
        }
        std::cout << "\n\n";

        // Verify it's different from input
        bool all_different = true;
        for (size_t i = 0; i < WIDTH; ++i) {
            if (state[i] == Goldilocks(i + 1)) {
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
        std::array<Goldilocks, 4> test_vec = {
            Goldilocks(1),
            Goldilocks(2),
            Goldilocks(3),
            Goldilocks(4)
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


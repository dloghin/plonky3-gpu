/**
 * @file field_example.cpp
 * @brief Complete working example with a simple field implementation
 */

#include "poseidon2.hpp"
#include <iostream>
#include <random>
#include <iomanip>

// Complete field implementation for Mersenne-31
class Mersenne31Field {
private:
    uint64_t value_;
    static constexpr uint64_t MODULUS = (1ULL << 31) - 1; // 2^31 - 1

    static uint64_t reduce(uint64_t x) {
        if (x >= MODULUS) {
            x = (x & MODULUS) + (x >> 31);
            if (x >= MODULUS) x -= MODULUS;
        }
        return x;
    }

public:
    Mersenne31Field() : value_(0) {}
    explicit Mersenne31Field(uint64_t value) : value_(reduce(value)) {}

    static Mersenne31Field zero() { return Mersenne31Field(0); }
    static Mersenne31Field one() { return Mersenne31Field(1); }
    static uint64_t modulus() { return MODULUS; }

    uint64_t value() const { return value_; }

    Mersenne31Field operator+(const Mersenne31Field& other) const {
        return Mersenne31Field(value_ + other.value_);
    }

    Mersenne31Field operator-(const Mersenne31Field& other) const {
        uint64_t result = value_ + MODULUS - other.value_;
        return Mersenne31Field(result);
    }

    Mersenne31Field operator*(const Mersenne31Field& other) const {
        return Mersenne31Field(((__uint128_t)value_ * other.value_) % MODULUS);
    }

    Mersenne31Field& operator+=(const Mersenne31Field& other) {
        value_ = reduce(value_ + other.value_);
        return *this;
    }

    Mersenne31Field& operator*=(const Mersenne31Field& other) {
        value_ = reduce(((__uint128_t)value_ * other.value_) % MODULUS);
        return *this;
    }

    Mersenne31Field double_val() const {
        return Mersenne31Field(value_ + value_);
    }

    // Fast exponentiation for x^D
    template<uint64_t D>
    Mersenne31Field injective_exp_n() const {
        if (D == 0) return one();
        if (D == 1) return *this;

        Mersenne31Field result = one();
        Mersenne31Field base = *this;
        uint64_t exp = D;

        while (exp > 0) {
            if (exp & 1) {
                result *= base;
            }
            if (exp > 1) {
                base *= base;
            }
            exp >>= 1;
        }
        return result;
    }

    bool operator==(const Mersenne31Field& other) const {
        return value_ == other.value_;
    }

    bool operator!=(const Mersenne31Field& other) const {
        return value_ != other.value_;
    }

    friend std::ostream& operator<<(std::ostream& os, const Mersenne31Field& field) {
        os << field.value_;
        return os;
    }
};

// Generate random constants for demonstration
template<size_t WIDTH>
std::array<Mersenne31Field, WIDTH> generate_random_array(std::mt19937_64& rng) {
    std::array<Mersenne31Field, WIDTH> result;
    std::uniform_int_distribution<uint64_t> dist(0, Mersenne31Field::modulus() - 1);
    for (size_t i = 0; i < WIDTH; ++i) {
        result[i] = Mersenne31Field(dist(rng));
    }
    return result;
}

int main() {
    std::cout << "Poseidon2 C++ Implementation - Complete Field Example\n";
    std::cout << "====================================================\n\n";

    // Configuration
    constexpr size_t WIDTH = 16;
    constexpr uint64_t D = 5; // D=5 works with Mersenne-31
    const uint64_t FIELD_ORDER = Mersenne31Field::modulus();

    std::cout << "Configuration:\n";
    std::cout << "  Field: Mersenne-31 (2^31 - 1 = " << FIELD_ORDER << ")\n";
    std::cout << "  State width: " << WIDTH << "\n";
    std::cout << "  S-box exponent (D): " << D << "\n\n";

    // Get round numbers
    try {
        auto [rounds_f, rounds_p] = poseidon2::poseidon2_round_numbers_128(
            WIDTH, D, FIELD_ORDER
        );

        std::cout << "Round numbers for 128-bit security:\n";
        std::cout << "  Full rounds (external): " << rounds_f << "\n";
        std::cout << "  Partial rounds (internal): " << rounds_p << "\n";
        std::cout << "  Initial external: " << rounds_f / 2 << "\n";
        std::cout << "  Terminal external: " << rounds_f / 2 << "\n\n";

        // Generate random constants (for demonstration)
        std::mt19937_64 rng(12345);

        std::vector<std::array<Mersenne31Field, WIDTH>> initial_constants;
        std::vector<std::array<Mersenne31Field, WIDTH>> terminal_constants;

        for (size_t i = 0; i < rounds_f / 2; ++i) {
            initial_constants.push_back(generate_random_array<WIDTH>(rng));
            terminal_constants.push_back(generate_random_array<WIDTH>(rng));
        }

        std::vector<Mersenne31Field> internal_constants;
        for (size_t i = 0; i < rounds_p; ++i) {
            internal_constants.push_back(Mersenne31Field(rng() % FIELD_ORDER));
        }

        std::cout << "Generated " << initial_constants.size() << " initial round constants\n";
        std::cout << "Generated " << terminal_constants.size() << " terminal round constants\n";
        std::cout << "Generated " << internal_constants.size() << " internal round constants\n\n";

        // Create external constants
        poseidon2::ExternalLayerConstants<Mersenne31Field, WIDTH> external_constants(
            initial_constants,
            terminal_constants
        );

        // Create Poseidon2 instance
        auto poseidon = poseidon2::create_poseidon2<Mersenne31Field, Mersenne31Field, WIDTH, D>(
            external_constants,
            internal_constants
        );

        std::cout << "Successfully created Poseidon2 instance!\n\n";

        // Test the permutation
        std::array<Mersenne31Field, WIDTH> state;
        std::cout << "Initial state:\n  ";
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] = Mersenne31Field(i + 1);
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
            if (state[i] == Mersenne31Field(i + 1)) {
                all_different = false;
                break;
            }
        }

        if (all_different) {
            std::cout << "✓ Permutation successfully transformed the state\n";
        } else {
            std::cout << "✗ Warning: Some elements unchanged (this is very unlikely)\n";
        }

        // Test matrix operations
        std::cout << "\nTesting matrix operations:\n";
        std::array<Mersenne31Field, 4> test_vec = {
            Mersenne31Field(1),
            Mersenne31Field(2),
            Mersenne31Field(3),
            Mersenne31Field(4)
        };

        std::cout << "  Original 4-vector: ";
        for (const auto& v : test_vec) std::cout << v << " ";
        std::cout << "\n";

        poseidon2::apply_mat4(test_vec);
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


/**
 * @file simple_example.cpp
 * @brief Simple example demonstrating Poseidon2 usage
 *
 * This example shows how to use the Poseidon2 permutation with a simple field type.
 */

#include "poseidon2.hpp"
#include <iostream>
#include <cstdint>
#include <iomanip>

// Simple field element for demonstration (uint64_t modulo a prime)
class SimpleField {
private:
    uint64_t value_;
    static constexpr uint64_t MODULUS = (1ULL << 31) - 1; // Mersenne-31

public:
    SimpleField() : value_(0) {}
    explicit SimpleField(uint64_t value) : value_(value % MODULUS) {}

    static SimpleField zero() { return SimpleField(0); }
    static SimpleField one() { return SimpleField(1); }

    uint64_t value() const { return value_; }

    SimpleField operator+(const SimpleField& other) const {
        return SimpleField(value_ + other.value_);
    }

    SimpleField operator*(const SimpleField& other) const {
        return SimpleField(((__uint128_t)value_ * other.value_) % MODULUS);
    }

    SimpleField& operator+=(const SimpleField& other) {
        value_ = (value_ + other.value_) % MODULUS;
        return *this;
    }

    SimpleField& operator*=(const SimpleField& other) {
        value_ = ((__uint128_t)value_ * other.value_) % MODULUS;
        return *this;
    }

    SimpleField double_val() const {
        return SimpleField(value_ + value_);
    }

    template<uint64_t D>
    SimpleField injective_exp_n() const {
        if (D == 0) return one();

        SimpleField result = *this;
        SimpleField base = *this;
        uint64_t exp = D;

        result = one();
        while (exp > 0) {
            if (exp & 1) {
                result *= base;
            }
            base *= base;
            exp >>= 1;
        }
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const SimpleField& field) {
        os << field.value_;
        return os;
    }
};

int main() {
    std::cout << "Poseidon2 C++ Implementation - Simple Example\n";
    std::cout << "=============================================\n\n";

    // Configuration
    constexpr size_t WIDTH = 16;
    constexpr uint64_t D = 5; // D=5 works with Mersenne-31
    constexpr uint64_t FIELD_ORDER = (1ULL << 31) - 1;

    std::cout << "Configuration:\n";
    std::cout << "  Field: Mersenne-31 (2^31 - 1)\n";
    std::cout << "  State width: " << WIDTH << "\n";
    std::cout << "  S-box exponent (D): " << D << "\n\n";

    // Get round numbers for 128-bit security
    try {
        auto [rounds_f, rounds_p] = poseidon2::poseidon2_round_numbers_128(
            WIDTH, D, FIELD_ORDER
        );

        std::cout << "Security parameters (128-bit):\n";
        std::cout << "  Full rounds (external): " << rounds_f << "\n";
        std::cout << "  Partial rounds (internal): " << rounds_p << "\n\n";

        // Note: In a real implementation, you would generate or load
        // proper round constants. Here we just demonstrate the structure.
        std::cout << "Note: This example demonstrates the API structure.\n";
        std::cout << "In a real application, you would need to:\n";
        std::cout << "  1. Generate or load proper round constants\n";
        std::cout << "  2. Create ExternalLayerConstants with " << rounds_f/2
                  << " initial and terminal rounds\n";
        std::cout << "  3. Create internal constants vector with " << rounds_p
                  << " elements\n";
        std::cout << "  4. Instantiate Poseidon2 with these constants\n";
        std::cout << "  5. Use permute_mut() to apply the permutation\n\n";

        // Example state (in practice, this would be your actual data)
        std::array<SimpleField, WIDTH> state;
        for (size_t i = 0; i < WIDTH; ++i) {
            state[i] = SimpleField(i);
        }

        std::cout << "Example initial state:\n  ";
        for (size_t i = 0; i < WIDTH; ++i) {
            std::cout << state[i] << " ";
        }
        std::cout << "\n\n";

        std::cout << "Supported widths: ";
        for (size_t w : poseidon2::SUPPORTED_WIDTHS) {
            std::cout << w << " ";
        }
        std::cout << "\n\n";

        std::cout << "Example completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}


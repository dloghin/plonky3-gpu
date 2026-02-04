/**
 * @file field_demo.cpp
 * @brief Demonstration of field arithmetic for Baby Bear, Goldilocks, and Mersenne-31
 */

#include "baby_bear.hpp"
#include "goldilocks.hpp"
#include "mersenne31.hpp"
#include <iostream>
#include <iomanip>

using namespace p3_field;

template<typename F>
void test_field(const std::string& name) {
    std::cout << "\n=== " << name << " Field Tests ===\n";

    // Basic operations
    F a = F::ONE + F::ONE + F::ONE; // 3
    F b = F::TWO + F::TWO;           // 4

    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    std::cout << "a + b = " << (a + b) << "\n";
    std::cout << "a * b = " << (a * b) << "\n";
    std::cout << "a - b = " << (a - b) << "\n";

    // Inverse
    if (a != F::ZERO) {
        F a_inv = a.inverse();
        F product = a * a_inv;
        std::cout << "a^(-1) = " << a_inv << "\n";
        std::cout << "a * a^(-1) = " << product << " (should be 1)\n";
    }

    // Powers
    F two = F::TWO;
    std::cout << "2^0 = " << two.exp_u64(0) << "\n";
    std::cout << "2^1 = " << two.exp_u64(1) << "\n";
    std::cout << "2^2 = " << two.exp_u64(2) << "\n";
    std::cout << "2^3 = " << two.exp_u64(3) << "\n";
    std::cout << "2^10 = " << two.exp_u64(10) << "\n";

    // Square and cube
    F x = F::TWO + F::ONE; // 3
    std::cout << "x = " << x << "\n";
    std::cout << "x^2 = " << x.square() << "\n";
    std::cout << "x^3 = " << x.cube() << "\n";

    // Doubling
    std::cout << "double(x) = " << x.double_val() << "\n";
}

int main() {
    std::cout << "C++ Field Arithmetic Library - Demonstration\n";
    std::cout << "============================================\n";

    // Test Baby Bear
    test_field<BabyBear>("Baby Bear (p = 2^31 - 2^27 + 1)");

    // Test Goldilocks
    test_field<Goldilocks>("Goldilocks (p = 2^64 - 2^32 + 1)");

    // Test Mersenne-31
    test_field<Mersenne31>("Mersenne-31 (p = 2^31 - 1)");

    // Specific tests for each field
    std::cout << "\n=== Field-Specific Tests ===\n";

    // Baby Bear constants
    std::cout << "\nBaby Bear Prime: " << BabyBear::PRIME << " (0x"
              << std::hex << BabyBear::PRIME << std::dec << ")\n";

    // Goldilocks constants
    std::cout << "Goldilocks Prime: " << Goldilocks::PRIME << " (0x"
              << std::hex << Goldilocks::PRIME << std::dec << ")\n";

    // Mersenne-31 constants
    std::cout << "Mersenne-31 Prime: " << Mersenne31::PRIME << " (0x"
              << std::hex << Mersenne31::PRIME << std::dec << ")\n";

    // Test injective power maps
    std::cout << "\n=== Injective Power Maps ===\n";

    BabyBear bb(7);
    auto bb_pow5 = bb.injective_exp_n<5>();
    std::cout << "Baby Bear: 7^5 = " << bb_pow5 << "\n";

    Goldilocks gl(7);
    auto gl_pow7 = gl.injective_exp_n<7>();
    std::cout << "Goldilocks: 7^7 = " << gl_pow7 << "\n";

    Mersenne31 m31(7);
    auto m31_pow5 = m31.injective_exp_n<5>();
    std::cout << "Mersenne-31: 7^5 = " << m31_pow5 << "\n";

    std::cout << "\nAll tests completed successfully!\n";

    return 0;
}


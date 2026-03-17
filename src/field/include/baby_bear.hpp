#pragma once

#include "field.hpp"
#include "cuda_compat.hpp"
#include <cstdint>
#include <array>

#if !P3_CUDA_ENABLED
#include <iostream>
#include <stdexcept>
#endif

namespace p3_field {

/**
 * @brief The Baby Bear prime field: p = 2^31 - 2^27 + 1
 *
 * This is the unique 31-bit prime with the highest possible 2-adicity (27).
 * Prime value: 0x78000001 = 2013265921
 *
 * This implementation supports both CPU (C++) and GPU (CUDA) execution.
 */
class BabyBear : public PrimeField<BabyBear> {
private:
    uint32_t value_;

public:
    static constexpr uint32_t PRIME = 0x78000001; // 2^31 - 2^27 + 1
    static constexpr uint64_t PRIME_U64 = PRIME;

    // Two-adic field constants
    // p - 1 = 2^27 * 15, so the 2-adicity is 27
    static constexpr size_t TWO_ADICITY = 27;

    // Number of usable random bits per field element (floor(log2(p)) = 30)
    static constexpr size_t FIELD_BITS = 31;

    // Multiplicative group generator (primitive root of F_p*)
    // 31 is a primitive root modulo p
    static constexpr uint32_t GENERATOR_VAL = 31;

    // Primitive 2^27-th root of unity in F_p
    static constexpr uint32_t ROOT_OF_UNITY_2_POW_27 = 440564289u;

    // Extension-field primitive roots used for bits=28,29 in BabyBear^4
    static constexpr uint32_t EXT_ROOT_2_POW_28_COEFF = 929455875u;
    static constexpr uint32_t EXT_ROOT_2_POW_29_COEFF = 1483681942u;

    // Constructors
    P3_HOST_DEVICE P3_CONSTEXPR_HD BabyBear() : value_(0) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit BabyBear(uint32_t value) : value_(reduce(value)) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit BabyBear(uint64_t value) : value_(reduce64(value)) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit BabyBear(int32_t value) : value_(from_signed(value)) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit BabyBear(int64_t value) : value_(from_signed64(value)) {}

    // Static factory methods for zero/one (GPU compatible)
    P3_HOST_DEVICE static BabyBear zero_val() { return BabyBear(); }
    P3_HOST_DEVICE static BabyBear one_val() { return BabyBear(static_cast<uint32_t>(1)); }
    P3_HOST_DEVICE static BabyBear two_val() { return BabyBear(static_cast<uint32_t>(2)); }
    P3_HOST_DEVICE static BabyBear neg_one_val() { return BabyBear(PRIME - 1); }

    // Legacy static constants (CPU only, for backward compatibility)
#if !P3_CUDA_ENABLED
    static const BabyBear ZERO;
    static const BabyBear ONE;
    static const BabyBear TWO;
    static const BabyBear NEG_ONE;
#endif

    // Access
    P3_HOST_DEVICE P3_CONSTEXPR_HD uint32_t value() const { return value_; }
    P3_HOST_DEVICE P3_CONSTEXPR_HD uint64_t as_canonical_u64() const { return value_; }

    // Arithmetic operations
    P3_HOST_DEVICE P3_CONSTEXPR_HD BabyBear add(const BabyBear& other) const {
        uint64_t sum = static_cast<uint64_t>(value_) + static_cast<uint64_t>(other.value_);
        return BabyBear(sum);
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD BabyBear sub(const BabyBear& other) const {
        uint64_t diff = static_cast<uint64_t>(value_) + PRIME - static_cast<uint64_t>(other.value_);
        return BabyBear(diff);
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD BabyBear mul(const BabyBear& other) const {
        uint64_t prod = static_cast<uint64_t>(value_) * static_cast<uint64_t>(other.value_);
        return BabyBear(prod);
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD bool equals(const BabyBear& other) const {
        return value_ == other.value_;
    }

    // Inverse using Fermat's little theorem: a^(-1) = a^(p-2)
    P3_HOST_DEVICE BabyBear inv() const {
#if !P3_CUDA_ENABLED
        if (value_ == 0) {
            throw std::runtime_error("Cannot invert zero");
        }
#else
        P3_ASSERT(value_ != 0);
#endif
        // Use Fermat's little theorem: a^(p-1) = 1, so a^(-1) = a^(p-2)
        return exp_u64(PRIME - 2);
    }

    // Injective power map for D=5 (x^5 is injective for Baby Bear)
    template<uint64_t D>
    P3_HOST_DEVICE BabyBear injective_exp_n() const {
        return exp_const_u64<D>();
    }

    // For D=7, Baby Bear doesn't support it (gcd(7, p-1) = 7)
    // But for D=5, we can compute the inverse:
    // 5 * 1725656503 ≡ 1 (mod p-1) for D=5
    template<uint64_t D>
    P3_HOST_DEVICE BabyBear injective_exp_root_n() const {
        // This is a simplified inverse for D=5
        // Real implementation would use the proper inverse exponent
        return exp_u64(1725656503);
    }

    // Returns the multiplicative group generator
    P3_HOST_DEVICE static BabyBear generator() {
        return BabyBear(static_cast<uint32_t>(GENERATOR_VAL));
    }

    // Returns a primitive 2^bits-th root of unity.
    // The primitive 2^27-th root is 440564289.
    // For smaller bits k, the generator is g^(2^(27-k)).
    P3_HOST_DEVICE static BabyBear two_adic_generator(size_t bits) {
        // Bounds check: bits must be in [0, TWO_ADICITY]
#if !P3_CUDA_ENABLED
        if (bits > TWO_ADICITY) {
            throw std::invalid_argument("bits exceeds TWO_ADICITY (27) for BabyBear");
        }
#else
        P3_ASSERT(bits <= TWO_ADICITY);
#endif
        // Primitive 2^27-th root of unity
        BabyBear g(static_cast<uint32_t>(ROOT_OF_UNITY_2_POW_27));
        // Raise to 2^(27-bits) to get primitive 2^bits-th root
        uint64_t exp = static_cast<uint64_t>(1) << (TWO_ADICITY - bits);
        return g.exp_u64(exp);
    }

    // Returns a primitive 2^bits-th root of unity as a degree-4 extension field element.
    // For bits <= 27: embeds the base field generator.
    // For bits == 28: [0, 0, 929455875, 0] (i.e., 929455875 * alpha^2).
    // For bits == 29: [0, 0, 0, 1483681942] (i.e., 1483681942 * alpha^3).
    // These are computed such that they are valid generators for the degree-4 extension
    // BinomialExtensionField<BabyBear, 4> with minimal polynomial x^4 - 11.
    P3_HOST_DEVICE static std::array<BabyBear, 4> ext_two_adic_generator(size_t bits) {
#if !P3_CUDA_ENABLED
        if (bits > 29) {
            throw std::invalid_argument("bits exceeds EXT_TWO_ADICITY (29) for BinomialExtensionField<BabyBear,4>");
        }
#else
        P3_ASSERT(bits <= 29);
#endif
        if (bits <= TWO_ADICITY) {
            // Embed base field generator into extension field
            return {two_adic_generator(bits), BabyBear(), BabyBear(), BabyBear()};
        } else if (bits == 28) {
            // Primitive 2^28-th root: 929455875 * alpha^2
            return {BabyBear(), BabyBear(), BabyBear(static_cast<uint32_t>(EXT_ROOT_2_POW_28_COEFF)), BabyBear()};
        } else if (bits == 29) {
            // Primitive 2^29-th root: 1483681942 * alpha^3
            return {BabyBear(), BabyBear(), BabyBear(), BabyBear(static_cast<uint32_t>(EXT_ROOT_2_POW_29_COEFF))};
        }

        // Fallback for unexpected values (especially important on CUDA builds)
        P3_ASSERT(false);
        return {BabyBear(), BabyBear(), BabyBear(), BabyBear()};
    }

    // Powers iterator: yields this^0, this^1, this^2, ...
    // Forward-declare the return type; defined after the class.
#if !P3_CUDA_ENABLED
    struct PowersRange;
    PowersRange powers() const;
#endif

    // Display (CPU only)
#if !P3_CUDA_ENABLED
    friend std::ostream& operator<<(std::ostream& os, const BabyBear& field) {
        os << field.value_;
        return os;
    }
#endif

private:
    // Reduction modulo PRIME
    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint32_t reduce(uint32_t value) {
        if (value >= PRIME) {
            value -= PRIME;
        }
        return value;
    }

    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint32_t reduce64(uint64_t value) {
        // Simple reduction for 64-bit values
        uint32_t result = static_cast<uint32_t>(value % PRIME_U64);
        return result;
    }

    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint32_t from_signed(int32_t value) {
        if (value >= 0) {
            return reduce(static_cast<uint32_t>(value));
        } else {
            // Handle negative: add PRIME until positive
            int64_t v = static_cast<int64_t>(value);
            while (v < 0) v += PRIME;
            return static_cast<uint32_t>(v);
        }
    }

    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint32_t from_signed64(int64_t value) {
        if (value >= 0) {
            return reduce64(static_cast<uint64_t>(value));
        } else {
            // Handle negative
            int64_t v = value % static_cast<int64_t>(PRIME);
            if (v < 0) v += PRIME;
            return static_cast<uint32_t>(v);
        }
    }
};

// Define constants outside class (CPU only)
#if !P3_CUDA_ENABLED
inline const BabyBear BabyBear::ZERO = BabyBear();
inline const BabyBear BabyBear::ONE = BabyBear(static_cast<uint32_t>(1));
inline const BabyBear BabyBear::TWO = BabyBear(static_cast<uint32_t>(2));
inline const BabyBear BabyBear::NEG_ONE = BabyBear(PRIME - 1);

// PowersRange: lazy infinite sequence of successive powers of a BabyBear element.
// Yields: base^0, base^1, base^2, ...
// Use begin() to obtain an Iterator and advance it manually with ++.
// There is intentionally no end() — this prevents accidental use in range-based
// for loops, which would loop forever. Use an explicit counter instead.
struct BabyBear::PowersRange {
    BabyBear base;

    struct Iterator {
        BabyBear current;
        BabyBear base_val;

        explicit Iterator(BabyBear b)
            : current(BabyBear(static_cast<uint32_t>(1u))), base_val(b) {}

        BabyBear operator*() const { return current; }
        Iterator& operator++() { current = current * base_val; return *this; }
        bool operator!=(const Iterator& other) const { return current != other.current; }
    };

    Iterator begin() const { return Iterator(base); }
    // No end(): this is an infinite range. Use begin() + manual counting.
};

inline BabyBear::PowersRange BabyBear::powers() const {
    return PowersRange{*this};
}
#endif

// Device constants for CUDA (must be defined in a .cu file if needed)
#if P3_CUDA_ENABLED
__device__ __constant__ uint32_t BABYBEAR_PRIME = 0x78000001;
#endif

} // namespace p3_field

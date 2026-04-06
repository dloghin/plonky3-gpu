#pragma once

#include "field.hpp"
#include "cuda_compat.hpp"
#include <cstdint>

#if !P3_CUDA_ENABLED
#include <iostream>
#include <stdexcept>
#endif

namespace p3_field {

/**
 * @brief Mersenne-31 prime field: p = 2^31 - 1
 *
 * Prime value: 0x7fffffff = 2147483647
 *
 * This implementation supports both CPU (C++) and GPU (CUDA) execution.
 */
class Mersenne31 : public PrimeField<Mersenne31> {
private:
    uint32_t value_;

public:
    static constexpr uint32_t PRIME = 0x7fffffff; // 2^31 - 1
    static constexpr uint64_t PRIME_U64 = PRIME;

    // Constructors
    P3_HOST_DEVICE P3_CONSTEXPR_HD Mersenne31() : value_(0) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit Mersenne31(uint32_t value) : value_(reduce(value)) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit Mersenne31(uint64_t value) : value_(reduce64(value)) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit Mersenne31(int32_t value) : value_(from_signed(value)) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit Mersenne31(int64_t value) : value_(from_signed64(value)) {}

    // Static factory methods for zero/one (GPU compatible)
    P3_HOST_DEVICE static Mersenne31 zero_val() { return Mersenne31(); }
    P3_HOST_DEVICE static Mersenne31 one_val() { return Mersenne31(static_cast<uint32_t>(1)); }
    P3_HOST_DEVICE static Mersenne31 two_val() { return Mersenne31(static_cast<uint32_t>(2)); }
    P3_HOST_DEVICE static Mersenne31 neg_one_val() { return Mersenne31(PRIME - 1); }

    // Legacy static constants (CPU only, for backward compatibility)
#if !P3_CUDA_ENABLED
    static const Mersenne31 ZERO;
    static const Mersenne31 ONE;
    static const Mersenne31 TWO;
    static const Mersenne31 NEG_ONE;
#endif

    // Access
    P3_HOST_DEVICE P3_CONSTEXPR_HD uint32_t value() const { return value_; }
    P3_HOST_DEVICE P3_CONSTEXPR_HD uint64_t as_canonical_u64() const { return value_; }
    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint64_t modulus() { return PRIME; }

    // Arithmetic operations
    P3_HOST_DEVICE P3_CONSTEXPR_HD Mersenne31 add(const Mersenne31& other) const {
        uint64_t sum = static_cast<uint64_t>(value_) + static_cast<uint64_t>(other.value_);
        return Mersenne31(sum);
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD Mersenne31 sub(const Mersenne31& other) const {
        uint64_t diff = static_cast<uint64_t>(value_) + PRIME - static_cast<uint64_t>(other.value_);
        return Mersenne31(diff);
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD Mersenne31 mul(const Mersenne31& other) const {
        uint64_t prod = static_cast<uint64_t>(value_) * static_cast<uint64_t>(other.value_);
        return Mersenne31(prod);
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD bool equals(const Mersenne31& other) const {
        return value_ == other.value_;
    }

    // Inverse using Fermat's little theorem
    P3_HOST_DEVICE Mersenne31 inv() const {
#if !P3_CUDA_ENABLED
        if (value_ == 0) {
            throw std::runtime_error("Cannot invert zero");
        }
#else
        P3_ASSERT(value_ != 0);
#endif
        return exp_u64(PRIME - 2);
    }

    // Injective power map for D=5 (x^5 is injective for Mersenne-31)
    template<uint64_t D>
    P3_HOST_DEVICE Mersenne31 injective_exp_n() const {
        return exp_const_u64<D>();
    }

    // Display (CPU only)
#if !P3_CUDA_ENABLED
    friend std::ostream& operator<<(std::ostream& os, const Mersenne31& field) {
        os << field.value_;
        return os;
    }
#endif

private:
    // Reduction modulo PRIME (2^31 - 1)
    // Fast reduction using property that 2^31 ≡ 1 (mod 2^31-1)
    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint32_t reduce(uint32_t value) {
        if (value >= PRIME) {
            value -= PRIME;
        }
        return value;
    }

    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint32_t reduce64(uint64_t value) {
        // For Mersenne prime: (a * 2^31 + b) mod (2^31-1) = (a + b) mod (2^31-1)
        uint32_t hi = static_cast<uint32_t>(value >> 31);
        uint32_t lo = static_cast<uint32_t>(value) & PRIME;
        uint32_t sum = lo + hi;
        return reduce(sum);
    }

    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint32_t from_signed(int32_t value) {
        if (value >= 0) {
            return reduce(static_cast<uint32_t>(value));
        } else {
            int64_t v = static_cast<int64_t>(value);
            while (v < 0) v += PRIME;
            return static_cast<uint32_t>(v);
        }
    }

    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint32_t from_signed64(int64_t value) {
        if (value >= 0) {
            return reduce64(static_cast<uint64_t>(value));
        } else {
            int64_t v = value % static_cast<int64_t>(PRIME);
            if (v < 0) v += PRIME;
            return static_cast<uint32_t>(v);
        }
    }
};

// Define constants outside class (CPU only)
#if !P3_CUDA_ENABLED
inline const Mersenne31 Mersenne31::ZERO = Mersenne31();
inline const Mersenne31 Mersenne31::ONE = Mersenne31(static_cast<uint32_t>(1));
inline const Mersenne31 Mersenne31::TWO = Mersenne31(static_cast<uint32_t>(2));
inline const Mersenne31 Mersenne31::NEG_ONE = Mersenne31(PRIME - 1);
#endif

// Device constants for CUDA (must be defined in a .cu file if needed)
#if P3_CUDA_ENABLED
__device__ __constant__ uint32_t MERSENNE31_PRIME = 0x7fffffff;
#endif

} // namespace p3_field

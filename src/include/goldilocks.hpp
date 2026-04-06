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
 * @brief The Goldilocks prime field: p = 2^64 - 2^32 + 1
 *
 * Prime value: 0xffffffff00000001 = 18446744069414584321
 *
 * This implementation supports both CPU (C++) and GPU (CUDA) execution.
 */
class Goldilocks : public PrimeField<Goldilocks> {
private:
    uint64_t value_;

public:
    static constexpr uint64_t PRIME = 0xffffffff00000001ULL; // 2^64 - 2^32 + 1
    static constexpr uint64_t NEG_ORDER = 0xffffffffULL; // 2^32 - 1

    // Constructors
    P3_HOST_DEVICE P3_CONSTEXPR_HD Goldilocks() : value_(0) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit Goldilocks(uint64_t value) : value_(value) {} // Non-canonical OK
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit Goldilocks(int64_t value) : value_(from_signed(value)) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit Goldilocks(int value) : value_(static_cast<uint64_t>(value)) {}
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit Goldilocks(uint32_t value) : value_(static_cast<uint64_t>(value)) {}

    // Static factory methods for zero/one (GPU compatible)
    P3_HOST_DEVICE static Goldilocks zero_val() { return Goldilocks(static_cast<uint64_t>(0)); }
    P3_HOST_DEVICE static Goldilocks one_val() { return Goldilocks(static_cast<uint64_t>(1)); }
    P3_HOST_DEVICE static Goldilocks two_val() { return Goldilocks(static_cast<uint64_t>(2)); }
    P3_HOST_DEVICE static Goldilocks neg_one_val() { return Goldilocks(PRIME - 1); }

    // Legacy static constants (CPU only, for backward compatibility)
#if !P3_CUDA_ENABLED
    static const Goldilocks ZERO;
    static const Goldilocks ONE;
    static const Goldilocks TWO;
    static const Goldilocks NEG_ONE;
#endif

    // Access
    P3_HOST_DEVICE P3_CONSTEXPR_HD uint64_t value() const { return value_; }

    P3_HOST_DEVICE P3_CONSTEXPR_HD uint64_t as_canonical_u64() const {
        uint64_t c = value_;
        if (c >= PRIME) {
            c -= PRIME;
        }
        return c;
    }

    // Arithmetic operations
    P3_HOST_DEVICE P3_CONSTEXPR_HD Goldilocks add(const Goldilocks& other) const {
        uint64_t sum = value_ + other.value_;
        // Check for overflow
        bool over = sum < value_;
        if (over) {
            sum += NEG_ORDER;
        }
        // Check for second overflow (rare)
        if (sum < NEG_ORDER && over) {
            sum += NEG_ORDER;
        }
        return Goldilocks(sum);
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD Goldilocks sub(const Goldilocks& other) const {
        uint64_t diff = value_ - other.value_;
        // Check for underflow
        bool under = diff > value_;
        if (under) {
            diff -= NEG_ORDER;
        }
        // Check for second underflow (rare)
        if (under && diff > (uint64_t)(-(int64_t)NEG_ORDER)) {
            diff -= NEG_ORDER;
        }
        return Goldilocks(diff);
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD Goldilocks mul(const Goldilocks& other) const {
#if P3_CUDA_ENABLED
        // CUDA path: use portable 128-bit multiplication
        cuda_util::uint128_t prod = cuda_util::mul64(value_, other.value_);
        return reduce128_portable(prod.high64(), prod.low64());
#else
        // CPU path: use native 128-bit
        __uint128_t prod = static_cast<__uint128_t>(value_) * static_cast<__uint128_t>(other.value_);
        return reduce128(prod);
#endif
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD bool equals(const Goldilocks& other) const {
        return as_canonical_u64() == other.as_canonical_u64();
    }

    // Inverse using Fermat's little theorem
    P3_HOST_DEVICE Goldilocks inv() const {
#if !P3_CUDA_ENABLED
        if (value_ == 0) {
            throw std::runtime_error("Cannot invert zero");
        }
#else
        P3_ASSERT(value_ != 0);
#endif
        // Use Fermat's little theorem: a^(-1) = a^(p-2)
        return exp_u64(PRIME - 2);
    }

    // Injective power map for D=7 (x^7 is injective for Goldilocks)
    template<uint64_t D>
    P3_HOST_DEVICE Goldilocks injective_exp_n() const {
        return exp_const_u64<D>();
    }

    // Display (CPU only)
#if !P3_CUDA_ENABLED
    friend std::ostream& operator<<(std::ostream& os, const Goldilocks& field) {
        os << field.as_canonical_u64();
        return os;
    }
#endif

private:
#if !P3_CUDA_ENABLED
    // CPU path: native 128-bit reduction
    static constexpr Goldilocks reduce128(__uint128_t x) {
        uint64_t x_lo = static_cast<uint64_t>(x);
        uint64_t x_hi = static_cast<uint64_t>(x >> 64);

        // Split x_hi into high and low 32 bits
        uint64_t x_hi_hi = x_hi >> 32;
        uint64_t x_hi_lo = x_hi & NEG_ORDER;

        // Compute x_lo - x_hi_hi
        uint64_t t0 = x_lo - x_hi_hi;
        bool borrow = t0 > x_lo;
        if (borrow) {
            t0 -= NEG_ORDER;
        }

        // Compute x_hi_lo * NEG_ORDER
        uint64_t t1 = x_hi_lo * NEG_ORDER;

        // Add t0 and t1
        uint64_t t2 = t0 + t1;
        bool carry = t2 < t0;
        if (carry) {
            t2 += NEG_ORDER;
        }

        return Goldilocks(t2);
    }
#endif

    // Portable 128-bit reduction (works on both CPU and GPU)
    P3_HOST_DEVICE static Goldilocks reduce128_portable(uint64_t x_hi, uint64_t x_lo) {
        // Split x_hi into high and low 32 bits
        uint64_t x_hi_hi = x_hi >> 32;
        uint64_t x_hi_lo = x_hi & NEG_ORDER;

        // Compute x_lo - x_hi_hi
        uint64_t t0 = x_lo - x_hi_hi;
        bool borrow = t0 > x_lo;
        if (borrow) {
            t0 -= NEG_ORDER;
        }

        // Compute x_hi_lo * NEG_ORDER
        uint64_t t1 = x_hi_lo * NEG_ORDER;

        // Add t0 and t1
        uint64_t t2 = t0 + t1;
        bool carry = t2 < t0;
        if (carry) {
            t2 += NEG_ORDER;
        }

        return Goldilocks(t2);
    }

    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint64_t from_signed(int64_t value) {
        if (value >= 0) {
            return static_cast<uint64_t>(value);
        } else {
            // Convert negative to field element
            return PRIME + static_cast<uint64_t>(value);
        }
    }
};

// Define constants outside class (CPU only)
#if !P3_CUDA_ENABLED
inline const Goldilocks Goldilocks::ZERO = Goldilocks(static_cast<uint64_t>(0));
inline const Goldilocks Goldilocks::ONE = Goldilocks(static_cast<uint64_t>(1));
inline const Goldilocks Goldilocks::TWO = Goldilocks(static_cast<uint64_t>(2));
inline const Goldilocks Goldilocks::NEG_ONE = Goldilocks(PRIME - 1);
#endif

// Device constants for CUDA (must be defined in a .cu file if needed)
#if P3_CUDA_ENABLED
__device__ __constant__ uint64_t GOLDILOCKS_PRIME = 0xffffffff00000001ULL;
__device__ __constant__ uint64_t GOLDILOCKS_NEG_ORDER = 0xffffffffULL;
#endif

} // namespace p3_field

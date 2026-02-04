#pragma once

#include "cuda_compat.hpp"
#include <cstdint>
#include <cstddef>

#if !P3_CUDA_ENABLED
#include <iostream>
#include <stdexcept>
#endif

namespace p3_field {

/**
 * @brief Base class for prime fields
 *
 * Provides common interface for all prime field implementations.
 * All methods are marked for both host and device execution.
 */
template<typename Derived>
class PrimeField {
public:
    // Static interface (CRTP pattern)
    P3_HOST_DEVICE static Derived zero() { return Derived::zero_val(); }
    P3_HOST_DEVICE static Derived one() { return Derived::one_val(); }

    // Required operations
    P3_HOST_DEVICE Derived operator+(const Derived& other) const {
        return static_cast<const Derived*>(this)->add(other);
    }

    P3_HOST_DEVICE Derived operator-(const Derived& other) const {
        return static_cast<const Derived*>(this)->sub(other);
    }

    P3_HOST_DEVICE Derived operator*(const Derived& other) const {
        return static_cast<const Derived*>(this)->mul(other);
    }

    P3_HOST_DEVICE Derived& operator+=(const Derived& other) {
        *static_cast<Derived*>(this) = *static_cast<const Derived*>(this) + other;
        return *static_cast<Derived*>(this);
    }

    P3_HOST_DEVICE Derived& operator-=(const Derived& other) {
        *static_cast<Derived*>(this) = *static_cast<const Derived*>(this) - other;
        return *static_cast<Derived*>(this);
    }

    P3_HOST_DEVICE Derived& operator*=(const Derived& other) {
        *static_cast<Derived*>(this) = *static_cast<const Derived*>(this) * other;
        return *static_cast<Derived*>(this);
    }

    P3_HOST_DEVICE Derived operator-() const {
        return Derived::zero_val() - *static_cast<const Derived*>(this);
    }

    P3_HOST_DEVICE bool operator==(const Derived& other) const {
        return static_cast<const Derived*>(this)->equals(other);
    }

    P3_HOST_DEVICE bool operator!=(const Derived& other) const {
        return !(*this == other);
    }

    // Common operations
    P3_HOST_DEVICE Derived double_val() const {
        const auto& self = *static_cast<const Derived*>(this);
        return self + self;
    }

    P3_HOST_DEVICE Derived square() const {
        const auto& self = *static_cast<const Derived*>(this);
        return self * self;
    }

    P3_HOST_DEVICE Derived cube() const {
        const auto& self = *static_cast<const Derived*>(this);
        return self.square() * self;
    }

    // Fast exponentiation
    P3_HOST_DEVICE Derived exp_u64(uint64_t power) const {
        if (power == 0) return Derived::one_val();
        if (power == 1) return *static_cast<const Derived*>(this);

        Derived result = Derived::one_val();
        Derived base = *static_cast<const Derived*>(this);

        while (power > 0) {
            if (power & 1) {
                result *= base;
            }
            if (power > 1) {
                base = base.square();
            }
            power >>= 1;
        }
        return result;
    }

    // Constant-time exponentiation for small powers
    template<uint64_t POWER>
    P3_HOST_DEVICE Derived exp_const_u64() const {
        if (POWER == 0) {
            return Derived::one_val();
        } else if (POWER == 1) {
            return *static_cast<const Derived*>(this);
        } else if (POWER == 2) {
            return static_cast<const Derived*>(this)->square();
        } else if (POWER == 3) {
            return static_cast<const Derived*>(this)->cube();
        } else if (POWER == 4) {
            return static_cast<const Derived*>(this)->square().square();
        } else if (POWER == 5) {
            auto sq = static_cast<const Derived*>(this)->square();
            return sq.square() * (*static_cast<const Derived*>(this));
        } else if (POWER == 7) {
            auto sq = static_cast<const Derived*>(this)->square();
            auto cu = sq * (*static_cast<const Derived*>(this));
            auto qu = sq.square();
            return cu * qu;
        } else {
            return static_cast<const Derived*>(this)->exp_u64(POWER);
        }
    }

    // Inverse (must be implemented by derived class)
    P3_HOST_DEVICE Derived inverse() const {
        return static_cast<const Derived*>(this)->inv();
    }

#if !P3_CUDA_ENABLED
    // CPU-only static constants (for backward compatibility)
    static const Derived ZERO;
    static const Derived ONE;
#endif
};

} // namespace p3_field

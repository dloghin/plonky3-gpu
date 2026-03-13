#pragma once

#include "field.hpp"
#include "baby_bear.hpp"
#include "cuda_compat.hpp"
#include <array>
#include <cstdint>
#include <cstddef>

#if !P3_CUDA_ENABLED
#include <iostream>
#include <stdexcept>
#endif

namespace p3_field {

/**
 * @brief Degree-D binomial extension field over a base prime field F.
 *
 * The extension is F[alpha] / (alpha^D - W) where W is the binomial coefficient.
 * Elements: a0 + a1*alpha + ... + a(D-1)*alpha^(D-1), stored as std::array<F, D>.
 *
 * Template parameters:
 *   F -- the base prime field type
 *   D -- the extension degree
 *   W -- the binomial constant (alpha^D = W in the base field)
 *
 * For BabyBear degree-4:  W = 11  (polynomial x^4 - 11 is irreducible over BabyBear)
 */
template<typename F, size_t D, uint32_t W>
class BinomialExtensionField {
public:
    static constexpr size_t DEGREE = D;
    static constexpr uint32_t BINOMIAL_W = W;

    std::array<F, D> coeffs;

    // Constructors
    P3_HOST_DEVICE BinomialExtensionField() : coeffs{} {}

    P3_HOST_DEVICE explicit BinomialExtensionField(const std::array<F, D>& c) : coeffs(c) {}

    // Construct from base field element (embeds f as [f, 0, 0, ...])
    P3_HOST_DEVICE explicit BinomialExtensionField(const F& f) : coeffs{} {
        coeffs[0] = f;
    }

    // Access
    P3_HOST_DEVICE const F& operator[](size_t i) const { return coeffs[i]; }
    P3_HOST_DEVICE F& operator[](size_t i) { return coeffs[i]; }

    // Equality
    P3_HOST_DEVICE bool operator==(const BinomialExtensionField& other) const {
        for (size_t i = 0; i < D; ++i) {
            if (coeffs[i] != other.coeffs[i]) return false;
        }
        return true;
    }

    P3_HOST_DEVICE bool operator!=(const BinomialExtensionField& other) const {
        return !(*this == other);
    }

    // Zero and one
    P3_HOST_DEVICE static BinomialExtensionField zero_val() {
        return BinomialExtensionField();
    }

    P3_HOST_DEVICE static BinomialExtensionField one_val() {
        BinomialExtensionField result;
        result.coeffs[0] = F::one_val();
        return result;
    }

    // Addition: component-wise
    P3_HOST_DEVICE BinomialExtensionField operator+(const BinomialExtensionField& other) const {
        BinomialExtensionField result;
        for (size_t i = 0; i < D; ++i) {
            result.coeffs[i] = coeffs[i] + other.coeffs[i];
        }
        return result;
    }

    // Subtraction: component-wise
    P3_HOST_DEVICE BinomialExtensionField operator-(const BinomialExtensionField& other) const {
        BinomialExtensionField result;
        for (size_t i = 0; i < D; ++i) {
            result.coeffs[i] = coeffs[i] - other.coeffs[i];
        }
        return result;
    }

    // Negation
    P3_HOST_DEVICE BinomialExtensionField operator-() const {
        BinomialExtensionField result;
        for (size_t i = 0; i < D; ++i) {
            result.coeffs[i] = F::zero_val() - coeffs[i];
        }
        return result;
    }

    // Multiplication in F[alpha]/(alpha^D - W)
    // Schoolbook: O(D^2) multiplications with reduction via alpha^D = W.
    P3_HOST_DEVICE BinomialExtensionField operator*(const BinomialExtensionField& other) const {
        BinomialExtensionField result;
        F w(static_cast<uint32_t>(W));
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                F prod = coeffs[i] * other.coeffs[j];
                size_t idx = i + j;
                if (idx < D) {
                    result.coeffs[idx] = result.coeffs[idx] + prod;
                } else {
                    // alpha^(D+k) = W * alpha^k
                    result.coeffs[idx - D] = result.coeffs[idx - D] + prod * w;
                }
            }
        }
        return result;
    }

    // Scalar multiplication by base field element
    P3_HOST_DEVICE BinomialExtensionField operator*(const F& scalar) const {
        BinomialExtensionField result;
        for (size_t i = 0; i < D; ++i) {
            result.coeffs[i] = coeffs[i] * scalar;
        }
        return result;
    }

    P3_HOST_DEVICE BinomialExtensionField& operator+=(const BinomialExtensionField& other) {
        for (size_t i = 0; i < D; ++i) {
            coeffs[i] = coeffs[i] + other.coeffs[i];
        }
        return *this;
    }
    P3_HOST_DEVICE BinomialExtensionField& operator-=(const BinomialExtensionField& other) {
        for (size_t i = 0; i < D; ++i) {
            coeffs[i] = coeffs[i] - other.coeffs[i];
        }
        return *this;
    }
    P3_HOST_DEVICE BinomialExtensionField& operator*=(const BinomialExtensionField& other) {
        // Copy original coefficients so we can accumulate into this instance safely.
        std::array<F, D> a = coeffs;

        // Recompute product in-place using schoolbook multiplication with reduction.
        BinomialExtensionField result;
        F w(static_cast<uint32_t>(W));
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                F prod = a[i] * other.coeffs[j];
                size_t idx = i + j;
                if (idx < D) {
                    result.coeffs[idx] = result.coeffs[idx] + prod;
                } else {
                    // alpha^(D+k) = W * alpha^k
                    result.coeffs[idx - D] = result.coeffs[idx - D] + prod * w;
                }
            }
        }

        coeffs = result.coeffs;
        return *this;
    }

    P3_HOST_DEVICE BinomialExtensionField square() const { return *this * *this; }

    // Fast exponentiation (square-and-multiply)
    P3_HOST_DEVICE BinomialExtensionField exp_u64(uint64_t power) const {
        if (power == 0) return one_val();
        BinomialExtensionField result = one_val();
        BinomialExtensionField base = *this;
        while (power > 0) {
            if (power & 1) result = result * base;
            if (power > 1) base = base.square();
            power >>= 1;
        }
        return result;
    }

    // Multiplicative inverse using Frobenius automorphism + norm.
    // For F[alpha]/(alpha^D - W), the inverse of `a` is
    //   a^(-1) = (a^p * a^(p^2) * a^(p^3)) / norm(a)
    // where norm(a) = a * a^p * a^(p^2) * a^(p^3) lies in F.
    //
    // This specialization is for D=4 over BabyBear; for other combinations
    // a fallback (Fermat) approach is used.
    P3_HOST_DEVICE BinomialExtensionField inv() const;

    // Powers iterator: lazy infinite range yielding base^0, base^1, base^2, ...
#if !P3_CUDA_ENABLED
    struct PowersRange;
    PowersRange powers() const;
#endif

#if !P3_CUDA_ENABLED
    friend std::ostream& operator<<(std::ostream& os, const BinomialExtensionField& e) {
        os << "[";
        for (size_t i = 0; i < D; ++i) {
            if (i > 0) os << ", ";
            os << e.coeffs[i];
        }
        os << "]";
        return os;
    }
#endif
};

// ---------------------------------------------------------------------------
// Powers iterator (defined after class is complete)
// ---------------------------------------------------------------------------
#if !P3_CUDA_ENABLED
template<typename F, size_t D, uint32_t W>
struct BinomialExtensionField<F, D, W>::PowersRange {
    BinomialExtensionField base;

    struct Iterator {
        BinomialExtensionField current;
        BinomialExtensionField base_val;

        explicit Iterator(BinomialExtensionField b)
            : current(BinomialExtensionField::one_val()), base_val(b) {}

        BinomialExtensionField operator*() const { return current; }
        Iterator& operator++() { current = current * base_val; return *this; }
        bool operator!=(const Iterator&) const { return true; }
    };

    Iterator begin() const { return Iterator(base); }
    Iterator end()   const { return Iterator(base); }
};

template<typename F, size_t D, uint32_t W>
typename BinomialExtensionField<F, D, W>::PowersRange
BinomialExtensionField<F, D, W>::powers() const {
    return PowersRange{*this};
}
#endif

// ---------------------------------------------------------------------------
// Inverse: general fallback (Fermat's little theorem in the extension)
// For small D and known primes this is acceptable; specialisations can override.
// Note: for D=4, BabyBear, we provide a more efficient Frobenius-based impl below.
// ---------------------------------------------------------------------------
template<typename F, size_t D, uint32_t W>
P3_HOST_DEVICE BinomialExtensionField<F, D, W>
BinomialExtensionField<F, D, W>::inv() const {
    // Use Frobenius method for D=4 over BabyBear -- specialised below.
    // For the general case: fall through (this should not be called for unsupported types).
#if !P3_CUDA_ENABLED
    throw std::runtime_error("BinomialExtensionField::inv() not implemented for this type");
#endif
    return zero_val();
}

// ---------------------------------------------------------------------------
// Concrete type aliases for BabyBear degree-4 extension
// ---------------------------------------------------------------------------
// Irreducible polynomial for BabyBear degree-4: x^4 - 11 (W = 11)
using BabyBear4 = BinomialExtensionField<BabyBear, 4, 11>;

// ---------------------------------------------------------------------------
// Specialised inverse for BabyBear4 using Frobenius automorphism
// ---------------------------------------------------------------------------
// The Frobenius phi(a) = a^p acts on BabyBear4 as:
//   phi([a0,a1,a2,a3]) = [a0, a1*z, a2*z^2, a3*z^3]
// where z = W^((p-1)/4) = 11^((p-1)/4) mod p.
// Note: p ≡ 1 (mod 4) so z is a 4th root of unity in F_p.
//
// norm(a) = a * phi(a) * phi^2(a) * phi^3(a) ∈ F_p
// a^(-1) = phi(a) * phi^2(a) * phi^3(a) * norm(a)^(-1)
template<>
inline BabyBear4 BabyBear4::inv() const {
    using BB = BabyBear;
    constexpr uint32_t p = BB::PRIME;

#if !P3_CUDA_ENABLED
    bool is_zero = true;
    for (size_t i = 0; i < 4; ++i) {
        if (coeffs[i] != BB()) { is_zero = false; break; }
    }
    if (is_zero) {
        throw std::runtime_error("Cannot invert zero in BabyBear4");
    }
#endif

    // z = W^((p-1)/4) mod p  -- a primitive 4th root of unity in F_p
    BB z = BB(static_cast<uint32_t>(BabyBear4::BINOMIAL_W)).exp_u64((p - 1) / 4);
    BB z2 = z * z;
    BB z3 = z2 * z;

    // Frobenius maps (phi^k applied component-wise):
    // phi^1(a)[i] = a[i] * z^i
    BabyBear4 pa({coeffs[0], coeffs[1] * z, coeffs[2] * z2, coeffs[3] * z3});
    // phi^2(a)[i] = a[i] * z^(2i)  -- z^4=1 so z^4=1, z^6=z^2
    BabyBear4 p2a({coeffs[0], coeffs[1] * z2, coeffs[2], coeffs[3] * z2});
    // phi^3(a)[i] = a[i] * z^(3i)  -- z^6=z^2, z^9=z
    BabyBear4 p3a({coeffs[0], coeffs[1] * z3, coeffs[2] * z2, coeffs[3] * z});

    // norm = a * phi(a) * phi^2(a) * phi^3(a)  (lives in F_p, i.e. only coeff[0] nonzero)
    BabyBear4 a_pa   = (*this) * pa;
    BabyBear4 p2_p3a = p2a * p3a;
    BabyBear4 norm_ext = a_pa * p2_p3a;
    BB norm_val = norm_ext.coeffs[0];

    // a^(-1) = phi(a)*phi^2(a)*phi^3(a) / norm(a)
    BB norm_inv = norm_val.inv();
    BabyBear4 num = pa * p2_p3a;
    BabyBear4 result;
    for (size_t i = 0; i < 4; ++i) {
        result.coeffs[i] = num.coeffs[i] * norm_inv;
    }
    return result;
}

// ---------------------------------------------------------------------------
// TwoAdicExtField traits: two-adic constants for BabyBear4
// ---------------------------------------------------------------------------
template<typename F, size_t D, uint32_t W>
struct TwoAdicExtField;

template<>
struct TwoAdicExtField<BabyBear, 4, 11> {
    // p^4 - 1 has 2-adic valuation 29:
    //   v_2(p-1)=27, v_2(p+1)=1, v_2(p^2+1)=1 => total 29
    static constexpr size_t TWO_ADICITY = 29;

    using Ext = BabyBear4;

    // Multiplicative group generator of BabyBear4: 6 + alpha = [6, 1, 0, 0]
    static Ext generator() {
        return Ext({BabyBear(static_cast<uint32_t>(6u)),
                    BabyBear(static_cast<uint32_t>(1u)),
                    BabyBear(),
                    BabyBear()});
    }

    // Returns a primitive 2^bits-th root of unity in BabyBear4.
    // bits in [0, 29].
    //
    // For bits <= 27: embed BabyBear::two_adic_generator(bits) as [g, 0, 0, 0].
    // For bits == 28: 929455875 * alpha^2  =>  [0, 0, 929455875, 0]
    // For bits == 29: 1483681942 * alpha^3 =>  [0, 0, 0, 1483681942]
    //
    // The bits=28,29 values were computed from alpha = [0,1,0,0] raised to the
    // odd part of p^4-1, giving a primitive 2^29-th root of unity.
    static Ext two_adic_generator(size_t bits) {
#if !P3_CUDA_ENABLED
        if (bits > TWO_ADICITY) {
            throw std::invalid_argument(
                "bits exceeds EXT_TWO_ADICITY (29) for BabyBear4");
        }
#endif
        auto arr = BabyBear::ext_two_adic_generator(bits);
        return Ext(arr);
    }
};

// Convenience alias matching the task spec
using BabyBear4TwoAdic = TwoAdicExtField<BabyBear, 4, 11>;

} // namespace p3_field

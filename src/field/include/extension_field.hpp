#pragma once

/**
 * @file extension_field.hpp
 * @brief BinomialExtensionField<F, D> -- degree-D extension of base field F
 *
 * Represents elements of F[X]/(X^D - W) where W is a field-specific constant.
 * Ported from plonky3/field/src/extension/binomial_extension.rs
 *
 * Primary use case: BinomialExtensionField<BabyBear, 4> (quartic extension, W=11)
 */

#include "field.hpp"
#include "cuda_compat.hpp"
#include <array>
#include <cstdint>
#include <cstddef>

#if !P3_CUDA_ENABLED
#include <iostream>
#include <stdexcept>
#include <vector>
#endif

namespace p3_field {

// Forward declarations
class BabyBear;

// ---------------------------------------------------------------------------
// Traits: constants for BinomialExtensionField<F, D>
// Specialize this for each (F, D) pair used.
// ---------------------------------------------------------------------------

template<typename F, size_t D>
struct BinomialExtensionTraits;

// Specialization for BabyBear quartic (D=4)
// Irreducible polynomial: x^4 - 11
// DTH_ROOT = W^((p-1)/D) mod p = 11^((2013265920)/4) mod 2013265921 = 1728404513
//   Satisfies: DTH_ROOT^2 = p-1 = -1,  DTH_ROOT^4 = 1  (primitive 4th root of unity)
// EXT_GENERATOR = [6, 1, 0, 0]
// EXT_TWO_ADICITY = 29
template<>
struct BinomialExtensionTraits<BabyBear, 4> {
    static constexpr uint32_t W_VALUE = 11;
    static constexpr uint32_t DTH_ROOT_VALUE = 1728404513;
    static constexpr uint32_t EXT_GENERATOR_0 = 6;
    static constexpr uint32_t EXT_GENERATOR_1 = 1;
    static constexpr uint32_t EXT_GENERATOR_2 = 0;
    static constexpr uint32_t EXT_GENERATOR_3 = 0;
    static constexpr uint32_t EXT_TWO_ADICITY = 29;
};

// ---------------------------------------------------------------------------
// BinomialExtensionField<F, D>
// ---------------------------------------------------------------------------

template<typename F, size_t D>
class BinomialExtensionField {
public:
    std::array<F, D> value;

    using BaseField = F;
    using Traits = BinomialExtensionTraits<F, D>;
    static constexpr size_t DEGREE = D;

    // -------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------

    P3_HOST_DEVICE BinomialExtensionField() {
        for (size_t i = 0; i < D; ++i) value[i] = F::zero();
    }

    P3_HOST_DEVICE explicit BinomialExtensionField(const std::array<F, D>& coeffs)
        : value(coeffs) {}

    // Embed a base-field element: (f, 0, 0, ..., 0)
    P3_HOST_DEVICE static BinomialExtensionField from_base(const F& f) {
        BinomialExtensionField result;
        result.value[0] = f;
        return result;
    }

    // -------------------------------------------------------------------
    // Field constants
    // -------------------------------------------------------------------

    P3_HOST_DEVICE static BinomialExtensionField zero_val() {
        return BinomialExtensionField{};
    }

    P3_HOST_DEVICE static BinomialExtensionField one_val() {
        BinomialExtensionField result;
        result.value[0] = F::one();
        return result;
    }

    P3_HOST_DEVICE static BinomialExtensionField two_val() {
        BinomialExtensionField result;
        result.value[0] = F::one() + F::one();
        return result;
    }

    P3_HOST_DEVICE static BinomialExtensionField neg_one_val() {
        BinomialExtensionField result;
        result.value[0] = -F::one();
        return result;
    }

#if !P3_CUDA_ENABLED
    static const BinomialExtensionField ZERO;
    static const BinomialExtensionField ONE;
    static const BinomialExtensionField TWO;
    static const BinomialExtensionField NEG_ONE;
    static const BinomialExtensionField GENERATOR;
#endif

    // W: the constant in x^D - W
    P3_HOST_DEVICE static F W() {
        return F(static_cast<uint32_t>(Traits::W_VALUE));
    }

    // DTH_ROOT: used in Frobenius automorphism
    P3_HOST_DEVICE static F DTH_ROOT() {
        return F(static_cast<uint32_t>(Traits::DTH_ROOT_VALUE));
    }

    // -------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------

    P3_HOST_DEVICE const std::array<F, D>& as_basis_coefficients() const {
        return value;
    }

    P3_HOST_DEVICE bool is_in_basefield() const {
        for (size_t i = 1; i < D; ++i) {
            if (value[i] != F::zero()) return false;
        }
        return true;
    }

    P3_HOST_DEVICE bool equals(const BinomialExtensionField& other) const {
        for (size_t i = 0; i < D; ++i) {
            if (value[i] != other.value[i]) return false;
        }
        return true;
    }

    // -------------------------------------------------------------------
    // Addition / Subtraction
    // -------------------------------------------------------------------

    P3_HOST_DEVICE BinomialExtensionField add(const BinomialExtensionField& other) const {
        BinomialExtensionField result;
        for (size_t i = 0; i < D; ++i) result.value[i] = value[i] + other.value[i];
        return result;
    }

    P3_HOST_DEVICE BinomialExtensionField sub(const BinomialExtensionField& other) const {
        BinomialExtensionField result;
        for (size_t i = 0; i < D; ++i) result.value[i] = value[i] - other.value[i];
        return result;
    }

    P3_HOST_DEVICE BinomialExtensionField neg() const {
        BinomialExtensionField result;
        for (size_t i = 0; i < D; ++i) result.value[i] = -value[i];
        return result;
    }

    P3_HOST_DEVICE BinomialExtensionField operator+(const BinomialExtensionField& other) const {
        return add(other);
    }
    P3_HOST_DEVICE BinomialExtensionField operator-(const BinomialExtensionField& other) const {
        return sub(other);
    }
    P3_HOST_DEVICE BinomialExtensionField operator-() const { return neg(); }

    P3_HOST_DEVICE BinomialExtensionField& operator+=(const BinomialExtensionField& other) {
        *this = add(other); return *this;
    }
    P3_HOST_DEVICE BinomialExtensionField& operator-=(const BinomialExtensionField& other) {
        *this = sub(other); return *this;
    }

    P3_HOST_DEVICE bool operator==(const BinomialExtensionField& other) const {
        return equals(other);
    }
    P3_HOST_DEVICE bool operator!=(const BinomialExtensionField& other) const {
        return !equals(other);
    }

    // -------------------------------------------------------------------
    // Scalar multiplication (base field element)
    // -------------------------------------------------------------------

    P3_HOST_DEVICE BinomialExtensionField scalar_mul(const F& scalar) const {
        BinomialExtensionField result;
        for (size_t i = 0; i < D; ++i) result.value[i] = value[i] * scalar;
        return result;
    }

    P3_HOST_DEVICE BinomialExtensionField operator*(const F& scalar) const {
        return scalar_mul(scalar);
    }

    // -------------------------------------------------------------------
    // Multiplication (extension field)
    // -------------------------------------------------------------------

    P3_HOST_DEVICE BinomialExtensionField mul(const BinomialExtensionField& other) const {
        if constexpr (D == 4) {
            return quartic_mul(other);
        } else {
            return general_mul(other);
        }
    }

    P3_HOST_DEVICE BinomialExtensionField operator*(const BinomialExtensionField& other) const {
        return mul(other);
    }
    P3_HOST_DEVICE BinomialExtensionField& operator*=(const BinomialExtensionField& other) {
        *this = mul(other); return *this;
    }

    // -------------------------------------------------------------------
    // Squaring
    // -------------------------------------------------------------------

    P3_HOST_DEVICE BinomialExtensionField square() const {
        if constexpr (D == 4) {
            return quartic_square();
        } else {
            return mul(*this);
        }
    }

    // -------------------------------------------------------------------
    // Halving: component-wise x/2
    // -------------------------------------------------------------------

    P3_HOST_DEVICE BinomialExtensionField halve() const {
        F two_inv = (F::one() + F::one()).inverse();
        BinomialExtensionField result;
        for (size_t i = 0; i < D; ++i) result.value[i] = value[i] * two_inv;
        return result;
    }

    // -------------------------------------------------------------------
    // Frobenius automorphism: x -> x^p
    // frobenius(a0, a1, ...) = (a0, a1*DTH_ROOT, a2*DTH_ROOT^2, ...)
    // -------------------------------------------------------------------

    P3_HOST_DEVICE BinomialExtensionField frobenius() const {
        BinomialExtensionField result;
        F dth = DTH_ROOT();
        F power = F::one();
        for (size_t i = 0; i < D; ++i) {
            result.value[i] = value[i] * power;
            power = power * dth;
        }
        return result;
    }

    P3_HOST_DEVICE BinomialExtensionField repeated_frobenius(size_t count) const {
        BinomialExtensionField result = *this;
        for (size_t i = 0; i < count; ++i) {
            result = result.frobenius();
        }
        return result;
    }

    // -------------------------------------------------------------------
    // Inverse and division
    // -------------------------------------------------------------------

    P3_HOST_DEVICE BinomialExtensionField inv() const {
        if constexpr (D == 4) {
            return quartic_inv();
        } else {
            return pseudo_inv();
        }
    }

    P3_HOST_DEVICE BinomialExtensionField inverse() const { return inv(); }

    P3_HOST_DEVICE BinomialExtensionField operator/(const BinomialExtensionField& other) const {
        return mul(other.inv());
    }
    P3_HOST_DEVICE BinomialExtensionField& operator/=(const BinomialExtensionField& other) {
        *this = mul(other.inv()); return *this;
    }

    // -------------------------------------------------------------------
    // Exponentiation
    // -------------------------------------------------------------------

    P3_HOST_DEVICE BinomialExtensionField exp_u64(uint64_t power) const {
        if (power == 0) return one_val();
        BinomialExtensionField result = one_val();
        BinomialExtensionField base = *this;
        while (power > 0) {
            if (power & 1) result = result.mul(base);
            if (power > 1) base = base.square();
            power >>= 1;
        }
        return result;
    }

    P3_HOST_DEVICE BinomialExtensionField exp_power_of_2(uint32_t power_log) const {
        BinomialExtensionField result = *this;
        for (uint32_t i = 0; i < power_log; ++i) {
            result = result.square();
        }
        return result;
    }

#if !P3_CUDA_ENABLED
    // Returns vector of powers: [1, x, x^2, ..., x^(n-1)]
    std::vector<BinomialExtensionField> powers(size_t n) const {
        std::vector<BinomialExtensionField> result;
        result.reserve(n);
        BinomialExtensionField current = one_val();
        for (size_t i = 0; i < n; ++i) {
            result.push_back(current);
            current = current.mul(*this);
        }
        return result;
    }
#endif

#if !P3_CUDA_ENABLED
    friend std::ostream& operator<<(std::ostream& os, const BinomialExtensionField& ext) {
        os << "(";
        for (size_t i = 0; i < D; ++i) {
            if (i > 0) os << ", ";
            os << ext.value[i];
        }
        os << ")";
        return os;
    }
#endif

private:
    // -------------------------------------------------------------------
    // Optimized quartic multiplication (D=4)
    // Computes (a0+a1*X+a2*X^2+a3*X^3) * (b0+b1*X+b2*X^2+b3*X^3) mod (X^4 - W)
    //
    // c0 = a0*b0 + W*(a1*b3 + a2*b2 + a3*b1)
    // c1 = a0*b1 + a1*b0 + W*(a2*b3 + a3*b2)
    // c2 = a0*b2 + a1*b1 + a2*b0 + W*(a3*b3)
    // c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0
    // -------------------------------------------------------------------
    P3_HOST_DEVICE BinomialExtensionField quartic_mul(const BinomialExtensionField& b) const {
        const F& a0 = value[0]; const F& a1 = value[1];
        const F& a2 = value[2]; const F& a3 = value[3];
        const F& b0 = b.value[0]; const F& b1 = b.value[1];
        const F& b2 = b.value[2]; const F& b3 = b.value[3];
        const F w = W();

        std::array<F, 4> c;
        c[0] = a0*b0 + w*(a1*b3 + a2*b2 + a3*b1);
        c[1] = a0*b1 + a1*b0 + w*(a2*b3 + a3*b2);
        c[2] = a0*b2 + a1*b1 + a2*b0 + w*(a3*b3);
        c[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0;
        return BinomialExtensionField(c);
    }

    // -------------------------------------------------------------------
    // Optimized quartic squaring (D=4)
    // c0 = a0^2 + W*(2*a1*a3 + a2^2)
    // c1 = 2*(a0*a1 + W*a2*a3)
    // c2 = 2*a0*a2 + a1^2 + W*a3^2
    // c3 = 2*(a0*a3 + a1*a2)
    // -------------------------------------------------------------------
    P3_HOST_DEVICE BinomialExtensionField quartic_square() const {
        const F& a0 = value[0]; const F& a1 = value[1];
        const F& a2 = value[2]; const F& a3 = value[3];
        const F w = W();
        const F two = F::one() + F::one();

        std::array<F, 4> c;
        c[0] = a0*a0 + w*(two*a1*a3 + a2*a2);
        c[1] = two*(a0*a1 + w*a2*a3);
        c[2] = two*a0*a2 + a1*a1 + w*a3*a3;
        c[3] = two*(a0*a3 + a1*a2);
        return BinomialExtensionField(c);
    }

    // -------------------------------------------------------------------
    // General schoolbook multiplication (any D)
    // c[k] = sum_{i+j=k} a[i]*b[j]  +  W * sum_{i+j=k+D} a[i]*b[j]
    // -------------------------------------------------------------------
    P3_HOST_DEVICE BinomialExtensionField general_mul(const BinomialExtensionField& b) const {
        const F w = W();
        std::array<F, D> c;
        for (size_t i = 0; i < D; ++i) c[i] = F::zero();

        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < D; ++j) {
                size_t k = i + j;
                F term = value[i] * b.value[j];
                if (k < D) {
                    c[k] = c[k] + term;
                } else {
                    c[k - D] = c[k - D] + w * term;
                }
            }
        }
        return BinomialExtensionField(c);
    }

    // -------------------------------------------------------------------
    // Quartic inverse using tower decomposition:
    //   Write element as A + X*B where:
    //     A = (a0, a2) in F[Y]/(Y^2 - W)  (Y = X^2)
    //     B = (a1, a3) in F[Y]/(Y^2 - W)
    //
    //   Norm: N = A^2 - Y*B^2  (a quadratic element)
    //   Quadratic inverse: N^{-1} = (n0 - n1*Y) / (n0^2 - W*n1^2)
    //   Quartic inverse: (A - X*B) * N^{-1}
    // -------------------------------------------------------------------
    P3_HOST_DEVICE BinomialExtensionField quartic_inv() const {
        const F& a0 = value[0]; const F& a1 = value[1];
        const F& a2 = value[2]; const F& a3 = value[3];
        const F w = W();
        const F two = F::one() + F::one();

        // A^2 in the quadratic F[Y]/(Y^2 - W):
        //   A = a0 + a2*Y
        //   A^2 = (a0^2 + W*a2^2) + 2*a0*a2*Y
        F a_sq_0 = a0*a0 + w*a2*a2;
        F a_sq_1 = two*a0*a2;

        // Y*B^2 where B = a1 + a3*Y:
        //   B^2 = (a1^2 + W*a3^2) + 2*a1*a3*Y
        //   Y*B^2 = 2*W*a1*a3 + (a1^2 + W*a3^2)*Y
        F yb_sq_0 = two*w*a1*a3;
        F yb_sq_1 = a1*a1 + w*a3*a3;

        // N = A^2 - Y*B^2
        F n0 = a_sq_0 - yb_sq_0;
        F n1 = a_sq_1 - yb_sq_1;

        // N_det = n0^2 - W*n1^2  (scalar in base field)
        F n_det = n0*n0 - w*n1*n1;
        F n_det_inv = n_det.inverse();

        // N^{-1} = (n0*n_det_inv, -n1*n_det_inv)
        F ni0 = n0 * n_det_inv;
        F ni1 = (-n1) * n_det_inv;

        // Quartic inverse = (a0, -a1, a2, -a3) * (ni0, 0, ni1, 0)
        // Expanding via quartic mul formula with b = (ni0, 0, ni1, 0):
        //   c0 = a0*ni0 + W*( -a1*0 + a2*ni1 + -a3*0) = a0*ni0 + W*a2*ni1
        //   c1 = a0*0   + -a1*ni0 + W*(a2*0 + -a3*ni1) = -a1*ni0 - W*a3*ni1
        //   c2 = a0*ni1 + -a1*0   + a2*ni0 + W*(-a3*0) = a0*ni1 + a2*ni0
        //   c3 = a0*0   + -a1*ni1 + a2*0   + -a3*ni0   = -a1*ni1 - a3*ni0
        std::array<F, 4> c;
        c[0] = a0*ni0 + w*a2*ni1;
        c[1] = (-a1)*ni0 + w*(-a3)*ni1;
        c[2] = a0*ni1 + a2*ni0;
        c[3] = (-a1)*ni1 + (-a3)*ni0;
        return BinomialExtensionField(c);
    }

    // -------------------------------------------------------------------
    // Pseudo-inverse for general D using Frobenius automorphism:
    //   norm = product of all Frobenius conjugates
    //   pseudo_inv = (product of all conjugates except self) / norm
    //
    // norm = x * frob(x) * frob^2(x) * ... * frob^{D-1}(x) is in base field
    // -------------------------------------------------------------------
    P3_HOST_DEVICE BinomialExtensionField pseudo_inv() const {
        // Compute all conjugates
        BinomialExtensionField conj[D];
        conj[0] = *this;
        for (size_t i = 1; i < D; ++i) {
            conj[i] = conj[i-1].frobenius();
        }

        // Compute norm (product of all conjugates) -- lands in base field
        BinomialExtensionField norm_ext = one_val();
        for (size_t i = 0; i < D; ++i) {
            norm_ext = norm_ext.mul(conj[i]);
        }
        // norm_ext.value[0] is the base-field norm; higher coefficients should be zero
        F norm = norm_ext.value[0];
        F norm_inv = norm.inverse();

        // Inverse = (conj[1] * conj[2] * ... * conj[D-1]) * norm_inv
        BinomialExtensionField result = one_val();
        for (size_t i = 1; i < D; ++i) {
            result = result.mul(conj[i]);
        }
        return result.scalar_mul(norm_inv);
    }
};

// ---------------------------------------------------------------------------
// CPU-only static constant definitions
// ---------------------------------------------------------------------------

#if !P3_CUDA_ENABLED

template<typename F, size_t D>
const BinomialExtensionField<F, D> BinomialExtensionField<F, D>::ZERO =
    BinomialExtensionField<F, D>::zero_val();

template<typename F, size_t D>
const BinomialExtensionField<F, D> BinomialExtensionField<F, D>::ONE =
    BinomialExtensionField<F, D>::one_val();

template<typename F, size_t D>
const BinomialExtensionField<F, D> BinomialExtensionField<F, D>::TWO =
    BinomialExtensionField<F, D>::two_val();

template<typename F, size_t D>
const BinomialExtensionField<F, D> BinomialExtensionField<F, D>::NEG_ONE =
    BinomialExtensionField<F, D>::neg_one_val();

// GENERATOR is specialized per (F, D) -- defined below after BabyBear specialization

#endif // !P3_CUDA_ENABLED

// ---------------------------------------------------------------------------
// BabyBear quartic: GENERATOR specialization
// EXT_GENERATOR = [6, 1, 0, 0]
// ---------------------------------------------------------------------------

#if !P3_CUDA_ENABLED

// We use an explicit specialization helper to avoid needing to include baby_bear.hpp
// The actual BabyBear-specific generator is computed at runtime from the trait constants.

// Provide a free function for the BabyBear quartic generator
template<typename F, size_t D>
inline BinomialExtensionField<F, D> make_ext_generator() {
    BinomialExtensionField<F, D> gen;
    // Generic: use first coefficient = W constant (not a true generator)
    gen.value[0] = F(static_cast<uint32_t>(BinomialExtensionTraits<F, D>::EXT_GENERATOR_0));
    gen.value[1] = F(static_cast<uint32_t>(BinomialExtensionTraits<F, D>::EXT_GENERATOR_1));
    if constexpr (D > 2) {
        gen.value[2] = F(static_cast<uint32_t>(BinomialExtensionTraits<F, D>::EXT_GENERATOR_2));
    }
    if constexpr (D > 3) {
        gen.value[3] = F(static_cast<uint32_t>(BinomialExtensionTraits<F, D>::EXT_GENERATOR_3));
    }
    return gen;
}

template<typename F, size_t D>
const BinomialExtensionField<F, D> BinomialExtensionField<F, D>::GENERATOR =
    make_ext_generator<F, D>();

#endif // !P3_CUDA_ENABLED

// ---------------------------------------------------------------------------
// Convenience type alias
// ---------------------------------------------------------------------------

// Include baby_bear.hpp AFTER the template so specializations see BabyBear's definition.
// Users of this header should include extension_field.hpp (which pulls in baby_bear.hpp).

} // namespace p3_field

// Include baby_bear.hpp so that BinomialExtensionField<BabyBear, 4> can be instantiated.
// This must come after the namespace definition above.
#include "baby_bear.hpp"

namespace p3_field {

// Convenience alias for the primary use case
using BabyBearExt4 = BinomialExtensionField<BabyBear, 4>;

} // namespace p3_field

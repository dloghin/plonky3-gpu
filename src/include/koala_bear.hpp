#pragma once

/**
 * KoalaBear prime field: p = 2^31 - 2^24 + 1 = 0x7f000001 (Plonky3 / p3-koala-bear).
 * Elements are stored in Montgomery form (R = 2^32 mod p), matching p3-monty-31.
 */

#include "field.hpp"
#include "cuda_compat.hpp"
#include <array>
#include <cstdint>

#if !P3_CUDA_ENABLED
#include <iostream>
#include <stdexcept>
#endif

namespace p3_field {

namespace koala_bear_detail {

P3_HOST_DEVICE P3_CONSTEXPR_HD inline uint32_t monty_reduce_u64(uint64_t x) {
    constexpr uint32_t PRIME = 0x7f000001u;
    constexpr uint32_t MU    = 0x81000001u;
    constexpr uint32_t MONTY_BITS = 32;
    constexpr uint64_t MASK64 = 0xFFFFFFFFu;
    
    uint64_t t = (x * static_cast<uint64_t>(MU)) & MASK64;
    uint64_t u = t * static_cast<uint64_t>(PRIME);

    const bool over = (x < u);
    const uint64_t x_sub_u = x - u;

    uint32_t hi = static_cast<uint32_t>(x_sub_u >> MONTY_BITS);
    uint32_t corr = over ? PRIME : 0u;
    return hi + corr;
}

P3_HOST_DEVICE P3_CONSTEXPR_HD inline uint32_t to_monty(uint32_t canonical) {
    constexpr uint32_t PRIME = 0x7f000001u;
    uint64_t c = canonical % PRIME;
    return static_cast<uint32_t>((c << 32) % PRIME);
}

P3_HOST_DEVICE P3_CONSTEXPR_HD inline uint32_t from_monty(uint32_t monty) {
    return monty_reduce_u64(static_cast<uint64_t>(monty));
}

} // namespace koala_bear_detail

class KoalaBear : public PrimeField<KoalaBear> {
private:
    uint32_t monty_;

public:
    static constexpr uint32_t PRIME     = 0x7f000001u;
    static constexpr uint64_t PRIME_U64 = PRIME;
    static constexpr uint32_t MONTY_MU  = 0x81000001u;
    static constexpr uint32_t MONTY_BITS = 32;
    static constexpr uint32_t MONTY_MASK = 0xFFFFFFFFu;

    static constexpr size_t TWO_ADICITY = 24;
    static constexpr size_t FIELD_BITS  = 31;

    /// Canonical cube root inverse exponent: 3 * 1420470955 ≡ 1 (mod p - 1)
    static constexpr uint64_t INJECTIVE_EXP_ROOT_D3 = 1420470955u;

    /// Multiplicative generator (canonical 3), per p3-monty-31 FieldParameters.
    static constexpr uint32_t GENERATOR_VAL = 3u;

    /// Montgomery representation of small constants (from p3-monty-31).
    static constexpr uint32_t MONTY_ONE_VAL    = 0x01fffffeu;
    static constexpr uint32_t MONTY_TWO_VAL    = 0x03fffffcu;
    static constexpr uint32_t MONTY_NEG_ONE_VAL = 0x7d000003u;
    static constexpr uint32_t MONTY_HALF_VAL   = 0x00ffffffu;

    /// Canonical entries for primitive 2^i-th roots (index i), from p3-koala-bear TwoAdicData.
    static constexpr std::array<uint32_t, 25> TWO_ADIC_GENERATOR_CANON = {
        0x00000001u, 0x7f000000u, 0x7e010002u, 0x6832fe4au, 0x08dbd69cu, 0x0a28f031u, 0x5c4a5b99u, 0x29b75a80u,
        0x17668b8au, 0x27ad539bu, 0x334d48c7u, 0x7744959cu, 0x768fc6fau, 0x303964b2u, 0x3e687d4du, 0x45a60e61u,
        0x6e2f4d7au, 0x163bd499u, 0x6c4a8a45u, 0x143ef899u, 0x514ddcadu, 0x484ef19bu, 0x205d63c3u, 0x68e7dd49u,
        0x6ac49f88u,
    };

    P3_HOST_DEVICE P3_CONSTEXPR_HD KoalaBear() : monty_(0u) {}

    /// Construct from canonical u32 (reduced mod p).
    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit KoalaBear(uint32_t canonical)
        : monty_(koala_bear_detail::to_monty(canonical)) {}

    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit KoalaBear(uint64_t canonical)
        : monty_(koala_bear_detail::to_monty(static_cast<uint32_t>(canonical % PRIME_U64))) {}

    P3_HOST_DEVICE P3_CONSTEXPR_HD explicit KoalaBear(int32_t v)
        : monty_(koala_bear_detail::to_monty(from_signed_u32(v))) {}

    /// Internal: already Montgomery-reduced limb in [0, p).
    P3_HOST_DEVICE P3_CONSTEXPR_HD static KoalaBear from_monty_rep(uint32_t m) {
        KoalaBear x;
        x.monty_ = m;
        return x;
    }

    P3_HOST_DEVICE static KoalaBear zero_val() { return KoalaBear(); }
    P3_HOST_DEVICE static KoalaBear one_val() { return from_monty_rep(MONTY_ONE_VAL); }
    P3_HOST_DEVICE static KoalaBear two_val() { return from_monty_rep(MONTY_TWO_VAL); }
    P3_HOST_DEVICE static KoalaBear neg_one_val() { return from_monty_rep(MONTY_NEG_ONE_VAL); }

#if !P3_CUDA_ENABLED
    static const KoalaBear ZERO;
    static const KoalaBear ONE;
    static const KoalaBear TWO;
    static const KoalaBear NEG_ONE;
#endif

    P3_HOST_DEVICE P3_CONSTEXPR_HD uint32_t value() const { return koala_bear_detail::from_monty(monty_); }

    P3_HOST_DEVICE P3_CONSTEXPR_HD uint64_t as_canonical_u64() const {
        return static_cast<uint64_t>(value());
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD uint32_t monty_rep() const { return monty_; }

    P3_HOST_DEVICE P3_CONSTEXPR_HD KoalaBear add(const KoalaBear& o) const {
        uint32_t s = monty_ + o.monty_;
        if (s >= PRIME) {
            s -= PRIME;
        }
        return from_monty_rep(s);
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD KoalaBear sub(const KoalaBear& o) const {
        uint32_t d = (monty_ >= o.monty_) ? (monty_ - o.monty_) : (monty_ + (PRIME - o.monty_));
        return from_monty_rep(d);
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD KoalaBear mul(const KoalaBear& o) const {
        uint64_t p = static_cast<uint64_t>(monty_) * static_cast<uint64_t>(o.monty_);
        return from_monty_rep(koala_bear_detail::monty_reduce_u64(p));
    }

    P3_HOST_DEVICE P3_CONSTEXPR_HD bool equals(const KoalaBear& o) const { return monty_ == o.monty_; }

    P3_HOST_DEVICE KoalaBear inv() const {
#if !P3_CUDA_ENABLED
        if (monty_ == 0u) {
            throw std::runtime_error("Cannot invert zero");
        }
#else
        P3_ASSERT(monty_ != 0u);
#endif
        uint32_t c = value();
        return KoalaBear(pow_mod_u32(c, PRIME - 2u));
    }

    /// Monty-31 limb halve (matches p3-monty-31 `halve_u32`).
    P3_HOST_DEVICE P3_CONSTEXPR_HD KoalaBear halve() const {
        uint32_t shr = monty_ >> 1u;
        uint32_t lo = monty_ & 1u;
        uint32_t shr_corr = shr + ((PRIME + 1u) >> 1u);
        uint32_t out = (lo == 0u) ? shr : shr_corr;
        return from_monty_rep(out);
    }

    /// Multiply by 2^exp (canonical 2) in the field.
    P3_HOST_DEVICE KoalaBear mul_2exp_u64(uint64_t exp) const {
        return mul(two_val().exp_u64(exp));
    }

    /// Divide by 2^exp (MontyField31 semantics).
    P3_HOST_DEVICE KoalaBear div_2exp_u64(uint64_t exp) const {
        if (exp <= 32u) {
            uint64_t long_prod = static_cast<uint64_t>(monty_) << (32u - static_cast<uint32_t>(exp));
            return from_monty_rep(koala_bear_detail::monty_reduce_u64(long_prod));
        }
        KoalaBear h = from_monty_rep(MONTY_HALF_VAL);
        return mul(h.exp_u64(exp));
    }

    template<uint64_t D>
    P3_HOST_DEVICE KoalaBear injective_exp_n() const {
        static_assert(D == 3u, "KoalaBear Poseidon S-box uses D=3 only");
        return exp_const_u64<3>();
    }

    template<uint64_t D>
    P3_HOST_DEVICE KoalaBear injective_exp_root_n() const {
        static_assert(D == 3u, "KoalaBear Poseidon S-box uses D=3 only");
        return exp_u64(INJECTIVE_EXP_ROOT_D3);
    }

    P3_HOST_DEVICE static KoalaBear generator() { return KoalaBear(GENERATOR_VAL); }

    P3_HOST_DEVICE static KoalaBear two_adic_generator(size_t bits) {
#if !P3_CUDA_ENABLED
        if (bits > TWO_ADICITY) {
            throw std::invalid_argument("bits exceeds TWO_ADICITY (24) for KoalaBear");
        }
#else
        P3_ASSERT(bits <= TWO_ADICITY);
#endif
        return KoalaBear(TWO_ADIC_GENERATOR_CANON[bits]);
    }

#if !P3_CUDA_ENABLED
    friend std::ostream& operator<<(std::ostream& os, const KoalaBear& f) {
        os << f.value();
        return os;
    }
#endif

private:
    P3_HOST_DEVICE static P3_CONSTEXPR_HD uint32_t from_signed_u32(int32_t v) {
        if (v >= 0) {
            return static_cast<uint32_t>(v) % PRIME;
        }
        int64_t x = static_cast<int64_t>(v) % static_cast<int64_t>(PRIME);
        if (x < 0) {
            x += static_cast<int64_t>(PRIME);
        }
        return static_cast<uint32_t>(x);
    }

    P3_HOST_DEVICE static uint32_t pow_mod_u32(uint32_t base, uint32_t e) {
        uint64_t r = 1u;
        uint64_t b = base % PRIME;
        uint64_t mod = PRIME;
        while (e > 0u) {
            if (e & 1u) {
                r = (r * b) % mod;
            }
            b = (b * b) % mod;
            e >>= 1u;
        }
        return static_cast<uint32_t>(r);
    }
};

#if !P3_CUDA_ENABLED
inline const KoalaBear KoalaBear::ZERO = KoalaBear();
inline const KoalaBear KoalaBear::ONE = KoalaBear::one_val();
inline const KoalaBear KoalaBear::TWO = KoalaBear::two_val();
inline const KoalaBear KoalaBear::NEG_ONE = KoalaBear::neg_one_val();
#endif

} // namespace p3_field

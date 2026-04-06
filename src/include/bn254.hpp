#pragma once

#include "cuda_compat.hpp"
#include "field.hpp"
#include <array>
#include <cstdint>
#include <string>

#if !P3_CUDA_ENABLED
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#endif

namespace p3_field {

// ============================================================================
// 256-bit arithmetic helpers (4 × uint64_t, little-endian limbs)
// ============================================================================
namespace bn254_detail {

/// Carrying add: returns (a + b + carry_in, carry_out)
P3_HOST_DEVICE P3_INLINE void carrying_add(uint64_t a, uint64_t b,
                                           bool carry_in, uint64_t &out,
                                           bool &carry_out) {
  uint64_t s1 = a + b;
  bool c1 = s1 < a;
  uint64_t s2 = s1 + static_cast<uint64_t>(carry_in);
  bool c2 = s2 < s1;
  out = s2;
  carry_out = c1 | c2;
}

/// Borrowing sub: returns (a - b - borrow_in, borrow_out)
P3_HOST_DEVICE P3_INLINE void borrowing_sub(uint64_t a, uint64_t b,
                                            bool borrow_in, uint64_t &out,
                                            bool &borrow_out) {
  uint64_t d1 = a - b;
  bool c1 = d1 > a;
  uint64_t d2 = d1 - static_cast<uint64_t>(borrow_in);
  bool c2 = d2 > d1;
  out = d2;
  borrow_out = c1 | c2;
}

using Limbs = uint64_t[4];

/// 256-bit wrapping addition: out = a + b, returns overflow flag
P3_HOST_DEVICE P3_INLINE bool wrapping_add(const Limbs a, const Limbs b,
                                           Limbs out) {
  bool carry = false;
  carrying_add(a[0], b[0], false, out[0], carry);
  carrying_add(a[1], b[1], carry, out[1], carry);
  carrying_add(a[2], b[2], carry, out[2], carry);
  carrying_add(a[3], b[3], carry, out[3], carry);
  return carry;
}

/// 256-bit wrapping subtraction: out = a - b, returns underflow flag
P3_HOST_DEVICE P3_INLINE bool wrapping_sub(const Limbs a, const Limbs b,
                                           Limbs out) {
  bool borrow = false;
  borrowing_sub(a[0], b[0], false, out[0], borrow);
  borrowing_sub(a[1], b[1], borrow, out[1], borrow);
  borrowing_sub(a[2], b[2], borrow, out[2], borrow);
  borrowing_sub(a[3], b[3], borrow, out[3], borrow);
  return borrow;
}

/// 128-bit multiply helper (portable)
P3_HOST_DEVICE P3_INLINE void mul128(uint64_t a, uint64_t b, uint64_t &lo,
                                     uint64_t &hi) {
#if P3_CUDA_ENABLED && defined(__CUDA_ARCH__)
  lo = a * b;
  hi = __umul64hi(a, b);
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
  __uint128_t prod = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
#pragma GCC diagnostic pop
  lo = static_cast<uint64_t>(prod);
  hi = static_cast<uint64_t>(prod >> 64);
#endif
}

/// Multiply 256-bit a by 64-bit b. Returns lowest limb and remaining 4 limbs.
/// out[0..3] = high limbs, return value = lowest limb
P3_HOST_DEVICE P3_INLINE uint64_t mul_small(const Limbs a, uint64_t b,
                                            Limbs out) {
  uint64_t lo, hi;

  // limb 0
  mul128(a[0], b, lo, hi);
  uint64_t out_0 = lo;
  uint64_t acc_lo = hi;

  // limb 1
  mul128(a[1], b, lo, hi);
  uint64_t sum = acc_lo + lo;
  bool carry = sum < acc_lo;
  out[0] = sum;
  acc_lo = hi + static_cast<uint64_t>(carry);

  // limb 2
  mul128(a[2], b, lo, hi);
  sum = acc_lo + lo;
  carry = sum < acc_lo;
  out[1] = sum;
  acc_lo = hi + static_cast<uint64_t>(carry);

  // limb 3
  mul128(a[3], b, lo, hi);
  sum = acc_lo + lo;
  carry = sum < acc_lo;
  out[2] = sum;
  out[3] = hi + static_cast<uint64_t>(carry);

  return out_0;
}

/// Multiply 256-bit a by 64-bit b and add 256-bit c.
/// out[0..3] = high limbs, return value = lowest limb
P3_HOST_DEVICE P3_INLINE uint64_t mul_small_and_acc(const Limbs a, uint64_t b,
                                                    const Limbs c, Limbs out) {
  uint64_t lo, hi;

  // limb 0: a[0]*b + c[0]
  mul128(a[0], b, lo, hi);
  uint64_t sum = lo + c[0];
  bool carry = sum < lo;
  uint64_t out_0 = sum;
  uint64_t acc = hi + static_cast<uint64_t>(carry);

  // limb 1: a[1]*b + c[1] + acc
  mul128(a[1], b, lo, hi);
  sum = lo + c[1];
  carry = sum < lo;
  uint64_t sum2 = sum + acc;
  bool carry2 = sum2 < sum;
  out[0] = sum2;
  acc = hi + static_cast<uint64_t>(carry) + static_cast<uint64_t>(carry2);

  // limb 2: a[2]*b + c[2] + acc
  mul128(a[2], b, lo, hi);
  sum = lo + c[2];
  carry = sum < lo;
  sum2 = sum + acc;
  carry2 = sum2 < sum;
  out[1] = sum2;
  acc = hi + static_cast<uint64_t>(carry) + static_cast<uint64_t>(carry2);

  // limb 3: a[3]*b + c[3] + acc
  mul128(a[3], b, lo, hi);
  sum = lo + c[3];
  carry = sum < lo;
  sum2 = sum + acc;
  carry2 = sum2 < sum;
  out[2] = sum2;
  out[3] = hi + static_cast<uint64_t>(carry) + static_cast<uint64_t>(carry2);

  return out_0;
}

/// BN254 prime in little-endian limbs
static constexpr uint64_t BN254_PRIME[4] = {
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL,
};

/// P^{-1} mod 2^64
static constexpr uint64_t BN254_MONTY_MU_64 = 0x3d1e0a6c10000001ULL;

/// R^2 mod P where R = 2^256
static constexpr uint64_t BN254_MONTY_R_SQ[4] = {
    0x1bb8e645ae216da7ULL,
    0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL,
    0x0216d0b17f4e44a5ULL,
};

/// Interleaved Montgomery reduction on a 320-bit value (acc0, acc[4]).
/// Returns (acc0 + acc * 2^64) * 2^{-64} mod P
P3_HOST_DEVICE P3_INLINE void
interleaved_monty_reduction(uint64_t acc0, const Limbs acc, Limbs out) {
  uint64_t t = acc0 * BN254_MONTY_MU_64;
  Limbs u;
  mul_small(BN254_PRIME, t, u);

  Limbs sub;
  bool under = wrapping_sub(acc, u, sub);
  if (under) {
    wrapping_add(sub, BN254_PRIME, out);
  } else {
    out[0] = sub[0];
    out[1] = sub[1];
    out[2] = sub[2];
    out[3] = sub[3];
  }
}

/// Montgomery multiplication: returns lhs * rhs * R^{-1} mod P
/// Requires lhs < P. No constraint on rhs.
P3_HOST_DEVICE P3_INLINE void monty_mul(const Limbs lhs, const Limbs rhs,
                                        Limbs out) {
  Limbs acc, res;

  // Round 0
  uint64_t acc0 = mul_small(lhs, rhs[0], acc);
  interleaved_monty_reduction(acc0, acc, res);

  // Round 1
  acc0 = mul_small_and_acc(lhs, rhs[1], res, acc);
  interleaved_monty_reduction(acc0, acc, res);

  // Round 2
  acc0 = mul_small_and_acc(lhs, rhs[2], res, acc);
  interleaved_monty_reduction(acc0, acc, res);

  // Round 3
  acc0 = mul_small_and_acc(lhs, rhs[3], res, acc);
  interleaved_monty_reduction(acc0, acc, out);
}

/// Halve a BN254 element (in Montgomery form).
P3_HOST_DEVICE P3_INLINE void halve_bn254(const Limbs input, Limbs out) {
  // If odd, add P first
  uint64_t v0 = input[0], v1 = input[1], v2 = input[2], v3 = input[3];
  if (v0 & 1) {
    bool carry = false;
    carrying_add(v0, BN254_PRIME[0], false, v0, carry);
    carrying_add(v1, BN254_PRIME[1], carry, v1, carry);
    carrying_add(v2, BN254_PRIME[2], carry, v2, carry);
    carrying_add(v3, BN254_PRIME[3], carry, v3, carry);
    // carry can be ignored since sum < 2^256
  }
  // Shift right by 1
  out[0] = (v0 >> 1) | (v1 << 63);
  out[1] = (v1 >> 1) | (v2 << 63);
  out[2] = (v2 >> 1) | (v3 << 63);
  out[3] = (v3 >> 1);
}

/// Compare a < b (little-endian limbs)
P3_HOST_DEVICE P3_INLINE bool less_than(const Limbs a, const Limbs b) {
  if (a[3] != b[3])
    return a[3] < b[3];
  if (a[2] != b[2])
    return a[2] < b[2];
  if (a[1] != b[1])
    return a[1] < b[1];
  return a[0] < b[0];
}

// ============================================================================
// GCD-based inversion (Binary Extended Euclidean Algorithm)
// ============================================================================

/// Inner GCD loop operating on u64 approximations.
/// Performs NUM_ROUNDS iterations of the binary GCD algorithm.
P3_HOST_DEVICE P3_INLINE void gcd_inner(uint64_t &a, uint64_t &b, int64_t &f0,
                                        int64_t &g0, int64_t &f1, int64_t &g1,
                                        int num_rounds) {
  f0 = 1;
  g0 = 0;
  f1 = 0;
  g1 = 1;
  for (int round = 0; round < num_rounds; ++round) {
    if ((a & 1) == 0) {
      a >>= 1;
    } else {
      if (a < b) {
        uint64_t tmp = a;
        a = b;
        b = tmp;
        int64_t tf = f0;
        f0 = f1;
        f1 = tf;
        int64_t tg = g0;
        g0 = g1;
        g1 = tg;
      }
      a -= b;
      a >>= 1;
      f0 -= f1;
      g0 -= g1;
    }
    f1 <<= 1;
    g1 <<= 1;
  }
}

/// Get the number of significant bits of the upper 3 limbs.
/// Returns (limb_index, bits_mod_64) where limb_index is 1-based from limb 1.
P3_HOST_DEVICE P3_INLINE void num_bits(uint64_t limb_1, uint64_t limb_2,
                                       uint64_t limb_3, int &limb,
                                       int &bits_mod_64) {
  // Count leading zeros for each limb
  // Use a portable clz: __builtin_clzll on CPU, manual on CUDA
  auto clz64 = [](uint64_t x) -> int {
    if (x == 0)
      return 64;
#if P3_CUDA_ENABLED && defined(__CUDA_ARCH__)
    return __clzll(x);
#else
    return __builtin_clzll(x);
#endif
  };

  int64_t v3 = 64 - static_cast<int64_t>(clz64(limb_3));
  int64_t v2 = 64 - static_cast<int64_t>(clz64(limb_2));
  int64_t v1 = 64 - static_cast<int64_t>(clz64(limb_1));

  // Side-channel-resistant selection
  int64_t v3_non_0 = (-v3) >> 63;
  int64_t v2_non_0 = ((-v2) >> 63) & (~v3_non_0);
  int64_t v1_non_0 = ((-v1) >> 63) & (~v3_non_0) & (~v2_non_0);

  limb = static_cast<int>((1 - v2_non_0 - (v3_non_0 << 1)));
  bits_mod_64 =
      static_cast<int>((v3_non_0 & v3) | (v2_non_0 & v2) | (v1_non_0 & v1));
}

/// Get approximation of val for inner GCD loop.
P3_HOST_DEVICE P3_INLINE uint64_t get_approximation(const Limbs val, int limb,
                                                    int bits_mod_64) {
  const int HALF_WORD_SIZE = 32;
  uint64_t bottom_bits = val[0] & ((1ULL << (HALF_WORD_SIZE - 1)) - 1);

#if P3_CUDA_ENABLED && defined(__CUDA_ARCH__)
  // On CUDA, manually compute the 128-bit shift
  uint64_t lo_val = val[limb - 1];
  uint64_t hi_val = val[limb];
  int shift = bits_mod_64 + 64 - (HALF_WORD_SIZE + 1);
  uint64_t top_bits;
  if (shift >= 64) {
    top_bits = hi_val >> (shift - 64);
  } else if (shift == 0) {
    top_bits = lo_val;
  } else {
    top_bits = (lo_val >> shift) | (hi_val << (64 - shift));
  }
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
  __uint128_t joined = (static_cast<__uint128_t>(val[limb]) << 64) |
                       static_cast<__uint128_t>(val[limb - 1]);
  uint64_t top_bits = static_cast<uint64_t>(
      joined >> (bits_mod_64 + 64 - (HALF_WORD_SIZE + 1)));
#pragma GCC diagnostic pop
#endif

  return (top_bits << (HALF_WORD_SIZE - 1)) | bottom_bits;
}

/// Conditionally negate (2's complement) if sign == ~0ULL
P3_HOST_DEVICE P3_INLINE void conditional_neg(Limbs a, uint64_t sign) {
  bool carry;
  uint64_t neg_sign = static_cast<uint64_t>(-static_cast<int64_t>(sign));
  carrying_add(a[0] ^ sign, neg_sign, false, a[0], carry);
  carrying_add(a[1] ^ sign, 0, carry, a[1], carry);
  carrying_add(a[2] ^ sign, 0, carry, a[2], carry);
  carrying_add(a[3] ^ sign, 0, carry, a[3], carry);
}

/// 128-bit two's-complement add: (alo,ahi) + (blo,bhi) -> (slo,shi)
P3_HOST_DEVICE P3_INLINE void i128_add(uint64_t alo, uint64_t ahi, uint64_t blo,
                                       uint64_t bhi, uint64_t &slo, uint64_t &shi) {
  slo = alo + blo;
  uint64_t c1 = slo < alo ? 1u : 0u;
  shi = ahi + bhi + c1;
}

/// u * s as signed 128-bit two's complement (lo, hi), matching (__int128)u * s on CPU.
P3_HOST_DEVICE P3_INLINE void mul_u64_i64_i128(uint64_t u, int64_t s, uint64_t &lo,
                                               uint64_t &hi) {
  const bool neg = s < 0;
  uint64_t abs_s =
      neg ? (static_cast<uint64_t>(-(s + 1)) + 1u) : static_cast<uint64_t>(s);
  uint64_t pl, ph;
  mul128(u, abs_s, pl, ph);
  if (!neg) {
    lo = pl;
    hi = ph;
    return;
  }
  lo = static_cast<uint64_t>(0u - pl);
  hi = static_cast<uint64_t>(0u - ph - (pl != 0u ? 1u : 0u));
}

/// Arithmetic right shift of signed int128 (..., ahi) by 64 -> (nlo,nhi).
P3_HOST_DEVICE P3_INLINE void i128_shr64(uint64_t ahi, uint64_t &nlo, uint64_t &nhi) {
  nlo = ahi;
  nhi = static_cast<uint64_t>(static_cast<int64_t>(ahi) >> 63);
}

/// Signed linear combination: out = a*f + b*g (returns 320-bit result: out[4] +
/// hi_limb)
P3_HOST_DEVICE P3_INLINE void linear_comb_signed(const Limbs a, const Limbs b,
                                                 int64_t f, int64_t g,
                                                 Limbs out, int64_t &hi_limb) {
#if P3_CUDA_ENABLED && defined(__CUDA_ARCH__)
  // Exact signed 128-bit accumulation (matches __int128 path); no int64 overflow.
  uint64_t c_lo = 0, c_hi = 0;
  for (int i = 0; i < 4; ++i) {
    uint64_t af_lo, af_hi, bg_lo, bg_hi;
    mul_u64_i64_i128(a[i], f, af_lo, af_hi);
    mul_u64_i64_i128(b[i], g, bg_lo, bg_hi);
    uint64_t t_lo, t_hi;
    i128_add(af_lo, af_hi, bg_lo, bg_hi, t_lo, t_hi);
    uint64_t s_lo, s_hi;
    i128_add(c_lo, c_hi, t_lo, t_hi, s_lo, s_hi);
    out[i] = s_lo;
    i128_shr64(s_hi, c_lo, c_hi);
  }
  hi_limb = static_cast<int64_t>(c_lo);
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
  __int128_t carry = 0;
  for (int i = 0; i < 4; ++i) {
    carry += static_cast<__int128_t>(a[i]) * static_cast<__int128_t>(f) +
             static_cast<__int128_t>(b[i]) * static_cast<__int128_t>(g);
    out[i] = static_cast<uint64_t>(carry);
    carry >>= 64;
  }
  hi_limb = static_cast<int64_t>(carry);
#pragma GCC diagnostic pop
#endif
}

/// Unsigned linear combination: out5 = a*f + b*g (320-bit result)
P3_HOST_DEVICE P3_INLINE void linear_comb_unsigned(const Limbs a, const Limbs b,
                                                   uint64_t f, uint64_t g,
                                                   uint64_t out[5]) {
#if P3_CUDA_ENABLED && defined(__CUDA_ARCH__)
  uint64_t carry_val = 0;
  for (int i = 0; i < 4; ++i) {
    uint64_t af_lo, af_hi, bg_lo, bg_hi;
    af_lo = a[i] * f;
    af_hi = __umul64hi(a[i], f);
    bg_lo = b[i] * g;
    bg_hi = __umul64hi(b[i], g);

    uint64_t sum1 = af_lo + bg_lo;
    uint64_t c1 = (sum1 < af_lo) ? 1 : 0;
    uint64_t sum2 = sum1 + carry_val;
    uint64_t c2 = (sum2 < sum1) ? 1 : 0;
    out[i] = sum2;
    carry_val = af_hi + bg_hi + c1 + c2;
  }
  out[4] = carry_val;
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
  __uint128_t carry = 0;
  for (int i = 0; i < 4; ++i) {
    carry += static_cast<__uint128_t>(a[i]) * static_cast<__uint128_t>(f) +
             static_cast<__uint128_t>(b[i]) * static_cast<__uint128_t>(g);
    out[i] = static_cast<uint64_t>(carry);
    carry >>= 64;
  }
  out[4] = static_cast<uint64_t>(carry);
#pragma GCC diagnostic pop
#endif
}

/// Compute (a*f + b*g) / 2^k and return absolute value + sign.
/// sign is 0 if positive, -1 (all bits set) if negative.
P3_HOST_DEVICE P3_INLINE void linear_comb_div(const Limbs a, const Limbs b,
                                              int64_t f, int64_t g, int k,
                                              Limbs out, int64_t &sign) {
  Limbs product;
  int64_t hi_limb;
  linear_comb_signed(a, b, f, g, product, hi_limb);

  // Shift right by k bits
  out[0] = (product[0] >> k) | (product[1] << (64 - k));
  out[1] = (product[1] >> k) | (product[2] << (64 - k));
  out[2] = (product[2] >> k) | (product[3] << (64 - k));
  out[3] = (product[3] >> k) | (static_cast<uint64_t>(hi_limb) << (64 - k));

  sign = hi_limb >> 63;
  conditional_neg(out, static_cast<uint64_t>(sign));
}

/// Linear combination with Montgomery reduction: (a*f + b*g) * R^{-1} mod P
P3_HOST_DEVICE P3_INLINE void linear_comb_monty_red(const Limbs a,
                                                    const Limbs b, int64_t f,
                                                    int64_t g, Limbs out) {
  int64_t s_f = f >> 63;
  int64_t s_g = g >> 63;
  uint64_t abs_f = static_cast<uint64_t>(f < 0 ? -f : f);
  uint64_t abs_g = static_cast<uint64_t>(g < 0 ? -g : g);

  // If sign is negative, negate in the field (subtract from P)
  Limbs a_signed, b_signed;
  if (s_f == -1) {
    wrapping_sub(BN254_PRIME, a, a_signed);
  } else {
    a_signed[0] = a[0];
    a_signed[1] = a[1];
    a_signed[2] = a[2];
    a_signed[3] = a[3];
  }
  if (s_g == -1) {
    wrapping_sub(BN254_PRIME, b, b_signed);
  } else {
    b_signed[0] = b[0];
    b_signed[1] = b[1];
    b_signed[2] = b[2];
    b_signed[3] = b[3];
  }

  uint64_t product[5];
  linear_comb_unsigned(a_signed, b_signed, abs_f, abs_g, product);

  Limbs acc = {product[1], product[2], product[3], product[4]};
  interleaved_monty_reduction(product[0], acc, out);
}

/// 2^{1030} mod P — adjustment factor for GCD inversion
static constexpr uint64_t BN254_2_POW_1030[4] = {
    0x1f7ca21e7fcb111bULL,
    0x61a09399fcfe8a6cULL,
    0x1438cc5aab55aedbULL,
    0x020c9ba0aeb6b6c7ULL,
};

/// GCD-based inversion in BN254 field.
/// Based on Pornin's "Optimized Binary GCD for Modular Inversion" (eprint
/// 2020/972).
P3_HOST_DEVICE P3_INLINE void gcd_inversion(const Limbs input, Limbs out) {
  Limbs a, u, b, v;
  a[0] = input[0];
  a[1] = input[1];
  a[2] = input[2];
  a[3] = input[3];
  u[0] = BN254_2_POW_1030[0];
  u[1] = BN254_2_POW_1030[1];
  u[2] = BN254_2_POW_1030[2];
  u[3] = BN254_2_POW_1030[3];
  b[0] = BN254_PRIME[0];
  b[1] = BN254_PRIME[1];
  b[2] = BN254_PRIME[2];
  b[3] = BN254_PRIME[3];
  v[0] = 0;
  v[1] = 0;
  v[2] = 0;
  v[3] = 0;

  const int ROUND_SIZE = 31;
  const int NUM_ROUNDS = 15;
  const int FINAL_ROUND_SIZE = 41;
  // 15 * 31 + 41 = 506

  for (int i = 0; i < NUM_ROUNDS; ++i) {
    int limb, bits_mod_64;
    num_bits(a[1] | b[1], a[2] | b[2], a[3] | b[3], limb, bits_mod_64);
    uint64_t a_tilde = get_approximation(a, limb, bits_mod_64);
    uint64_t b_tilde = get_approximation(b, limb, bits_mod_64);

    int64_t f0, g0, f1, g1;
    gcd_inner(a_tilde, b_tilde, f0, g0, f1, g1, ROUND_SIZE);

    // Update a
    Limbs new_a;
    int64_t sign;
    linear_comb_div(a, b, f0, g0, ROUND_SIZE, new_a, sign);
    // If sign was negative, flip f0 and g0
    f0 = (f0 ^ sign) - sign;
    g0 = (g0 ^ sign) - sign;

    // Update b
    Limbs new_b;
    linear_comb_div(a, b, f1, g1, ROUND_SIZE, new_b, sign);
    f1 = (f1 ^ sign) - sign;
    g1 = (g1 ^ sign) - sign;

    // Update u, v
    Limbs new_u, new_v;
    linear_comb_monty_red(u, v, f0, g0, new_u);
    linear_comb_monty_red(u, v, f1, g1, new_v);

    a[0] = new_a[0];
    a[1] = new_a[1];
    a[2] = new_a[2];
    a[3] = new_a[3];
    b[0] = new_b[0];
    b[1] = new_b[1];
    b[2] = new_b[2];
    b[3] = new_b[3];
    u[0] = new_u[0];
    u[1] = new_u[1];
    u[2] = new_u[2];
    u[3] = new_u[3];
    v[0] = new_v[0];
    v[1] = new_v[1];
    v[2] = new_v[2];
    v[3] = new_v[3];
  }

  // Final round: a and b now fit in u64
  int64_t f0, g0, f1, g1;
  gcd_inner(a[0], b[0], f0, g0, f1, g1, FINAL_ROUND_SIZE);

  linear_comb_monty_red(u, v, f1, g1, out);
}

} // namespace bn254_detail

// ============================================================================
// Bn254 field element
// ============================================================================

/**
 * @brief The BN254 curve scalar field: p =
 * 21888242871839275222246405745257275088548364400416034343698204186575808495617
 *
 * Elements are stored in Montgomery form (aR mod P where R = 2^256).
 * Internal representation: 4 × uint64_t limbs in little-endian order.
 *
 * This implementation supports both CPU (C++) and GPU (CUDA) execution.
 */
class Bn254 : public PrimeField<Bn254> {
private:
  uint64_t value_[4];

public:
  // The BN254 prime
  static constexpr uint64_t PRIME[4] = {
      0x43e1f593f0000001ULL,
      0x2833e84879b97091ULL,
      0xb85045b68181585dULL,
      0x30644e72e131a029ULL,
  };

  static constexpr size_t TWO_ADICITY = 28;
  static constexpr size_t FIELD_BITS = 254;

  // Multiplicative group generator = 5
  static constexpr uint64_t GENERATOR_MONTY[4] = {
      0x1b0d0ef99fffffe6ULL,
      0xeaba68a3a32a913fULL,
      0x47d8eb76d8dd0689ULL,
      0x15d0085520f5bbc3ULL,
  };

  // Montgomery form of ONE (R mod P = 2^256 mod P)
  static constexpr uint64_t ONE_MONTY[4] = {
      0xac96341c4ffffffbULL,
      0x36fc76959f60cd29ULL,
      0x666ea36f7879462eULL,
      0x0e0a77c19a07df2fULL,
  };

  // Montgomery form of TWO
  static constexpr uint64_t TWO_MONTY[4] = {
      0x592c68389ffffff6ULL,
      0x6df8ed2b3ec19a53ULL,
      0xccdd46def0f28c5cULL,
      0x1c14ef83340fbe5eULL,
  };

  // Montgomery form of NEG_ONE (-1 mod P in Monty form)
  static constexpr uint64_t NEG_ONE_MONTY[4] = {
      0x974bc177a0000006ULL,
      0xf13771b2da58a367ULL,
      0x51e1a2470908122eULL,
      0x2259d6b14729c0faULL,
  };

  // R^2 mod P
  static constexpr uint64_t R_SQUARED[4] = {
      0x1bb8e645ae216da7ULL,
      0x53fe3ab1e35c59e3ULL,
      0x8c49833d53bb8085ULL,
      0x0216d0b17f4e44a5ULL,
  };

  // Two-adic generator: 5^{(P-1)/2^28} in Montgomery form
  static constexpr uint64_t TWO_ADIC_GENERATOR_MONTY[4] = {
      0x636e735580d13d9cULL,
      0xa22bf3742445ffd6ULL,
      0x56452ac01eb203d8ULL,
      0x1860ef942963f9e7ULL,
  };

  // -----------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------

  /// Default: zero
  P3_HOST_DEVICE P3_CONSTEXPR_HD Bn254() : value_{0, 0, 0, 0} {}

  /// From raw Montgomery-form limbs (no reduction)
  P3_HOST_DEVICE P3_CONSTEXPR_HD static Bn254 from_monty(const uint64_t v[4]) {
    Bn254 r;
    r.value_[0] = v[0];
    r.value_[1] = v[1];
    r.value_[2] = v[2];
    r.value_[3] = v[3];
    return r;
  }

  /// From canonical integer limbs (converts to Montgomery form)
  P3_HOST_DEVICE static Bn254 from_canonical(const uint64_t v[4]) {
    // Multiply by R^2 and reduce to get v*R mod P
    uint64_t monty[4];
    bn254_detail::monty_mul(bn254_detail::BN254_MONTY_R_SQ, v, monty);
    return from_monty(monty);
  }

  /// From a small uint64_t value (converts to Montgomery form)
  P3_HOST_DEVICE static Bn254 from_u64(uint64_t v) {
    uint64_t limbs[4] = {v, 0, 0, 0};
    return from_canonical(limbs);
  }

  /// From a small int64_t value
  P3_HOST_DEVICE static Bn254 from_i64(int64_t v) {
    if (v >= 0) {
      return from_u64(static_cast<uint64_t>(v));
    } else {
      return -from_u64(static_cast<uint64_t>(-v));
    }
  }

  // -----------------------------------------------------------------------
  // Factory methods (GPU compatible)
  // -----------------------------------------------------------------------

  P3_HOST_DEVICE static Bn254 zero_val() { return Bn254(); }

  P3_HOST_DEVICE static Bn254 one_val() { return from_monty(ONE_MONTY); }

  P3_HOST_DEVICE static Bn254 two_val() { return from_monty(TWO_MONTY); }

  P3_HOST_DEVICE static Bn254 neg_one_val() {
    return from_monty(NEG_ONE_MONTY);
  }

  P3_HOST_DEVICE static Bn254 generator() {
    return from_monty(GENERATOR_MONTY);
  }

  // Legacy CPU-only static constants
#if !P3_CUDA_ENABLED
  static const Bn254 ZERO;
  static const Bn254 ONE;
  static const Bn254 TWO;
  static const Bn254 NEG_ONE;
  static const Bn254 GENERATOR;
#endif

  // -----------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------

  /// Access the raw Montgomery-form limbs
  P3_HOST_DEVICE const uint64_t *limbs() const { return value_; }

  /// Check if element is zero
  P3_HOST_DEVICE bool is_zero() const {
    return value_[0] == 0 && value_[1] == 0 && value_[2] == 0 && value_[3] == 0;
  }

  /// Convert out of Montgomery form to canonical representation.
  /// Returns the value as 4 little-endian uint64_t limbs in [0, P).
  P3_HOST_DEVICE void as_canonical(uint64_t out[4]) const {
    // monty_mul(value, 1) strips out factor of R
    uint64_t one[4] = {1, 0, 0, 0};
    bn254_detail::monty_mul(value_, one, out);
  }

  /// Convert to canonical uint64 (only valid if value fits in 64 bits)
  P3_HOST_DEVICE uint64_t as_canonical_u64() const {
    uint64_t canonical[4];
    as_canonical(canonical);
    return canonical[0];
  }

  // -----------------------------------------------------------------------
  // Arithmetic operations (required by PrimeField CRTP)
  // -----------------------------------------------------------------------

  P3_HOST_DEVICE Bn254 add(const Bn254 &other) const {
    uint64_t sum[4];
    bn254_detail::wrapping_add(value_, other.value_, sum);

    uint64_t sum_corr[4];
    bool underflow =
        bn254_detail::wrapping_sub(sum, bn254_detail::BN254_PRIME, sum_corr);

    if (underflow) {
      return from_monty(sum);
    } else {
      return from_monty(sum_corr);
    }
  }

  P3_HOST_DEVICE Bn254 sub(const Bn254 &other) const {
    uint64_t diff[4];
    bool underflow = bn254_detail::wrapping_sub(value_, other.value_, diff);

    if (underflow) {
      uint64_t corrected[4];
      bn254_detail::wrapping_add(diff, bn254_detail::BN254_PRIME, corrected);
      return from_monty(corrected);
    }
    return from_monty(diff);
  }

  P3_HOST_DEVICE Bn254 mul(const Bn254 &other) const {
    uint64_t result[4];
    bn254_detail::monty_mul(value_, other.value_, result);
    return from_monty(result);
  }

  P3_HOST_DEVICE bool equals(const Bn254 &other) const {
    return value_[0] == other.value_[0] && value_[1] == other.value_[1] &&
           value_[2] == other.value_[2] && value_[3] == other.value_[3];
  }

  /// Inverse using GCD-based algorithm
  P3_HOST_DEVICE Bn254 inv() const {
#if !P3_CUDA_ENABLED
    if (is_zero()) {
      throw std::runtime_error("Cannot invert zero in Bn254");
    }
#else
    P3_ASSERT(!is_zero());
#endif
    uint64_t result[4];
    bn254_detail::gcd_inversion(value_, result);
    return from_monty(result);
  }

  /// Halve: compute a/2
  P3_HOST_DEVICE Bn254 halve() const {
    uint64_t result[4];
    bn254_detail::halve_bn254(value_, result);
    return from_monty(result);
  }

  /// Double: compute 2a
  P3_HOST_DEVICE Bn254 double_val() const { return add(*this); }

  /// Returns a primitive 2^bits-th root of unity.
  P3_HOST_DEVICE static Bn254 two_adic_generator(size_t bits) {
#if !P3_CUDA_ENABLED
    if (bits > TWO_ADICITY) {
      throw std::invalid_argument("bits exceeds TWO_ADICITY (28) for Bn254");
    }
#else
    P3_ASSERT(bits <= TWO_ADICITY);
#endif
    Bn254 omega = from_monty(TWO_ADIC_GENERATOR_MONTY);
    for (size_t i = bits; i < TWO_ADICITY; ++i) {
      omega = omega.square();
    }
    return omega;
  }

  /// Exponentiation by a 256-bit exponent using 4-bit windowed
  /// square-and-multiply
  P3_HOST_DEVICE Bn254 exp_u256(const uint64_t exp[4]) const {
    Bn254 table[16];
    table[0] = one_val();
    table[1] = *this;
    for (int i = 2; i < 16; ++i) {
      table[i] = table[i - 1] * (*this);
    }

    bool started = false;
    Bn254 result = one_val();

    for (int nibble_idx = 63; nibble_idx >= 0; --nibble_idx) {
      int limb_idx = nibble_idx / 16;
      int shift = (nibble_idx % 16) * 4;
      int nibble = static_cast<int>((exp[limb_idx] >> shift) & 0xf);

      if (started) {
        // Square 4 times
        result = result.square().square().square().square();
        if (nibble != 0) {
          result = result * table[nibble];
        }
      } else if (nibble != 0) {
        result = table[nibble];
        started = true;
      }
    }
    return result;
  }

  /// Injective power map x^5 (S-box for Poseidon2)
  template <uint64_t D> P3_HOST_DEVICE Bn254 injective_exp_n() const {
    return exp_const_u64<D>();
  }

  /// Fifth root: x^{5^{-1} mod (P-1)}
  template <uint64_t D> P3_HOST_DEVICE Bn254 injective_exp_root_n() const {
    static constexpr uint64_t BN254_FIFTH_ROOT_EXP[4] = {
        0xcfe7f7a98ccccccdULL,
        0x535cb9d394945a0dULL,
        0x93736af8679aad17ULL,
        0x26b6a528b427b354ULL,
    };
    return exp_u256(BN254_FIFTH_ROOT_EXP);
  }

  // Display (CPU only)
#if !P3_CUDA_ENABLED
  friend std::ostream &operator<<(std::ostream &os, const Bn254 &field) {
    uint64_t canonical[4];
    field.as_canonical(canonical);
    // Print as decimal would be complex; print hex limbs
    os << "Bn254(0x";
    for (int i = 3; i >= 0; --i) {
      os << std::hex << std::setfill('0') << std::setw(16) << canonical[i];
    }
    os << std::dec << ")";
    return os;
  }
#endif
};

// CPU-only static constant definitions
#if !P3_CUDA_ENABLED
inline const Bn254 Bn254::ZERO = Bn254::zero_val();
inline const Bn254 Bn254::ONE = Bn254::one_val();
inline const Bn254 Bn254::TWO = Bn254::two_val();
inline const Bn254 Bn254::NEG_ONE = Bn254::neg_one_val();
inline const Bn254 Bn254::GENERATOR = Bn254::generator();
#endif

} // namespace p3_field

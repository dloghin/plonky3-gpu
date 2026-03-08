#pragma once

/**
 * @file cuda_compat.hpp
 * @brief CUDA compatibility macros and utilities for CPU/GPU code sharing
 *
 * This header provides macros that allow the same code to compile for both
 * CPU (C++) and GPU (CUDA) targets without code duplication.
 */

#include <cstdint>

// Detect CUDA compilation
#ifdef __CUDACC__
    #define P3_CUDA_ENABLED 1
#else
    #define P3_CUDA_ENABLED 0
#endif

// Host/Device function qualifiers
#if P3_CUDA_ENABLED
    #define P3_HOST_DEVICE __host__ __device__
    #define P3_DEVICE __device__
    #define P3_HOST __host__
    #define P3_GLOBAL __global__
    #define P3_INLINE __forceinline__
#else
    #define P3_HOST_DEVICE
    #define P3_DEVICE
    #define P3_HOST
    #define P3_GLOBAL
    #define P3_INLINE inline
#endif

// Constexpr compatibility (CUDA < 11 doesn't support constexpr well)
#if P3_CUDA_ENABLED
    #define P3_CONSTEXPR
    #define P3_CONSTEXPR_HD P3_HOST_DEVICE
#else
    #define P3_CONSTEXPR constexpr
    #define P3_CONSTEXPR_HD constexpr
#endif

// Assertions for GPU code
#if P3_CUDA_ENABLED
    #define P3_ASSERT(cond) do { if (!(cond)) { asm("trap;"); } } while(0)
#else
    #include <cassert>
    #define P3_ASSERT(cond) assert(cond)
#endif

// CUDA error checking macro (host-only)
#if P3_CUDA_ENABLED
    #include <cuda_runtime.h>
    #include <stdexcept>
    #include <string>
    #define P3_CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                    std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
            } \
        } while(0)
#else
    // No-op for non-CUDA builds
    #define P3_CUDA_CHECK(call) (call)
#endif

// Utilities for 128-bit arithmetic on GPU (since __uint128_t is not available)
namespace p3_field {
namespace cuda_util {

/**
 * @brief 128-bit unsigned integer for GPU compatibility
 *
 * CUDA doesn't support __uint128_t, so we provide a struct for 128-bit operations
 */
struct uint128_t {
    uint64_t lo;
    uint64_t hi;

    P3_HOST_DEVICE P3_INLINE uint128_t() : lo(0), hi(0) {}
    P3_HOST_DEVICE P3_INLINE uint128_t(uint64_t low) : lo(low), hi(0) {}
    P3_HOST_DEVICE P3_INLINE uint128_t(uint64_t high, uint64_t low) : lo(low), hi(high) {}

    // Cast operators
    P3_HOST_DEVICE P3_INLINE explicit operator uint64_t() const { return lo; }

    // Right shift by 64
    P3_HOST_DEVICE P3_INLINE uint64_t high64() const { return hi; }
    P3_HOST_DEVICE P3_INLINE uint64_t low64() const { return lo; }
};

/**
 * @brief Multiply two 64-bit integers to get 128-bit result
 */
P3_HOST_DEVICE P3_INLINE uint128_t mul64(uint64_t a, uint64_t b) {
#if P3_CUDA_ENABLED && defined(__CUDA_ARCH__)
    // Use CUDA intrinsics for 64x64 -> 128 multiplication
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    return uint128_t(hi, lo);
#else
    // Use compiler's 128-bit type on host
    __uint128_t prod = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return uint128_t(static_cast<uint64_t>(prod >> 64), static_cast<uint64_t>(prod));
#endif
}

/**
 * @brief Add with carry for 64-bit integers
 */
P3_HOST_DEVICE P3_INLINE uint64_t add_with_carry(uint64_t a, uint64_t b, uint64_t& carry) {
    uint64_t sum = a + b;
    carry = (sum < a) ? 1 : 0;
    return sum;
}

/**
 * @brief Subtract with borrow for 64-bit integers
 */
P3_HOST_DEVICE P3_INLINE uint64_t sub_with_borrow(uint64_t a, uint64_t b, uint64_t& borrow) {
    uint64_t diff = a - b;
    borrow = (diff > a) ? 1 : 0;
    return diff;
}

} // namespace cuda_util
} // namespace p3_field


/**
 * @file goldilocks_cuda_example.cu
 * @brief CUDA example demonstrating Goldilocks field arithmetic on GPU
 *
 * This example shows:
 * - Field element creation on GPU
 * - Basic arithmetic (add, sub, mul) in parallel
 * - Exponentiation and inverse computation
 * - Data transfer between host and device
 */

#include "goldilocks.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

using namespace p3_field;

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief Kernel to perform element-wise field addition
 */
__global__ void goldilocks_add_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Goldilocks fa(a[idx]);
        Goldilocks fb(b[idx]);
        Goldilocks fc = fa + fb;
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to perform element-wise field multiplication
 */
__global__ void goldilocks_mul_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Goldilocks fa(a[idx]);
        Goldilocks fb(b[idx]);
        Goldilocks fc = fa * fb;
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to compute squares of field elements
 */
__global__ void goldilocks_square_kernel(
    const uint64_t* a,
    uint64_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Goldilocks fa(a[idx]);
        Goldilocks fc = fa.square();
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to compute x^7 (injective power map for Goldilocks)
 */
__global__ void goldilocks_pow7_kernel(
    const uint64_t* a,
    uint64_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Goldilocks fa(a[idx]);
        Goldilocks fc = fa.injective_exp_n<7>();
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to compute modular inverse
 */
__global__ void goldilocks_inverse_kernel(
    const uint64_t* a,
    uint64_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Goldilocks fa(a[idx]);
        if (fa.value() != 0) {
            Goldilocks fc = fa.inverse();
            result[idx] = fc.value();
        } else {
            result[idx] = 0; // Cannot invert zero
        }
    }
}

/**
 * @brief Kernel demonstrating complex field expression: (a + b) * (a - b)
 */
__global__ void goldilocks_difference_of_squares_kernel(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Goldilocks fa(a[idx]);
        Goldilocks fb(b[idx]);
        // (a + b) * (a - b) = a^2 - b^2
        Goldilocks sum = fa + fb;
        Goldilocks diff = fa - fb;
        Goldilocks fc = sum * diff;
        result[idx] = fc.value();
    }
}

void print_results(const char* name, const uint64_t* h_a, const uint64_t* h_b,
                   const uint64_t* h_result, int n, int max_print = 5) {
    printf("\n=== %s ===\n", name);
    for (int i = 0; i < min(n, max_print); i++) {
        printf("  [%d] %lu op %lu = %lu\n", i, h_a[i], h_b ? h_b[i] : 0, h_result[i]);
    }
    if (n > max_print) {
        printf("  ... (%d more elements)\n", n - max_print);
    }
}

int main() {
    printf("=========================================\n");
    printf("  Goldilocks Field CUDA Example\n");
    printf("  Prime: p = 2^64 - 2^32 + 1\n");
    printf("=========================================\n\n");

    // Problem size
    const int N = 1024 * 1024; // 1M elements
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Host arrays
    uint64_t* h_a = new uint64_t[N];
    uint64_t* h_b = new uint64_t[N];
    uint64_t* h_result = new uint64_t[N];

    // Initialize test data
    printf("Initializing %d field elements...\n", N);
    for (int i = 0; i < N; i++) {
        // Use values that won't overflow when added/multiplied
        h_a[i] = (static_cast<uint64_t>(i) * 12345) % Goldilocks::PRIME;
        h_b[i] = (static_cast<uint64_t>(i) * 67890 + 1) % Goldilocks::PRIME;
    }

    // Device arrays
    uint64_t *d_a, *d_b, *d_result;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_result, N * sizeof(uint64_t)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms;

    // Test 1: Addition
    CUDA_CHECK(cudaEventRecord(start));
    goldilocks_add_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    print_results("Addition (a + b)", h_a, h_b, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Verify first result on CPU
    {
        Goldilocks a(h_a[0]), b(h_b[0]);
        Goldilocks expected = a + b;
        printf("  Verification: %lu + %lu = %lu (GPU) vs %lu (CPU) - %s\n",
               h_a[0], h_b[0], h_result[0], expected.value(),
               h_result[0] == expected.value() ? "PASS" : "FAIL");
    }

    // Test 2: Multiplication
    CUDA_CHECK(cudaEventRecord(start));
    goldilocks_mul_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    print_results("Multiplication (a * b)", h_a, h_b, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Verify
    {
        Goldilocks a(h_a[1]), b(h_b[1]);
        Goldilocks expected = a * b;
        printf("  Verification: %lu * %lu = %lu (GPU) vs %lu (CPU) - %s\n",
               h_a[1], h_b[1], h_result[1], expected.value(),
               h_result[1] == expected.value() ? "PASS" : "FAIL");
    }

    // Test 3: Square
    CUDA_CHECK(cudaEventRecord(start));
    goldilocks_square_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    print_results("Square (a^2)", h_a, nullptr, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Test 4: Power of 7 (S-box for Goldilocks)
    CUDA_CHECK(cudaEventRecord(start));
    goldilocks_pow7_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    print_results("Power x^7 (S-box)", h_a, nullptr, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Test 5: Inverse (smaller size due to cost)
    const int N_INV = 1024;
    CUDA_CHECK(cudaEventRecord(start));
    goldilocks_inverse_kernel<<<(N_INV + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_result, N_INV);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N_INV * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    print_results("Inverse (a^(-1))", h_a, nullptr, h_result, N_INV);
    printf("  Time: %.3f ms (%.2f K ops/sec)\n", ms, N_INV / ms);

    // Verify: a * a^(-1) = 1
    {
        Goldilocks a(h_a[1]);
        Goldilocks inv(h_result[1]);
        Goldilocks product = a * inv;
        printf("  Verification: %lu * %lu = %lu (should be 1) - %s\n",
               h_a[1], h_result[1], product.as_canonical_u64(),
               product.as_canonical_u64() == 1 ? "PASS" : "FAIL");
    }

    // Test 6: Difference of squares
    CUDA_CHECK(cudaEventRecord(start));
    goldilocks_difference_of_squares_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    print_results("Difference of Squares (a+b)(a-b)", h_a, h_b, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    delete[] h_a;
    delete[] h_b;
    delete[] h_result;

    printf("\n=========================================\n");
    printf("  All Goldilocks CUDA tests completed!\n");
    printf("=========================================\n");

    return 0;
}


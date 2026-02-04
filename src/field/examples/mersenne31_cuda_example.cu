/**
 * @file mersenne31_cuda_example.cu
 * @brief CUDA example demonstrating Mersenne-31 field arithmetic on GPU
 *
 * This example shows:
 * - Field element creation on GPU
 * - Basic arithmetic (add, sub, mul) in parallel
 * - Exponentiation and inverse computation
 * - Data transfer between host and device
 * - Efficient Mersenne prime reduction
 */

#include "mersenne31.hpp"
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
__global__ void mersenne31_add_kernel(
    const uint32_t* a,
    const uint32_t* b,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Mersenne31 fa(a[idx]);
        Mersenne31 fb(b[idx]);
        Mersenne31 fc = fa + fb;
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to perform element-wise field subtraction
 */
__global__ void mersenne31_sub_kernel(
    const uint32_t* a,
    const uint32_t* b,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Mersenne31 fa(a[idx]);
        Mersenne31 fb(b[idx]);
        Mersenne31 fc = fa - fb;
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to perform element-wise field multiplication
 */
__global__ void mersenne31_mul_kernel(
    const uint32_t* a,
    const uint32_t* b,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Mersenne31 fa(a[idx]);
        Mersenne31 fb(b[idx]);
        Mersenne31 fc = fa * fb;
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to compute squares of field elements
 */
__global__ void mersenne31_square_kernel(
    const uint32_t* a,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Mersenne31 fa(a[idx]);
        Mersenne31 fc = fa.square();
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to compute x^5 (injective power map for Mersenne-31)
 */
__global__ void mersenne31_pow5_kernel(
    const uint32_t* a,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Mersenne31 fa(a[idx]);
        Mersenne31 fc = fa.injective_exp_n<5>();
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to compute modular inverse
 */
__global__ void mersenne31_inverse_kernel(
    const uint32_t* a,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Mersenne31 fa(a[idx]);
        if (fa.value() != 0) {
            Mersenne31 fc = fa.inverse();
            result[idx] = fc.value();
        } else {
            result[idx] = 0; // Cannot invert zero
        }
    }
}

/**
 * @brief Kernel demonstrating batch operations: compute (a_i * b_i + c_i)
 */
__global__ void mersenne31_fma_kernel(
    const uint32_t* a,
    const uint32_t* b,
    const uint32_t* c,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Mersenne31 fa(a[idx]);
        Mersenne31 fb(b[idx]);
        Mersenne31 fc(c[idx]);
        // Fused multiply-add: a * b + c
        Mersenne31 res = fa * fb + fc;
        result[idx] = res.value();
    }
}

/**
 * @brief Kernel for repeated squaring (used in FFT butterfly)
 */
__global__ void mersenne31_repeated_square_kernel(
    const uint32_t* a,
    uint32_t* result,
    int rounds,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Mersenne31 val(a[idx]);
        for (int i = 0; i < rounds; i++) {
            val = val.square();
        }
        result[idx] = val.value();
    }
}

/**
 * @brief Kernel demonstrating negation
 */
__global__ void mersenne31_neg_kernel(
    const uint32_t* a,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Mersenne31 fa(a[idx]);
        Mersenne31 fc = -fa;
        result[idx] = fc.value();
    }
}

void print_results(const char* name, const uint32_t* h_a, const uint32_t* h_b,
                   const uint32_t* h_result, int n, int max_print = 5) {
    printf("\n=== %s ===\n", name);
    for (int i = 0; i < min(n, max_print); i++) {
        printf("  [%d] %u op %u = %u\n", i, h_a[i], h_b ? h_b[i] : 0, h_result[i]);
    }
    if (n > max_print) {
        printf("  ... (%d more elements)\n", n - max_print);
    }
}

int main() {
    printf("=========================================\n");
    printf("  Mersenne-31 Field CUDA Example\n");
    printf("  Prime: p = 2^31 - 1 = %u\n", Mersenne31::PRIME);
    printf("=========================================\n\n");

    // Get device info
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("GPU: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Max threads per block: %d\n\n", prop.maxThreadsPerBlock);
    }

    // Problem size
    const int N = 1024 * 1024; // 1M elements
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Host arrays
    uint32_t* h_a = new uint32_t[N];
    uint32_t* h_b = new uint32_t[N];
    uint32_t* h_c = new uint32_t[N];
    uint32_t* h_result = new uint32_t[N];

    // Initialize test data
    printf("Initializing %d field elements...\n", N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (static_cast<uint32_t>(i) * 12345) % Mersenne31::PRIME;
        h_b[i] = (static_cast<uint32_t>(i) * 67890 + 1) % Mersenne31::PRIME;
        h_c[i] = (static_cast<uint32_t>(i) * 11111 + 2) % Mersenne31::PRIME;
    }

    // Device arrays
    uint32_t *d_a, *d_b, *d_c, *d_result;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result, N * sizeof(uint32_t)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms;

    // Test 1: Addition
    CUDA_CHECK(cudaEventRecord(start));
    mersenne31_add_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Addition (a + b)", h_a, h_b, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Verify first result on CPU
    {
        Mersenne31 a(h_a[0]), b(h_b[0]);
        Mersenne31 expected = a + b;
        printf("  Verification: %u + %u = %u (GPU) vs %u (CPU) - %s\n",
               h_a[0], h_b[0], h_result[0], expected.value(),
               h_result[0] == expected.value() ? "PASS" : "FAIL");
    }

    // Test 2: Subtraction
    CUDA_CHECK(cudaEventRecord(start));
    mersenne31_sub_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Subtraction (a - b)", h_a, h_b, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Test 3: Multiplication
    CUDA_CHECK(cudaEventRecord(start));
    mersenne31_mul_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Multiplication (a * b)", h_a, h_b, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Verify
    {
        Mersenne31 a(h_a[1]), b(h_b[1]);
        Mersenne31 expected = a * b;
        printf("  Verification: %u * %u = %u (GPU) vs %u (CPU) - %s\n",
               h_a[1], h_b[1], h_result[1], expected.value(),
               h_result[1] == expected.value() ? "PASS" : "FAIL");
    }

    // Test 4: Square
    CUDA_CHECK(cudaEventRecord(start));
    mersenne31_square_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Square (a^2)", h_a, nullptr, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Test 5: Power of 5 (S-box for Mersenne-31)
    CUDA_CHECK(cudaEventRecord(start));
    mersenne31_pow5_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Power x^5 (S-box)", h_a, nullptr, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Test 6: Inverse (smaller size due to cost)
    const int N_INV = 1024;
    CUDA_CHECK(cudaEventRecord(start));
    mersenne31_inverse_kernel<<<(N_INV + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_result, N_INV);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N_INV * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Inverse (a^(-1))", h_a, nullptr, h_result, N_INV);
    printf("  Time: %.3f ms (%.2f K ops/sec)\n", ms, N_INV / ms);

    // Verify: a * a^(-1) = 1
    {
        Mersenne31 a(h_a[1]);
        Mersenne31 inv(h_result[1]);
        Mersenne31 product = a * inv;
        printf("  Verification: %u * %u = %u (should be 1) - %s\n",
               h_a[1], h_result[1], product.value(),
               product.value() == 1 ? "PASS" : "FAIL");
    }

    // Test 7: Fused multiply-add
    CUDA_CHECK(cudaEventRecord(start));
    mersenne31_fma_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_c, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("\n=== Fused Multiply-Add (a * b + c) ===\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d] %u * %u + %u = %u\n", i, h_a[i], h_b[i], h_c[i], h_result[i]);
    }
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Test 8: Negation
    CUDA_CHECK(cudaEventRecord(start));
    mersenne31_neg_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Negation (-a)", h_a, nullptr, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Verify: a + (-a) = 0
    {
        Mersenne31 a(h_a[1]);
        Mersenne31 neg_a(h_result[1]);
        Mersenne31 sum = a + neg_a;
        printf("  Verification: %u + %u = %u (should be 0) - %s\n",
               h_a[1], h_result[1], sum.value(),
               sum.value() == 0 ? "PASS" : "FAIL");
    }

    // Test 9: Repeated squaring (10 rounds)
    const int ROUNDS = 10;
    CUDA_CHECK(cudaEventRecord(start));
    mersenne31_repeated_square_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_result, ROUNDS, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("\n=== Repeated Squaring (a^(2^%d)) ===\n", ROUNDS);
    for (int i = 0; i < 5; i++) {
        printf("  [%d] %u^(2^%d) = %u\n", i, h_a[i], ROUNDS, h_result[i]);
    }
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_result));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_result;

    printf("\n=========================================\n");
    printf("  All Mersenne-31 CUDA tests completed!\n");
    printf("=========================================\n");

    return 0;
}


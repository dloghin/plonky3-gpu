/**
 * @file baby_bear_cuda_example.cu
 * @brief CUDA example demonstrating Baby Bear field arithmetic on GPU
 *
 * This example shows:
 * - Field element creation on GPU
 * - Basic arithmetic (add, sub, mul) in parallel
 * - Exponentiation and inverse computation
 * - Data transfer between host and device
 */

#include "baby_bear.hpp"
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
__global__ void babybear_add_kernel(
    const uint32_t* a,
    const uint32_t* b,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        BabyBear fa(a[idx]);
        BabyBear fb(b[idx]);
        BabyBear fc = fa + fb;
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to perform element-wise field multiplication
 */
__global__ void babybear_mul_kernel(
    const uint32_t* a,
    const uint32_t* b,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        BabyBear fa(a[idx]);
        BabyBear fb(b[idx]);
        BabyBear fc = fa * fb;
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to compute squares of field elements
 */
__global__ void babybear_square_kernel(
    const uint32_t* a,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        BabyBear fa(a[idx]);
        BabyBear fc = fa.square();
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to compute x^5 (injective power map for Baby Bear)
 */
__global__ void babybear_pow5_kernel(
    const uint32_t* a,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        BabyBear fa(a[idx]);
        BabyBear fc = fa.injective_exp_n<5>();
        result[idx] = fc.value();
    }
}

/**
 * @brief Kernel to compute modular inverse
 */
__global__ void babybear_inverse_kernel(
    const uint32_t* a,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        BabyBear fa(a[idx]);
        if (fa.value() != 0) {
            BabyBear fc = fa.inverse();
            result[idx] = fc.value();
        } else {
            result[idx] = 0; // Cannot invert zero
        }
    }
}

/**
 * @brief Kernel for polynomial evaluation: ax^2 + bx + c
 */
__global__ void babybear_poly_eval_kernel(
    const uint32_t* x,
    uint32_t a_coeff,
    uint32_t b_coeff,
    uint32_t c_coeff,
    uint32_t* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        BabyBear fx(x[idx]);
        BabyBear fa(a_coeff);
        BabyBear fb(b_coeff);
        BabyBear fc(c_coeff);

        // Horner's method: a*x^2 + b*x + c = ((a*x) + b)*x + c
        BabyBear res = (fa * fx + fb) * fx + fc;
        result[idx] = res.value();
    }
}

/**
 * @brief Kernel for inner product of two vectors
 */
__global__ void babybear_dot_product_kernel(
    const uint32_t* a,
    const uint32_t* b,
    uint32_t* partial_sums,
    int n
) {
    __shared__ uint32_t sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one product
    BabyBear local_sum = BabyBear::zero_val();
    if (idx < n) {
        BabyBear fa(a[idx]);
        BabyBear fb(b[idx]);
        local_sum = fa * fb;
    }
    sdata[tid] = local_sum.value();
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            BabyBear left(sdata[tid]);
            BabyBear right(sdata[tid + s]);
            sdata[tid] = (left + right).value();
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
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
    printf("  Baby Bear Field CUDA Example\n");
    printf("  Prime: p = 2^31 - 2^27 + 1 = %u\n", BabyBear::PRIME);
    printf("=========================================\n\n");

    // Problem size
    const int N = 1024 * 1024; // 1M elements
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Host arrays
    uint32_t* h_a = new uint32_t[N];
    uint32_t* h_b = new uint32_t[N];
    uint32_t* h_result = new uint32_t[N];

    // Initialize test data
    printf("Initializing %d field elements...\n", N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (static_cast<uint32_t>(i) * 12345) % BabyBear::PRIME;
        h_b[i] = (static_cast<uint32_t>(i) * 67890 + 1) % BabyBear::PRIME;
    }

    // Device arrays
    uint32_t *d_a, *d_b, *d_result;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result, N * sizeof(uint32_t)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms;

    // Test 1: Addition
    CUDA_CHECK(cudaEventRecord(start));
    babybear_add_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Addition (a + b)", h_a, h_b, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Verify first result on CPU
    {
        BabyBear a(h_a[0]), b(h_b[0]);
        BabyBear expected = a + b;
        printf("  Verification: %u + %u = %u (GPU) vs %u (CPU) - %s\n",
               h_a[0], h_b[0], h_result[0], expected.value(),
               h_result[0] == expected.value() ? "PASS" : "FAIL");
    }

    // Test 2: Multiplication
    CUDA_CHECK(cudaEventRecord(start));
    babybear_mul_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Multiplication (a * b)", h_a, h_b, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Verify
    {
        BabyBear a(h_a[1]), b(h_b[1]);
        BabyBear expected = a * b;
        printf("  Verification: %u * %u = %u (GPU) vs %u (CPU) - %s\n",
               h_a[1], h_b[1], h_result[1], expected.value(),
               h_result[1] == expected.value() ? "PASS" : "FAIL");
    }

    // Test 3: Square
    CUDA_CHECK(cudaEventRecord(start));
    babybear_square_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Square (a^2)", h_a, nullptr, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Test 4: Power of 5 (S-box for Baby Bear)
    CUDA_CHECK(cudaEventRecord(start));
    babybear_pow5_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Power x^5 (S-box)", h_a, nullptr, h_result, N);
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Test 5: Inverse (smaller size due to cost)
    const int N_INV = 1024;
    CUDA_CHECK(cudaEventRecord(start));
    babybear_inverse_kernel<<<(N_INV + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_result, N_INV);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N_INV * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_results("Inverse (a^(-1))", h_a, nullptr, h_result, N_INV);
    printf("  Time: %.3f ms (%.2f K ops/sec)\n", ms, N_INV / ms);

    // Verify: a * a^(-1) = 1
    {
        BabyBear a(h_a[1]);
        BabyBear inv(h_result[1]);
        BabyBear product = a * inv;
        printf("  Verification: %u * %u = %u (should be 1) - %s\n",
               h_a[1], h_result[1], product.value(),
               product.value() == 1 ? "PASS" : "FAIL");
    }

    // Test 6: Polynomial evaluation
    uint32_t a_coeff = 3, b_coeff = 5, c_coeff = 7; // 3x^2 + 5x + 7
    CUDA_CHECK(cudaEventRecord(start));
    babybear_poly_eval_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, a_coeff, b_coeff, c_coeff, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("\n=== Polynomial Evaluation: 3x^2 + 5x + 7 ===\n");
    for (int i = 0; i < 5; i++) {
        printf("  f(%u) = %u\n", h_a[i], h_result[i]);
    }
    printf("  Time: %.3f ms (%.2f M ops/sec)\n", ms, N / (ms * 1000.0));

    // Test 7: Dot product
    uint32_t* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, NUM_BLOCKS * sizeof(uint32_t)));

    CUDA_CHECK(cudaEventRecord(start));
    babybear_dot_product_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_partial, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Final reduction on CPU
    uint32_t* h_partial = new uint32_t[NUM_BLOCKS];
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, NUM_BLOCKS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    BabyBear final_sum = BabyBear::zero_val();
    for (int i = 0; i < NUM_BLOCKS; i++) {
        final_sum += BabyBear(h_partial[i]);
    }
    printf("\n=== Dot Product <a, b> ===\n");
    printf("  Result: %u\n", final_sum.value());
    printf("  Time: %.3f ms\n", ms);

    delete[] h_partial;
    CUDA_CHECK(cudaFree(d_partial));

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
    printf("  All Baby Bear CUDA tests completed!\n");
    printf("=========================================\n");

    return 0;
}


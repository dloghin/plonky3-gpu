/**
 * @file mersenne31_poseidon2_cuda.cu
 * @brief CUDA example for Poseidon2 with Mersenne-31 field
 *
 * This example demonstrates:
 * - GPU-accelerated Poseidon2 permutation with Mersenne-31 field
 * - Batch hashing with Poseidon2
 * - Performance comparison between CPU and GPU
 */

#include "poseidon2_cuda.hpp"
#include "mersenne31.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>

using namespace p3_field;
using namespace poseidon2;

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

// Configuration for Mersenne-31 Poseidon2
// For 31-bit fields with D=5, WIDTH=16: rounds_f=8, rounds_p=14
constexpr size_t WIDTH = 16;
constexpr size_t ROUNDS_F = 8;
constexpr size_t ROUNDS_P = 14;
constexpr uint64_t D = 5;
constexpr size_t RATE = 8;  // For sponge hash

using Poseidon2Type = Poseidon2Cuda<Mersenne31, WIDTH, ROUNDS_F, ROUNDS_P, D>;
using SpongeType = Poseidon2SpongeCuda<Mersenne31, WIDTH, RATE, ROUNDS_F, ROUNDS_P, D>;

/**
 * @brief Generate random constants for Poseidon2
 */
void generate_random_constants(
    Mersenne31 initial[ROUNDS_F / 2][WIDTH],
    Mersenne31 terminal[ROUNDS_F / 2][WIDTH],
    Mersenne31 internal[ROUNDS_P]
) {
    std::mt19937 rng(12345);

    for (size_t r = 0; r < ROUNDS_F / 2; ++r) {
        for (size_t i = 0; i < WIDTH; ++i) {
            uint32_t val = rng() % Mersenne31::PRIME;
            initial[r][i] = Mersenne31(val);
            val = rng() % Mersenne31::PRIME;
            terminal[r][i] = Mersenne31(val);
        }
    }

    for (size_t r = 0; r < ROUNDS_P; ++r) {
        uint32_t val = rng() % Mersenne31::PRIME;
        internal[r] = Mersenne31(val);
    }
}

int main() {
    printf("=========================================\n");
    printf("  Poseidon2 CUDA Example - Mersenne-31\n");
    printf("  WIDTH=%zu, D=%lu, ROUNDS_F=%zu, ROUNDS_P=%zu\n",
           WIDTH, D, ROUNDS_F, ROUNDS_P);
    printf("  Prime: 2^31 - 1 = %u\n", Mersenne31::PRIME);
    printf("=========================================\n\n");

    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SMs: %d, Max threads/block: %d\n\n", prop.multiProcessorCount, prop.maxThreadsPerBlock);

    // Generate random constants
    printf("Generating random round constants...\n");
    Mersenne31 initial_constants[ROUNDS_F / 2][WIDTH];
    Mersenne31 terminal_constants[ROUNDS_F / 2][WIDTH];
    Mersenne31 internal_constants[ROUNDS_P];
    generate_random_constants(initial_constants, terminal_constants, internal_constants);

    // Initialize Poseidon2 on host
    Poseidon2Type h_poseidon;
    for (size_t r = 0; r < ROUNDS_F / 2; ++r) {
        for (size_t i = 0; i < WIDTH; ++i) {
            h_poseidon.initial_external_constants[r][i] = initial_constants[r][i];
            h_poseidon.terminal_external_constants[r][i] = terminal_constants[r][i];
        }
    }
    for (size_t r = 0; r < ROUNDS_P; ++r) {
        h_poseidon.internal_constants[r] = internal_constants[r];
    }

    // Copy Poseidon2 to device
    Poseidon2Type* d_poseidon;
    CUDA_CHECK(cudaMalloc(&d_poseidon, sizeof(Poseidon2Type)));
    CUDA_CHECK(cudaMemcpy(d_poseidon, &h_poseidon, sizeof(Poseidon2Type), cudaMemcpyHostToDevice));

    // =========================================
    // Test 1: Single permutation correctness
    // =========================================
    printf("\n=== Test 1: Single Permutation Correctness ===\n");

    // Create test input
    Mersenne31 h_state[WIDTH];
    printf("Input state: ");
    for (size_t i = 0; i < WIDTH; ++i) {
        h_state[i] = Mersenne31(static_cast<uint32_t>(i + 1));
        printf("%u ", h_state[i].value());
    }
    printf("\n");

    // CPU computation
    Mersenne31 cpu_result[WIDTH];
    for (size_t i = 0; i < WIDTH; ++i) cpu_result[i] = h_state[i];
    h_poseidon.permute(cpu_result);
    printf("CPU result:  ");
    for (size_t i = 0; i < WIDTH; ++i) {
        printf("%u ", cpu_result[i].value());
    }
    printf("\n");

    // GPU computation
    Mersenne31* d_state;
    CUDA_CHECK(cudaMalloc(&d_state, WIDTH * sizeof(Mersenne31)));
    CUDA_CHECK(cudaMemcpy(d_state, h_state, WIDTH * sizeof(Mersenne31), cudaMemcpyHostToDevice));
    poseidon2_permute_kernel<<<1, 1>>>(d_poseidon, d_state, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    Mersenne31 gpu_result[WIDTH];
    CUDA_CHECK(cudaMemcpy(gpu_result, d_state, WIDTH * sizeof(Mersenne31), cudaMemcpyDeviceToHost));
    printf("GPU result:  ");
    for (size_t i = 0; i < WIDTH; ++i) {
        printf("%u ", gpu_result[i].value());
    }
    printf("\n");

    // Verify
    bool match = true;
    for (size_t i = 0; i < WIDTH; ++i) {
        if (cpu_result[i].value() != gpu_result[i].value()) {
            match = false;
            break;
        }
    }
    printf("Verification: %s\n", match ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_state));

    // =========================================
    // Test 2: Batch permutation performance
    // =========================================
    printf("\n=== Test 2: Batch Permutation Performance ===\n");

    const size_t N_STATES = 1024 * 1024;  // 1M states
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N_STATES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Processing %zu states...\n", N_STATES);

    // Allocate host memory
    Mersenne31* h_states = new Mersenne31[N_STATES * WIDTH];
    std::mt19937 rng(42);
    for (size_t i = 0; i < N_STATES * WIDTH; ++i) {
        h_states[i] = Mersenne31(rng() % Mersenne31::PRIME);
    }

    // Allocate device memory
    Mersenne31* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, N_STATES * WIDTH * sizeof(Mersenne31)));
    CUDA_CHECK(cudaMemcpy(d_states, h_states, N_STATES * WIDTH * sizeof(Mersenne31), cudaMemcpyHostToDevice));

    // Warm up
    poseidon2_permute_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_poseidon, d_states, N_STATES);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-copy data for timing
    CUDA_CHECK(cudaMemcpy(d_states, h_states, N_STATES * WIDTH * sizeof(Mersenne31), cudaMemcpyHostToDevice));

    // GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    poseidon2_permute_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_poseidon, d_states, N_STATES);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    printf("GPU time: %.3f ms (%.2f M perms/sec)\n", gpu_ms, N_STATES / (gpu_ms * 1000.0));

    // CPU timing (small subset for comparison)
    const size_t CPU_TEST_SIZE = 1000;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < CPU_TEST_SIZE; ++i) {
        h_poseidon.permute(&h_states[i * WIDTH]);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    double cpu_rate = CPU_TEST_SIZE / (cpu_ms * 1000.0);
    printf("CPU time: %.3f ms for %zu perms (%.2f M perms/sec)\n", cpu_ms, CPU_TEST_SIZE, cpu_rate);
    printf("GPU Speedup: %.1fx\n", (N_STATES / (gpu_ms * 1000.0)) / cpu_rate);

    // =========================================
    // Test 3: Sponge hashing
    // =========================================
    printf("\n=== Test 3: Sponge Hashing ===\n");

    const size_t N_HASHES = 100000;
    const size_t INPUT_LEN = 8;

    // Initialize sponge
    SpongeType h_sponge;
    h_sponge.poseidon = h_poseidon;

    SpongeType* d_sponge;
    CUDA_CHECK(cudaMalloc(&d_sponge, sizeof(SpongeType)));
    CUDA_CHECK(cudaMemcpy(d_sponge, &h_sponge, sizeof(SpongeType), cudaMemcpyHostToDevice));

    // Prepare inputs
    Mersenne31* h_inputs = new Mersenne31[N_HASHES * INPUT_LEN];
    Mersenne31* h_outputs = new Mersenne31[N_HASHES * RATE];
    for (size_t i = 0; i < N_HASHES * INPUT_LEN; ++i) {
        h_inputs[i] = Mersenne31(rng() % Mersenne31::PRIME);
    }

    // Allocate device memory
    Mersenne31 *d_inputs, *d_outputs;
    CUDA_CHECK(cudaMalloc(&d_inputs, N_HASHES * INPUT_LEN * sizeof(Mersenne31)));
    CUDA_CHECK(cudaMalloc(&d_outputs, N_HASHES * RATE * sizeof(Mersenne31)));
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs, N_HASHES * INPUT_LEN * sizeof(Mersenne31), cudaMemcpyHostToDevice));

    // GPU hashing
    int hash_blocks = (N_HASHES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaEventRecord(start));
    poseidon2_hash_kernel<<<hash_blocks, BLOCK_SIZE>>>(d_sponge, d_inputs, d_outputs, INPUT_LEN, N_HASHES);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float hash_ms;
    CUDA_CHECK(cudaEventElapsedTime(&hash_ms, start, stop));
    printf("GPU hashing: %.3f ms for %zu hashes (%.2f M hashes/sec)\n",
           hash_ms, N_HASHES, N_HASHES / (hash_ms * 1000.0));

    // Verify first hash on CPU
    CUDA_CHECK(cudaMemcpy(h_outputs, d_outputs, N_HASHES * RATE * sizeof(Mersenne31), cudaMemcpyDeviceToHost));

    Mersenne31 cpu_hash[RATE];
    h_sponge.hash(h_inputs, INPUT_LEN, cpu_hash);

    bool hash_match = true;
    for (size_t i = 0; i < RATE; ++i) {
        if (cpu_hash[i].value() != h_outputs[i].value()) {
            hash_match = false;
            break;
        }
    }
    printf("Hash verification: %s\n", hash_match ? "PASS" : "FAIL");

    // Print sample output
    printf("Sample hash output: [");
    for (size_t i = 0; i < RATE; ++i) {
        printf("%u%s", h_outputs[i].value(), i < RATE - 1 ? ", " : "");
    }
    printf("]\n");

    // =========================================
    // Test 4: Different batch sizes
    // =========================================
    printf("\n=== Test 4: Performance vs Batch Size ===\n");

    size_t batch_sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    int num_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    for (int s = 0; s < num_sizes; ++s) {
        size_t batch = batch_sizes[s];
        int blocks = (batch + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Reset data
        CUDA_CHECK(cudaMemcpy(d_states, h_states, batch * WIDTH * sizeof(Mersenne31), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start));
        poseidon2_permute_kernel<<<blocks, BLOCK_SIZE>>>(d_poseidon, d_states, batch);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("  Batch %7zu: %.3f ms (%.2f M perms/sec)\n", batch, ms, batch / (ms * 1000.0));
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_poseidon));
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_sponge));
    CUDA_CHECK(cudaFree(d_inputs));
    CUDA_CHECK(cudaFree(d_outputs));
    delete[] h_states;
    delete[] h_inputs;
    delete[] h_outputs;

    printf("\n=========================================\n");
    printf("  All Mersenne-31 Poseidon2 tests done!\n");
    printf("=========================================\n");

    return 0;
}



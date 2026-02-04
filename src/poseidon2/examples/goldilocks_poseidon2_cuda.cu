/**
 * @file goldilocks_poseidon2_cuda.cu
 * @brief CUDA example for Poseidon2 with Goldilocks field
 *
 * This example demonstrates:
 * - GPU-accelerated Poseidon2 permutation
 * - Batch hashing with Poseidon2
 * - Performance comparison between CPU and GPU
 */

#include "poseidon2_cuda.hpp"
#include "goldilocks.hpp"
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

// Configuration for Goldilocks Poseidon2
constexpr size_t WIDTH = 12;
constexpr size_t ROUNDS_F = 8;
constexpr size_t ROUNDS_P = 22;
constexpr uint64_t D = 7;
constexpr size_t RATE = 8;  // For sponge hash

using Poseidon2Type = Poseidon2Cuda<Goldilocks, WIDTH, ROUNDS_F, ROUNDS_P, D>;
using SpongeType = Poseidon2SpongeCuda<Goldilocks, WIDTH, RATE, ROUNDS_F, ROUNDS_P, D>;

/**
 * @brief Generate random constants for Poseidon2
 */
void generate_random_constants(
    Goldilocks initial[ROUNDS_F / 2][WIDTH],
    Goldilocks terminal[ROUNDS_F / 2][WIDTH],
    Goldilocks internal[ROUNDS_P]
) {
    std::mt19937_64 rng(12345);

    for (size_t r = 0; r < ROUNDS_F / 2; ++r) {
        for (size_t i = 0; i < WIDTH; ++i) {
            uint64_t val = rng() % Goldilocks::PRIME;
            initial[r][i] = Goldilocks(val);
            val = rng() % Goldilocks::PRIME;
            terminal[r][i] = Goldilocks(val);
        }
    }

    for (size_t r = 0; r < ROUNDS_P; ++r) {
        uint64_t val = rng() % Goldilocks::PRIME;
        internal[r] = Goldilocks(val);
    }
}

/**
 * @brief CPU reference implementation of Poseidon2 permutation
 */
void poseidon2_cpu(const Poseidon2Type& poseidon, Goldilocks* state) {
    Goldilocks temp[WIDTH];
    for (size_t i = 0; i < WIDTH; ++i) {
        temp[i] = state[i];
    }
    poseidon.permute(temp);
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = temp[i];
    }
}

int main() {
    printf("=========================================\n");
    printf("  Poseidon2 CUDA Example - Goldilocks\n");
    printf("  WIDTH=%zu, D=%lu, ROUNDS_F=%zu, ROUNDS_P=%zu\n",
           WIDTH, D, ROUNDS_F, ROUNDS_P);
    printf("=========================================\n\n");

    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (Compute %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Generate random constants
    printf("Generating random round constants...\n");
    Goldilocks initial_constants[ROUNDS_F / 2][WIDTH];
    Goldilocks terminal_constants[ROUNDS_F / 2][WIDTH];
    Goldilocks internal_constants[ROUNDS_P];
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
    Goldilocks h_state[WIDTH];
    printf("Input state: ");
    for (size_t i = 0; i < WIDTH; ++i) {
        h_state[i] = Goldilocks(static_cast<uint64_t>(i + 1));
        printf("%lu ", h_state[i].value());
    }
    printf("\n");

    // CPU computation
    Goldilocks cpu_result[WIDTH];
    for (size_t i = 0; i < WIDTH; ++i) cpu_result[i] = h_state[i];
    h_poseidon.permute(cpu_result);
    printf("CPU result:  ");
    for (size_t i = 0; i < WIDTH; ++i) {
        printf("%lu ", cpu_result[i].as_canonical_u64());
    }
    printf("\n");

    // GPU computation
    Goldilocks* d_state;
    CUDA_CHECK(cudaMalloc(&d_state, WIDTH * sizeof(Goldilocks)));
    CUDA_CHECK(cudaMemcpy(d_state, h_state, WIDTH * sizeof(Goldilocks), cudaMemcpyHostToDevice));
    poseidon2_permute_kernel<<<1, 1>>>(d_poseidon, d_state, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    Goldilocks gpu_result[WIDTH];
    CUDA_CHECK(cudaMemcpy(gpu_result, d_state, WIDTH * sizeof(Goldilocks), cudaMemcpyDeviceToHost));
    printf("GPU result:  ");
    for (size_t i = 0; i < WIDTH; ++i) {
        printf("%lu ", gpu_result[i].as_canonical_u64());
    }
    printf("\n");

    // Verify
    bool match = true;
    for (size_t i = 0; i < WIDTH; ++i) {
        if (cpu_result[i].as_canonical_u64() != gpu_result[i].as_canonical_u64()) {
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
    Goldilocks* h_states = new Goldilocks[N_STATES * WIDTH];
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < N_STATES * WIDTH; ++i) {
        h_states[i] = Goldilocks(rng() % Goldilocks::PRIME);
    }

    // Allocate device memory
    Goldilocks* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, N_STATES * WIDTH * sizeof(Goldilocks)));
    CUDA_CHECK(cudaMemcpy(d_states, h_states, N_STATES * WIDTH * sizeof(Goldilocks), cudaMemcpyHostToDevice));

    // Warm up
    poseidon2_permute_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_poseidon, d_states, N_STATES);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-copy data for timing
    CUDA_CHECK(cudaMemcpy(d_states, h_states, N_STATES * WIDTH * sizeof(Goldilocks), cudaMemcpyHostToDevice));

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
    Goldilocks* h_inputs = new Goldilocks[N_HASHES * INPUT_LEN];
    Goldilocks* h_outputs = new Goldilocks[N_HASHES * RATE];
    for (size_t i = 0; i < N_HASHES * INPUT_LEN; ++i) {
        h_inputs[i] = Goldilocks(rng() % Goldilocks::PRIME);
    }

    // Allocate device memory
    Goldilocks *d_inputs, *d_outputs;
    CUDA_CHECK(cudaMalloc(&d_inputs, N_HASHES * INPUT_LEN * sizeof(Goldilocks)));
    CUDA_CHECK(cudaMalloc(&d_outputs, N_HASHES * RATE * sizeof(Goldilocks)));
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs, N_HASHES * INPUT_LEN * sizeof(Goldilocks), cudaMemcpyHostToDevice));

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
    CUDA_CHECK(cudaMemcpy(h_outputs, d_outputs, N_HASHES * RATE * sizeof(Goldilocks), cudaMemcpyDeviceToHost));

    Goldilocks cpu_hash[RATE];
    h_sponge.hash(h_inputs, INPUT_LEN, cpu_hash);

    bool hash_match = true;
    for (size_t i = 0; i < RATE; ++i) {
        if (cpu_hash[i].as_canonical_u64() != h_outputs[i].as_canonical_u64()) {
            hash_match = false;
            break;
        }
    }
    printf("Hash verification: %s\n", hash_match ? "PASS" : "FAIL");

    // Print sample output
    printf("Sample hash output: [");
    for (size_t i = 0; i < RATE; ++i) {
        printf("%lu%s", h_outputs[i].as_canonical_u64(), i < RATE - 1 ? ", " : "");
    }
    printf("]\n");

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
    printf("  All Goldilocks Poseidon2 tests done!\n");
    printf("=========================================\n");

    return 0;
}



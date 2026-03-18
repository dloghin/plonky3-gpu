/**
 * @file merkle_tree_cuda.cu
 * @brief CUDA Merkle tree correctness and performance test.
 *
 * Tests:
 *   1. Single-matrix: GPU tree root matches CPU tree root.
 *   2. Multi-matrix mixed height: GPU matches CPU.
 *   3. cap_height > 0: GPU produces correct-size cap.
 *   4. Performance comparison for large matrices.
 *
 * Uses BabyBear field with Poseidon2 (WIDTH=16, RATE=8, DIGEST_ELEMS=8).
 * Round constants are initialized to deterministic test values (not
 * cryptographically secure, but sufficient for correctness testing).
 */

#include "merkle_tree_cuda.hpp"
#include "merkle_tree.hpp"
#include "merkle_tree_mmcs.hpp"
#include "poseidon2_cuda.hpp"
#include "poseidon2.hpp"
#include "padding_free_sponge.hpp"
#include "truncated_permutation.hpp"
#include "baby_bear.hpp"
#include "dense_matrix.hpp"

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <random>
#include <cstring>

using namespace p3_merkle_tree;
using namespace p3_matrix;
using namespace p3_field;
using namespace poseidon2;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
constexpr size_t WIDTH    = 16;
constexpr size_t RATE     = 8;
constexpr size_t DIGEST   = 8;   // DIGEST_ELEMS
constexpr size_t ROUNDS_F = 8;
constexpr size_t ROUNDS_P = 14;
constexpr uint64_t D_EXP  = 5;

using BB        = BabyBear;
using PosType   = Poseidon2Cuda<BB, WIDTH, ROUNDS_F, ROUNDS_P, D_EXP>;

// ---------------------------------------------------------------------------
// Initialize Poseidon2Cuda with deterministic test constants
// ---------------------------------------------------------------------------
static PosType make_test_poseidon() {
    PosType p;
    std::mt19937 rng(0xdeadbeef);
    for (size_t r = 0; r < ROUNDS_F / 2; ++r) {
        for (size_t i = 0; i < WIDTH; ++i) {
            p.initial_external_constants[r][i]  = BB(rng() % BB::PRIME);
            p.terminal_external_constants[r][i] = BB(rng() % BB::PRIME);
        }
    }
    for (size_t r = 0; r < ROUNDS_P; ++r) {
        p.internal_constants[r] = BB(rng() % BB::PRIME);
    }
    return p;
}

// ---------------------------------------------------------------------------
// Build a CPU MerkleTree using the same Poseidon2 constants via the
// CPU fallback path in build_merkle_tree_cuda.
// ---------------------------------------------------------------------------
static MerkleTree<BB, BB, DIGEST> build_cpu_tree(
    std::vector<RowMajorMatrix<BB>> matrices,
    const PosType& poseidon,
    size_t cap_height)
{
    // Use the CPU fallback (same function, non-CUDA compilation path not
    // available here since we ARE in a .cu file).  We call build_merkle_tree_cuda
    // but on CPU by copying matrices.  To get a true CPU reference, we run the
    // CUDA version on CPU data with a synchronous approach — but actually we
    // just run build_merkle_tree_cuda which on CUDA hardware uses the GPU.
    //
    // For the correctness test we want a known-good reference.  We compute it
    // by running the same build_merkle_tree_cuda call with the same poseidon and
    // comparing two independent calls (they must be equal).
    //
    // True cross-validation against the CPU PaddingFreeSponge is done by
    // rebuilding with the CPU hasher/compressor (see test_cpu_vs_gpu below).
    return build_merkle_tree_cuda<BB, WIDTH, RATE, DIGEST, ROUNDS_F, ROUNDS_P, D_EXP>(
        std::move(matrices), poseidon, cap_height);
}

// ---------------------------------------------------------------------------
// Compare two MerkleCap values
// ---------------------------------------------------------------------------
static bool caps_equal(const MerkleCap<BB, DIGEST>& a,
                       const MerkleCap<BB, DIGEST>& b)
{
    return a == b;
}

// ---------------------------------------------------------------------------
// Create a Poseidon2Wrapper that the CPU PaddingFreeSponge / TruncatedPermutation
// expect (expose permute_mut(std::array<BB,WIDTH>&)).
// ---------------------------------------------------------------------------
struct CPUPermWrapper {
    PosType poseidon;
    void permute_mut(std::array<BB, WIDTH>& state) const {
        poseidon.permute(state.data());
    }
};

using CPUSponge  = p3_symmetric::PaddingFreeSponge<CPUPermWrapper, BB, WIDTH, RATE, DIGEST>;
using CPUTrunc   = p3_symmetric::TruncatedPermutation<CPUPermWrapper, BB, 2, DIGEST, WIDTH>;
using CPUMmcs    = MerkleTreeMmcs<BB, BB, CPUSponge, CPUTrunc, DIGEST>;

// ---------------------------------------------------------------------------
// Helper: make a random BabyBear matrix
// ---------------------------------------------------------------------------
static RowMajorMatrix<BB> make_random_matrix(size_t rows, size_t cols,
                                              std::mt19937& rng)
{
    std::vector<BB> vals(rows * cols);
    for (auto& v : vals) v = BB(rng() % BB::PRIME);
    return RowMajorMatrix<BB>(std::move(vals), cols);
}

// ---------------------------------------------------------------------------
// Test: CUDA tree root equals CPU tree root
// ---------------------------------------------------------------------------
static bool test_cuda_matches_cpu(const char* label,
                                   std::vector<RowMajorMatrix<BB>> matrices_cuda,
                                   std::vector<RowMajorMatrix<BB>> matrices_cpu,
                                   const PosType& poseidon,
                                   size_t cap_height)
{
    // Build CPU tree using PaddingFreeSponge / TruncatedPermutation
    CPUPermWrapper wrapper{poseidon};
    CPUSponge sponge{wrapper};
    CPUTrunc  trunc{wrapper};
    CPUMmcs   mmcs(std::move(sponge), std::move(trunc), cap_height);
    auto [cpu_cap, cpu_tree] = mmcs.commit(std::move(matrices_cpu));

    // Build GPU tree
    MerkleTree<BB, BB, DIGEST> gpu_tree =
        build_merkle_tree_cuda<BB, WIDTH, RATE, DIGEST, ROUNDS_F, ROUNDS_P, D_EXP>(
            std::move(matrices_cuda), poseidon, cap_height);
    MerkleCap<BB, DIGEST> gpu_cap = get_cap(gpu_tree);

    bool ok = caps_equal(cpu_cap, gpu_cap);
    printf("  %-50s %s\n", label, ok ? "PASS" : "FAIL");
    if (!ok) {
        printf("    CPU cap[0][0] = %u\n", cpu_cap.cap[0][0].value());
        printf("    GPU cap[0][0] = %u\n", gpu_cap.cap[0][0].value());
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Test: opening proofs from GPU tree verify correctly
// ---------------------------------------------------------------------------
static bool test_openings(const char* label,
                           std::vector<RowMajorMatrix<BB>> matrices,
                           const PosType& poseidon,
                           size_t cap_height)
{
    // Build GPU tree
    // Keep a copy of dimensions for verify_batch
    std::vector<p3_matrix::Dimensions> dims;
    for (const auto& m : matrices) dims.push_back(m.dimensions());

    // Build a CPU MMCS instance for verify_batch
    CPUPermWrapper wrapper{poseidon};
    CPUSponge sponge{wrapper};
    CPUTrunc  trunc{wrapper};
    CPUMmcs   mmcs(std::move(sponge), std::move(trunc), cap_height);

    MerkleTree<BB, BB, DIGEST> gpu_tree =
        build_merkle_tree_cuda<BB, WIDTH, RATE, DIGEST, ROUNDS_F, ROUNDS_P, D_EXP>(
            std::move(matrices), poseidon, cap_height);
    MerkleCap<BB, DIGEST> gpu_cap = get_cap(gpu_tree);

    const size_t max_height = gpu_tree.digest_layers[0].size();
    bool all_ok = true;
    for (size_t i = 0; i < max_height; ++i) {
        // We need a CPU MMCS instance that owns the tree — but verify_batch
        // doesn't require prover data, only cap + dims + opening.
        // Build opening from the GPU tree's leaf matrices.
        MerkleTreeMmcs<BB, BB, CPUSponge, CPUTrunc, DIGEST> verify_mmcs(
            CPUSponge{CPUPermWrapper{poseidon}},
            CPUTrunc{CPUPermWrapper{poseidon}},
            cap_height);

        auto opening = verify_mmcs.open_batch(i, gpu_tree);
        bool ok = verify_mmcs.verify_batch(gpu_cap, dims, i, opening);
        if (!ok) {
            all_ok = false;
            printf("    Opening at index %zu FAILED\n", i);
            break;
        }
    }
    printf("  %-50s %s\n", label, all_ok ? "PASS" : "FAIL");
    return all_ok;
}

// ---------------------------------------------------------------------------
// Performance test
// ---------------------------------------------------------------------------
static void perf_test(size_t num_rows, size_t num_cols, const PosType& poseidon)
{
    printf("\n  Performance: %zu rows x %zu cols\n", num_rows, num_cols);

    std::mt19937 rng(42);
    RowMajorMatrix<BB> mat = make_random_matrix(num_rows, num_cols, rng);

    // GPU timing
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // Warm-up
    {
        auto mat_copy = mat;
        build_merkle_tree_cuda<BB, WIDTH, RATE, DIGEST, ROUNDS_F, ROUNDS_P, D_EXP>(
            {std::move(mat_copy)}, poseidon, 0);
    }

    cudaEventRecord(ev_start);
    {
        auto mat_copy = mat;
        build_merkle_tree_cuda<BB, WIDTH, RATE, DIGEST, ROUNDS_F, ROUNDS_P, D_EXP>(
            {std::move(mat_copy)}, poseidon, 0);
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);

    // CPU timing
    CPUPermWrapper wrapper{poseidon};
    CPUSponge sponge{wrapper};
    CPUTrunc  trunc{wrapper};
    CPUMmcs   mmcs(std::move(sponge), std::move(trunc), 0);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    {
        auto mat_copy = mat;
        mmcs.commit({std::move(mat_copy)});
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    printf("    GPU: %.2f ms\n", gpu_ms);
    printf("    CPU: %.2f ms\n", cpu_ms);
    if (gpu_ms > 0.0f) {
        printf("    Speedup: %.1fx\n", cpu_ms / gpu_ms);
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    // Check GPU
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    if (n_devices == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (Compute %d.%d)\n\n", prop.name, prop.major, prop.minor);

    printf("=== CUDA Merkle Tree Tests ===\n");
    printf("  BabyBear, WIDTH=%zu, RATE=%zu, DIGEST=%zu\n\n",
           WIDTH, RATE, DIGEST);

    const PosType poseidon = make_test_poseidon();
    std::mt19937 rng(99);
    int passed = 0, total = 0;

    // -------------------------------------------------------------------------
    printf("--- Correctness: CUDA root == CPU root ---\n");
    // -------------------------------------------------------------------------

    auto run_match = [&](const char* label,
                          std::vector<RowMajorMatrix<BB>> mats,
                          size_t cap_height) {
        // Make a copy for CPU
        std::vector<RowMajorMatrix<BB>> cpu_mats;
        for (const auto& m : mats) {
            cpu_mats.emplace_back(m.values, m.width());
        }
        ++total;
        if (test_cuda_matches_cpu(label, std::move(mats), std::move(cpu_mats),
                                  poseidon, cap_height))
            ++passed;
    };

    // Single row
    run_match("single row (1x3)",
              {make_random_matrix(1, 3, rng)}, 0);

    // Single matrix, small
    run_match("single matrix (4x3)",
              {make_random_matrix(4, 3, rng)}, 0);

    // Single matrix, cap_height=0
    run_match("single matrix (8x2, cap=0)",
              {make_random_matrix(8, 2, rng)}, 0);

    // Single matrix, cap_height=2
    run_match("single matrix (8x4, cap=2)",
              {make_random_matrix(8, 4, rng)}, 2);

    // Single matrix, 16 columns (FRI leaf case)
    run_match("single matrix (16x16, cap=0)",
              {make_random_matrix(16, 16, rng)}, 0);

    // Two matrices, same height
    {
        RowMajorMatrix<BB> m1 = make_random_matrix(4, 2, rng);
        RowMajorMatrix<BB> m2 = make_random_matrix(4, 3, rng);
        RowMajorMatrix<BB> m1c(m1.values, m1.width()), m2c(m2.values, m2.width());
        ++total;
        std::vector<RowMajorMatrix<BB>> gpu_mats, cpu_mats;
        gpu_mats.push_back(std::move(m1));  gpu_mats.push_back(std::move(m2));
        cpu_mats.push_back(std::move(m1c)); cpu_mats.push_back(std::move(m2c));
        if (test_cuda_matches_cpu("two matrices same height (4x2 + 4x3)",
                                  std::move(gpu_mats), std::move(cpu_mats),
                                  poseidon, 0))
            ++passed;
    }

    // Two matrices, mixed height
    {
        RowMajorMatrix<BB> tall  = make_random_matrix(8, 2, rng);
        RowMajorMatrix<BB> short_ = make_random_matrix(4, 3, rng);
        RowMajorMatrix<BB> tallc(tall.values, tall.width());
        RowMajorMatrix<BB> shortc(short_.values, short_.width());
        ++total;
        std::vector<RowMajorMatrix<BB>> gpu_mats, cpu_mats;
        gpu_mats.push_back(std::move(tall));   gpu_mats.push_back(std::move(short_));
        cpu_mats.push_back(std::move(tallc));  cpu_mats.push_back(std::move(shortc));
        if (test_cuda_matches_cpu("mixed height (8x2 tall + 4x3 short)",
                                  std::move(gpu_mats), std::move(cpu_mats),
                                  poseidon, 0))
            ++passed;
    }

    // Three matrices, mixed height
    {
        RowMajorMatrix<BB> m16 = make_random_matrix(16, 2, rng);
        RowMajorMatrix<BB> m8  = make_random_matrix(8,  3, rng);
        RowMajorMatrix<BB> m4  = make_random_matrix(4,  2, rng);
        RowMajorMatrix<BB> m16c(m16.values, m16.width());
        RowMajorMatrix<BB> m8c(m8.values,   m8.width());
        RowMajorMatrix<BB> m4c(m4.values,   m4.width());
        ++total;
        std::vector<RowMajorMatrix<BB>> gpu_mats, cpu_mats;
        gpu_mats.push_back(std::move(m16));  gpu_mats.push_back(std::move(m8));
        gpu_mats.push_back(std::move(m4));
        cpu_mats.push_back(std::move(m16c)); cpu_mats.push_back(std::move(m8c));
        cpu_mats.push_back(std::move(m4c));
        if (test_cuda_matches_cpu("three matrices mixed (16x2 + 8x3 + 4x2)",
                                  std::move(gpu_mats), std::move(cpu_mats),
                                  poseidon, 0))
            ++passed;
    }

    // Large matrix (2^10 rows) - correctness
    run_match("large matrix (1024x16, cap=0)",
              {make_random_matrix(1024, 16, rng)}, 0);

    // Large matrix with cap
    run_match("large matrix (1024x8, cap=3)",
              {make_random_matrix(1024, 8, rng)}, 3);

    // -------------------------------------------------------------------------
    printf("\n--- Opening proof verification ---\n");
    // -------------------------------------------------------------------------

    auto run_open = [&](const char* label,
                         std::vector<RowMajorMatrix<BB>> mats,
                         size_t cap_height) {
        ++total;
        if (test_openings(label, std::move(mats), poseidon, cap_height))
            ++passed;
    };

    run_open("single matrix openings (8x3, cap=0)",
             {make_random_matrix(8, 3, rng)}, 0);

    run_open("single matrix openings (8x3, cap=2)",
             {make_random_matrix(8, 3, rng)}, 2);

    {
        RowMajorMatrix<BB> tall  = make_random_matrix(8, 2, rng);
        RowMajorMatrix<BB> short_ = make_random_matrix(4, 3, rng);
        std::vector<RowMajorMatrix<BB>> mats;
        mats.push_back(std::move(tall));
        mats.push_back(std::move(short_));
        run_open("mixed height openings (8x2 + 4x3, cap=0)", std::move(mats), 0);
    }

    // -------------------------------------------------------------------------
    printf("\n--- Results ---\n");
    printf("  %d / %d tests passed\n\n", passed, total);
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    printf("--- Performance ---\n");
    // -------------------------------------------------------------------------
    perf_test(1 << 14, 16, poseidon);   // 16384 rows × 16 cols
    perf_test(1 << 16, 16, poseidon);   // 65536 rows × 16 cols
    perf_test(1 << 18, 8,  poseidon);   // 262144 rows × 8 cols (FRI folding)

    printf("\n=== Done ===\n");
    return (passed == total) ? 0 : 1;
}

/**
 * @file test_keccak_cuda.cu
 * @brief CUDA test for device-side Keccak-f[1600].
 */

#include <gtest/gtest.h>

#include "cuda_compat.hpp"
#include "keccak.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

using namespace p3_symmetric;

namespace {

P3_GLOBAL void keccak_f_kernel(const uint64_t* in_state, uint64_t* out_state) {
    uint64_t st[KECCAK_STATE_LANES]{};
    for (size_t i = 0; i < KECCAK_STATE_LANES; ++i) {
        st[i] = in_state[i];
    }

    keccak_detail::keccak_f1600(st);

    for (size_t i = 0; i < KECCAK_STATE_LANES; ++i) {
        out_state[i] = st[i];
    }
}

} // namespace

TEST(KeccakCuda, KernelLaunchMatchesHostPermutation) {
    int device_count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    std::array<uint64_t, KECCAK_STATE_LANES> in_state{};
    for (size_t i = 0; i < in_state.size(); ++i) {
        in_state[i] = static_cast<uint64_t>(i * i + 3 * i + 7);
    }

    auto expected = in_state;
    KeccakF host_perm;
    host_perm.permute_mut(expected);

    uint64_t* d_in = nullptr;
    uint64_t* d_out = nullptr;

    try {
        P3_CUDA_CHECK(cudaMalloc(&d_in, sizeof(uint64_t) * KECCAK_STATE_LANES));
        P3_CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint64_t) * KECCAK_STATE_LANES));

        P3_CUDA_CHECK(cudaMemcpy(
            d_in, in_state.data(), sizeof(uint64_t) * KECCAK_STATE_LANES, cudaMemcpyHostToDevice));

        keccak_f_kernel<<<1, 1>>>(d_in, d_out);
        P3_CUDA_CHECK(cudaGetLastError());
        P3_CUDA_CHECK(cudaDeviceSynchronize());

        std::array<uint64_t, KECCAK_STATE_LANES> got{};
        P3_CUDA_CHECK(cudaMemcpy(
            got.data(), d_out, sizeof(uint64_t) * KECCAK_STATE_LANES, cudaMemcpyDeviceToHost));

        EXPECT_EQ(got, expected);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        if (msg.find("unsupported toolchain") != std::string::npos) {
            GTEST_SKIP() << "CUDA driver/toolchain mismatch: " << msg;
        }
        throw;
    }

    if (d_in) P3_CUDA_CHECK(cudaFree(d_in));
    if (d_out) P3_CUDA_CHECK(cudaFree(d_out));
}

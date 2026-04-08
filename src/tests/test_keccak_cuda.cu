/**
 * @file test_keccak_cuda.cu
 * @brief CUDA test for device-side Keccak-f[1600].
 */

#include <gtest/gtest.h>

#include "cuda_compat.hpp"
#include "keccak.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

using namespace p3_symmetric;

namespace {

struct CudaFreeDeleter {
    void operator()(void* p) const noexcept {
        if (p) {
            (void)cudaFree(p);
        }
    }
};
using DeviceBuffer = std::unique_ptr<void, CudaFreeDeleter>;

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

    DeviceBuffer d_in;
    DeviceBuffer d_out;

    try {
        void* p_in = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_in, sizeof(uint64_t) * KECCAK_STATE_LANES));
        d_in.reset(p_in);

        void* p_out = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_out, sizeof(uint64_t) * KECCAK_STATE_LANES));
        d_out.reset(p_out);

        auto* d_in_u64 = static_cast<uint64_t*>(d_in.get());
        auto* d_out_u64 = static_cast<uint64_t*>(d_out.get());

        P3_CUDA_CHECK(cudaMemcpy(
            d_in_u64, in_state.data(), sizeof(uint64_t) * KECCAK_STATE_LANES, cudaMemcpyHostToDevice));

        keccak_f_kernel<<<1, 1>>>(d_in_u64, d_out_u64);
        P3_CUDA_CHECK(cudaGetLastError());
        P3_CUDA_CHECK(cudaDeviceSynchronize());

        std::array<uint64_t, KECCAK_STATE_LANES> got{};
        P3_CUDA_CHECK(cudaMemcpy(
            got.data(), d_out_u64, sizeof(uint64_t) * KECCAK_STATE_LANES, cudaMemcpyDeviceToHost));

        EXPECT_EQ(got, expected);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("unsupported toolchain") != std::string::npos) {
            GTEST_SKIP() << "CUDA driver/toolchain mismatch: " << msg;
        }
        throw;
    }
}

/**
 * @file test_sha256_cuda.cu
 * @brief CUDA test for device-side Sha256Compress.
 */

#include <gtest/gtest.h>

#include "cuda_compat.hpp"
#include "sha256.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

using p3_cuda_compat::DeviceBuffer;
using namespace p3_symmetric;

namespace {

P3_GLOBAL void sha256_compress_kernel(const uint8_t* left, const uint8_t* right, uint8_t* out) {
    Sha256Compress compress;
    compress.compress_raw(left, right, out);
}

} // namespace

TEST(Sha256Cuda, KernelLaunchMatchesHostResult) {
    int device_count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    std::array<uint8_t, 32> left{};
    std::array<uint8_t, 32> right{};
    right[0] = 0x80u;
    right[30] = 0x01u;

    const Sha256 sha;
    const auto expected = sha.hash_iter(left.data(), left.size());

    DeviceBuffer buf_left;
    DeviceBuffer buf_right;
    DeviceBuffer buf_out;

    try {
        void* p_left = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_left, 32));
        buf_left.reset(p_left);

        void* p_right = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_right, 32));
        buf_right.reset(p_right);

        void* p_out = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_out, 32));
        buf_out.reset(p_out);

        auto* d_left = static_cast<uint8_t*>(buf_left.get());
        auto* d_right = static_cast<uint8_t*>(buf_right.get());
        auto* d_out = static_cast<uint8_t*>(buf_out.get());

        P3_CUDA_CHECK(cudaMemcpy(d_left, left.data(), 32, cudaMemcpyHostToDevice));
        P3_CUDA_CHECK(cudaMemcpy(d_right, right.data(), 32, cudaMemcpyHostToDevice));

        sha256_compress_kernel<<<1, 1>>>(d_left, d_right, d_out);
        P3_CUDA_CHECK(cudaGetLastError());
        P3_CUDA_CHECK(cudaDeviceSynchronize());

        std::array<uint8_t, 32> got{};
        P3_CUDA_CHECK(cudaMemcpy(got.data(), d_out, 32, cudaMemcpyDeviceToHost));
        EXPECT_EQ(got, expected);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("unsupported toolchain") != std::string::npos) {
            GTEST_SKIP() << "CUDA driver/toolchain mismatch: " << msg;
        }
        throw;
    }
}

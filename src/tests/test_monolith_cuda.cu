/**
 * @file test_monolith_cuda.cu
 * @brief CUDA test for device-side Monolith permutation.
 */

#include <gtest/gtest.h>

#include "cuda_compat.hpp"
#include "mersenne31.hpp"
#include "monolith.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

using p3_cuda_compat::DeviceBuffer;
using monolith::MonolithMersenne31;
using p3_field::Mersenne31;

namespace {

P3_GLOBAL void monolith_permute_kernel(const uint32_t* in_state, uint32_t* out_state) {
    constexpr size_t WIDTH = 16;
    MonolithMersenne31<> monolith;
    std::array<Mersenne31, WIDTH> state{};
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = Mersenne31(in_state[i]);
    }

    monolith.permute_mut(state);

    for (size_t i = 0; i < WIDTH; ++i) {
        out_state[i] = state[i].value();
    }
}

} // namespace

TEST(MonolithCuda, KernelLaunchMatchesHostPermutation) {
    int device_count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    std::array<Mersenne31, 16> in_state{};
    for (uint32_t i = 0; i < in_state.size(); ++i) {
        in_state[i] = Mersenne31(i);
    }

    auto expected = in_state;
    MonolithMersenne31<> host_monolith;
    host_monolith.permute_mut(expected);

    std::array<uint32_t, 16> in_words{};
    std::array<uint32_t, 16> got_words{};
    std::array<uint32_t, 16> expected_words{};
    for (size_t i = 0; i < in_state.size(); ++i) {
        in_words[i] = in_state[i].value();
        expected_words[i] = expected[i].value();
    }

    DeviceBuffer buf_in;
    DeviceBuffer buf_out;

    try {
        void* p_in = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_in, sizeof(uint32_t) * in_words.size()));
        buf_in.reset(p_in);

        void* p_out = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_out, sizeof(uint32_t) * got_words.size()));
        buf_out.reset(p_out);

        auto* d_in = static_cast<uint32_t*>(buf_in.get());
        auto* d_out = static_cast<uint32_t*>(buf_out.get());

        P3_CUDA_CHECK(cudaMemcpy(
            d_in, in_words.data(), sizeof(uint32_t) * in_words.size(), cudaMemcpyHostToDevice));

        monolith_permute_kernel<<<1, 1>>>(d_in, d_out);
        P3_CUDA_CHECK(cudaGetLastError());
        P3_CUDA_CHECK(cudaDeviceSynchronize());

        P3_CUDA_CHECK(cudaMemcpy(
            got_words.data(), d_out, sizeof(uint32_t) * got_words.size(), cudaMemcpyDeviceToHost));

        EXPECT_EQ(got_words, expected_words);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("unsupported toolchain") != std::string::npos) {
            GTEST_SKIP() << "CUDA driver/toolchain mismatch: " << msg;
        }
        throw;
    }
}

TEST(MonolithCuda, KernelMatchesRustReferenceVector) {
    int device_count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }

    std::array<uint32_t, 16> in_words{};
    for (uint32_t i = 0; i < in_words.size(); ++i) {
        in_words[i] = i;
    }

    const std::array<uint32_t, 16> expected_words = {
        609156607u, 290107110u, 1900746598u, 1734707571u,
        2050994835u, 1648553244u, 1307647296u, 1941164548u,
        1707113065u, 1477714255u, 1170160793u, 93800695u,
        769879348u, 375548503u, 1989726444u, 1349325635u,
    };

    std::array<uint32_t, 16> got_words{};

    DeviceBuffer buf_in;
    DeviceBuffer buf_out;

    try {
        void* p_in = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_in, sizeof(uint32_t) * in_words.size()));
        buf_in.reset(p_in);

        void* p_out = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_out, sizeof(uint32_t) * got_words.size()));
        buf_out.reset(p_out);

        auto* d_in = static_cast<uint32_t*>(buf_in.get());
        auto* d_out = static_cast<uint32_t*>(buf_out.get());

        P3_CUDA_CHECK(cudaMemcpy(
            d_in, in_words.data(), sizeof(uint32_t) * in_words.size(), cudaMemcpyHostToDevice));

        monolith_permute_kernel<<<1, 1>>>(d_in, d_out);
        P3_CUDA_CHECK(cudaGetLastError());
        P3_CUDA_CHECK(cudaDeviceSynchronize());

        P3_CUDA_CHECK(cudaMemcpy(
            got_words.data(), d_out, sizeof(uint32_t) * got_words.size(), cudaMemcpyDeviceToHost));

        EXPECT_EQ(got_words, expected_words);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("unsupported toolchain") != std::string::npos) {
            GTEST_SKIP() << "CUDA driver/toolchain mismatch: " << msg;
        }
        throw;
    }
}

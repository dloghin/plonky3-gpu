#include <gtest/gtest.h>

#include "baby_bear.hpp"
#include "cuda_compat.hpp"
#include "poseidon2_cuda.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using p3_cuda_compat::DeviceBuffer;
using p3_field::BabyBear;
using poseidon2::Poseidon2Cuda;
using poseidon2::Poseidon2SpongeCuda;
using poseidon2::poseidon2_hash_kernel;
using poseidon2::poseidon2_permute_kernel;

namespace {

constexpr size_t WIDTH = 16;
constexpr size_t RATE = 8;
constexpr size_t ROUNDS_F = 8;
constexpr size_t ROUNDS_P = 14;
constexpr uint64_t D_EXP = 5;

using F = BabyBear;
using Poseidon = Poseidon2Cuda<F, WIDTH, ROUNDS_F, ROUNDS_P, D_EXP>;
using Sponge = Poseidon2SpongeCuda<F, WIDTH, RATE, ROUNDS_F, ROUNDS_P, D_EXP>;

Poseidon make_poseidon() {
    Poseidon p{};
    for (size_t r = 0; r < ROUNDS_F / 2; ++r) {
        for (size_t i = 0; i < WIDTH; ++i) {
            p.initial_external_constants[r][i] = F(static_cast<uint32_t>((13u * r + 7u * i + 3u) % F::PRIME));
            p.terminal_external_constants[r][i] = F(static_cast<uint32_t>((17u * r + 5u * i + 11u) % F::PRIME));
        }
    }
    for (size_t r = 0; r < ROUNDS_P; ++r) {
        p.internal_constants[r] = F(static_cast<uint32_t>((19u * r + 23u) % F::PRIME));
    }
    return p;
}

void ensure_cuda_or_skip() {
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "CUDA device not available";
    }
}

} // namespace

TEST(Poseidon2Cuda, PermuteKernelMatchesHost) {
    ensure_cuda_or_skip();

    const Poseidon poseidon = make_poseidon();
    std::array<F, WIDTH> host_state{};
    for (size_t i = 0; i < WIDTH; ++i) host_state[i] = F(static_cast<uint32_t>(i + 1));

    auto expected = host_state;
    poseidon.permute(expected.data());

    DeviceBuffer buf_poseidon;
    DeviceBuffer buf_states;
    try {
        void* p_poseidon = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_poseidon, sizeof(Poseidon)));
        buf_poseidon.reset(p_poseidon);

        void* p_states = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_states, sizeof(F) * WIDTH));
        buf_states.reset(p_states);

        auto* d_poseidon = static_cast<Poseidon*>(buf_poseidon.get());
        auto* d_states = static_cast<F*>(buf_states.get());

        P3_CUDA_CHECK(cudaMemcpy(d_poseidon, &poseidon, sizeof(Poseidon), cudaMemcpyHostToDevice));
        P3_CUDA_CHECK(cudaMemcpy(d_states, host_state.data(), sizeof(F) * WIDTH, cudaMemcpyHostToDevice));

        poseidon2_permute_kernel<F, WIDTH, ROUNDS_F, ROUNDS_P, D_EXP><<<1, 1>>>(d_poseidon, d_states, 1);
        P3_CUDA_CHECK(cudaGetLastError());
        P3_CUDA_CHECK(cudaDeviceSynchronize());

        std::array<F, WIDTH> got{};
        P3_CUDA_CHECK(cudaMemcpy(got.data(), d_states, sizeof(F) * WIDTH, cudaMemcpyDeviceToHost));
        EXPECT_EQ(got, expected);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("unsupported toolchain") != std::string::npos) {
            GTEST_SKIP() << "CUDA driver/toolchain mismatch: " << msg;
        }
        throw;
    }
}

TEST(Poseidon2Cuda, HashKernelMatchesHost) {
    ensure_cuda_or_skip();

    const Poseidon poseidon = make_poseidon();
    const Sponge sponge{poseidon};

    constexpr size_t input_len = 5;
    constexpr size_t n_hashes = 2;
    std::array<F, input_len * n_hashes> inputs{};
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = F(static_cast<uint32_t>(i + 3));

    std::array<F, RATE * n_hashes> expected{};
    sponge.hash(inputs.data(), input_len, expected.data());
    sponge.hash(inputs.data() + input_len, input_len, expected.data() + RATE);

    DeviceBuffer buf_sponge;
    DeviceBuffer buf_inputs;
    DeviceBuffer buf_outputs;
    try {
        void* p_sponge = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_sponge, sizeof(Sponge)));
        buf_sponge.reset(p_sponge);

        void* p_inputs = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_inputs, sizeof(F) * inputs.size()));
        buf_inputs.reset(p_inputs);

        void* p_outputs = nullptr;
        P3_CUDA_CHECK(cudaMalloc(&p_outputs, sizeof(F) * expected.size()));
        buf_outputs.reset(p_outputs);

        auto* d_sponge = static_cast<Sponge*>(buf_sponge.get());
        auto* d_inputs = static_cast<F*>(buf_inputs.get());
        auto* d_outputs = static_cast<F*>(buf_outputs.get());

        P3_CUDA_CHECK(cudaMemcpy(d_sponge, &sponge, sizeof(Sponge), cudaMemcpyHostToDevice));
        P3_CUDA_CHECK(cudaMemcpy(d_inputs, inputs.data(), sizeof(F) * inputs.size(), cudaMemcpyHostToDevice));

        poseidon2_hash_kernel<F, WIDTH, RATE, ROUNDS_F, ROUNDS_P, D_EXP><<<1, n_hashes>>>(
            d_sponge, d_inputs, d_outputs, input_len, n_hashes);
        P3_CUDA_CHECK(cudaGetLastError());
        P3_CUDA_CHECK(cudaDeviceSynchronize());

        std::array<F, RATE * n_hashes> got{};
        P3_CUDA_CHECK(cudaMemcpy(got.data(), d_outputs, sizeof(F) * got.size(), cudaMemcpyDeviceToHost));
        EXPECT_EQ(got, expected);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("unsupported toolchain") != std::string::npos) {
            GTEST_SKIP() << "CUDA driver/toolchain mismatch: " << msg;
        }
        throw;
    }
}


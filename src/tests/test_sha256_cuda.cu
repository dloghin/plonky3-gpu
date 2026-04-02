/**
 * @file test_sha256_cuda.cu
 * @brief CUDA test for device-side Sha256Compress.
 */

#include <gtest/gtest.h>

#include "sha256.hpp"

#include <array>
#include <cstdint>

using namespace p3_symmetric;

TEST(Sha256Cuda, NvccBuildPathMatchesSha256SingleBlock) {
    std::array<uint8_t, 32> left{};
    std::array<uint8_t, 32> right{};
    right[0] = 0x80u;
    right[30] = 0x01u;

    const Sha256 sha;
    const Sha256Compress compress;
    const auto expected = sha.hash_iter(left.data(), left.size());
    const auto got = compress.compress({left, right});
    EXPECT_EQ(got, expected);
}

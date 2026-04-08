#include <gtest/gtest.h>

#include "baby_bear.hpp"
#include "extension_field.hpp"
#include "fri_fold_cuda.hpp"
#include "fri_folding.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using p3_field::BabyBear;
using p3_field::BabyBear4;
using p3_fri::fold_matrix_cuda;
using p3_fri::TwoAdicFriFolding;

namespace {

using Val = BabyBear;
using EF = BabyBear4;

EF make_ef(uint32_t seed) {
    return EF({
        Val(seed % Val::PRIME),
        Val((seed + 17u) % Val::PRIME),
        Val((seed + 31u) % Val::PRIME),
        Val((seed + 53u) % Val::PRIME),
    });
}

std::vector<EF> make_input(size_t n) {
    std::vector<EF> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = make_ef(static_cast<uint32_t>(i * 7u + 11u));
    }
    return v;
}

void expect_fold_matches_cpu(size_t log_height, size_t log_arity) {
    const size_t n = size_t(1) << log_height;
    const auto input = make_input(n);
    const EF beta = make_ef(static_cast<uint32_t>(n + log_arity * 13u));

    const auto cpu = TwoAdicFriFolding<Val, EF>::fold_matrix(log_height, log_arity, beta, input);
    try {
        const auto gpu = fold_matrix_cuda<Val, EF>(beta, log_arity, log_height, input);
        EXPECT_EQ(gpu, cpu);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("unsupported toolchain") != std::string::npos) {
            GTEST_SKIP() << "CUDA runtime/kernel unavailable: " << msg;
        }
        throw;
    }
}

} // namespace

TEST(FriFoldCuda, Arity2MatchesCpu) {
    expect_fold_matches_cpu(10, 1);
}

TEST(FriFoldCuda, Arity4MatchesCpu) {
    expect_fold_matches_cpu(10, 2);
}

TEST(FriFoldCuda, Arity8MatchesCpu) {
    expect_fold_matches_cpu(12, 3);
}


#include <gtest/gtest.h>

#include "baby_bear.hpp"
#include "mersenne31.hpp"
#include "dense_matrix.hpp"
#include "radix2_dit.hpp"
#include "naive_dft.hpp"
#include "radix2_bowers.hpp"
#include "radix2_dit_parallel.hpp"
#include "mersenne31_dft.hpp"

#include <vector>
#include <cstdint>

using namespace p3_field;
using namespace p3_matrix;
using namespace p3_dft;

template<typename F>
static bool matrices_equal(const RowMajorMatrix<F>& a, const RowMajorMatrix<F>& b) {
    if (a.height() != b.height() || a.width() != b.width()) return false;
    for (size_t r = 0; r < a.height(); ++r) {
        for (size_t c = 0; c < a.width(); ++c) {
            if (a.get_unchecked(r, c) != b.get_unchecked(r, c)) return false;
        }
    }
    return true;
}

static RowMajorMatrix<BabyBear> make_babybear_matrix(size_t h, size_t w) {
    std::vector<BabyBear> vals;
    vals.reserve(h * w);
    for (size_t i = 0; i < h * w; ++i) {
        vals.emplace_back(static_cast<uint32_t>((i * 11939u + 17u) % BabyBear::PRIME));
    }
    return RowMajorMatrix<BabyBear>(std::move(vals), w);
}

TEST(DftVariants, NaiveMatchesRadix2DitUpTo256) {
    Radix2Dit<BabyBear> radix;
    NaiveDft<BabyBear> naive;
    for (size_t log_n = 1; log_n <= 8; ++log_n) {
        const size_t n = static_cast<size_t>(1) << log_n;
        auto input = make_babybear_matrix(n, 3);
        auto expected = radix.dft_batch(input);
        auto got = naive.dft_batch(make_babybear_matrix(n, 3));
        EXPECT_TRUE(matrices_equal(got, expected)) << "Mismatch at n=" << n;
    }
}

TEST(DftVariants, NaiveRoundTrip) {
    NaiveDft<BabyBear> naive;
    auto input = make_babybear_matrix(64, 2);
    auto recovered = naive.idft_batch(naive.dft_batch(input));
    EXPECT_TRUE(matrices_equal(recovered, make_babybear_matrix(64, 2)));
}

TEST(DftVariants, BowersMatchesRadix2Dit) {
    Radix2Dit<BabyBear> radix;
    Radix2Bowers<BabyBear> bowers;
    auto input = make_babybear_matrix(128, 4);
    auto expected = radix.dft_batch(input);
    auto got = bowers.dft_batch(make_babybear_matrix(128, 4));
    EXPECT_TRUE(matrices_equal(got, expected));
}

TEST(DftVariants, BowersRoundTrip) {
    Radix2Bowers<BabyBear> bowers;
    auto input = make_babybear_matrix(128, 3);
    auto recovered = bowers.idft_batch(bowers.dft_batch(input));
    EXPECT_TRUE(matrices_equal(recovered, make_babybear_matrix(128, 3)));
}

TEST(DftVariants, ParallelMatchesRadix2DitAcrossThreadCounts) {
    Radix2Dit<BabyBear> radix;
    auto input = make_babybear_matrix(256, 6);
    auto expected = radix.dft_batch(input);

    for (size_t threads : {size_t(1), size_t(2), size_t(4)}) {
        Radix2DitParallel<BabyBear> parallel(threads);
        auto got = parallel.dft_batch(make_babybear_matrix(256, 6));
        EXPECT_TRUE(matrices_equal(got, expected)) << "Mismatch with threads=" << threads;
    }
}

TEST(DftVariants, ParallelRoundTrip) {
    for (size_t threads : {size_t(1), size_t(2), size_t(8)}) {
        Radix2DitParallel<BabyBear> parallel(threads);
        auto input = make_babybear_matrix(128, 5);
        auto recovered = parallel.idft_batch(parallel.dft_batch(input));
        EXPECT_TRUE(matrices_equal(recovered, make_babybear_matrix(128, 5)))
            << "Round-trip mismatch with threads=" << threads;
    }
}

TEST(DftVariants, Mersenne31ComplexRoundTrip) {
    // Size 64 base-field DFT packed into size 32 complex DFT.
    std::vector<Mersenne31> vals;
    vals.reserve(64);
    for (size_t i = 0; i < 64; ++i) {
        vals.emplace_back(static_cast<uint32_t>((i * 911u + 7u) % Mersenne31::PRIME));
    }
    RowMajorMatrix<Mersenne31> input(std::move(vals), 1);

    auto freq = Mersenne31Dft::dft_batch(input);
    EXPECT_EQ(freq.height(), 33u);
    EXPECT_EQ(freq.width(), 1u);
    auto recovered = Mersenne31Dft::idft_batch(freq);

    RowMajorMatrix<Mersenne31> expected(64, 1);
    for (size_t i = 0; i < 64; ++i) {
        expected.set_unchecked(i, 0, Mersenne31(static_cast<uint32_t>((i * 911u + 7u) % Mersenne31::PRIME)));
    }
    EXPECT_TRUE(matrices_equal(recovered, expected));
}

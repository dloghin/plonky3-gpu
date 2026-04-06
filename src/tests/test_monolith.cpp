#include <gtest/gtest.h>

#include "mersenne31.hpp"
#include "monolith.hpp"
#include "padding_free_sponge.hpp"
#include "truncated_permutation.hpp"

#include <array>
#include <vector>

using monolith::MonolithMersenne31;
using p3_field::Mersenne31;
using p3_symmetric::PaddingFreeSponge;
using p3_symmetric::TruncatedPermutation;

TEST(MonolithMersenne31, RoundConstantsMatchReference) {
    const auto& rc = MonolithMersenne31<>::round_constants();

    EXPECT_EQ(rc[0][0], Mersenne31(1033436816u));
    EXPECT_EQ(rc[0][15], Mersenne31(718914942u));
    EXPECT_EQ(rc[4][0], Mersenne31(534908981u));
    EXPECT_EQ(rc[4][15], Mersenne31(1110912837u));
}

TEST(MonolithMersenne31, PermutationMatchesRustReferenceVector) {
    MonolithMersenne31<> monolith;
    std::array<Mersenne31, 16> input{};
    for (uint32_t i = 0; i < 16; ++i) {
        input[i] = Mersenne31(i);
    }

    const std::array<Mersenne31, 16> expected = {
        Mersenne31(609156607u), Mersenne31(290107110u), Mersenne31(1900746598u), Mersenne31(1734707571u),
        Mersenne31(2050994835u), Mersenne31(1648553244u), Mersenne31(1307647296u), Mersenne31(1941164548u),
        Mersenne31(1707113065u), Mersenne31(1477714255u), Mersenne31(1170160793u), Mersenne31(93800695u),
        Mersenne31(769879348u), Mersenne31(375548503u), Mersenne31(1989726444u), Mersenne31(1349325635u),
    };

    monolith.permute_mut(input);
    EXPECT_EQ(input, expected);
}

TEST(MonolithMersenne31, WorksWithPaddingFreeSponge) {
    constexpr size_t WIDTH = 16;
    constexpr size_t RATE = 8;
    constexpr size_t OUT = 8;
    PaddingFreeSponge<MonolithMersenne31<WIDTH, 5>, Mersenne31, WIDTH, RATE, OUT> sponge{
        MonolithMersenne31<WIDTH, 5>{}
    };

    std::vector<Mersenne31> input;
    for (uint32_t i = 1; i <= 8; ++i) {
        input.push_back(Mersenne31(i));
    }

    auto digest = sponge.hash_iter(input);
    const std::array<Mersenne31, OUT> expected = {
        Mersenne31(205038974u), Mersenne31(1315645601u), Mersenne31(1694164955u), Mersenne31(763338091u),
        Mersenne31(427378119u), Mersenne31(1352396535u), Mersenne31(1937967665u), Mersenne31(1899543966u),
    };
    EXPECT_EQ(digest, expected);
}

TEST(MonolithMersenne31, WorksWithTruncatedPermutation) {
    constexpr size_t WIDTH = 16;
    constexpr size_t N = 2;
    constexpr size_t CHUNK = 8;
    TruncatedPermutation<MonolithMersenne31<WIDTH, 5>, Mersenne31, N, CHUNK, WIDTH> compress{
        MonolithMersenne31<WIDTH, 5>{}
    };

    std::array<std::array<Mersenne31, CHUNK>, N> input{};
    uint32_t v = 1;
    for (size_t n = 0; n < N; ++n) {
        for (size_t i = 0; i < CHUNK; ++i) {
            input[n][i] = Mersenne31(v++);
        }
    }

    auto out = compress.compress(input);
    const std::array<Mersenne31, CHUNK> expected = {
        Mersenne31(1129788458u), Mersenne31(1777676151u), Mersenne31(2102297232u), Mersenne31(397626477u),
        Mersenne31(1813724644u), Mersenne31(326455788u), Mersenne31(1583148110u), Mersenne31(627558891u),
    };
    EXPECT_EQ(out, expected);
}

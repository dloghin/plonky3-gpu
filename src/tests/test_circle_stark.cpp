#include <gtest/gtest.h>

#include "circle_domain.hpp"
#include "circle_fft.hpp"
#include "circle_folding.hpp"
#include "circle_pcs.hpp"
#include "circle_stark.hpp"
#include "mersenne31.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

using F = p3_field::Mersenne31;

namespace {

struct CircleTestChallenger {
    template <typename Challenge>
    Challenge sample_challenge() {
        return Challenge(uint32_t{7});
    }

    template <typename Challenge>
    void observe_challenge(const Challenge&) {}
};

p3_matrix::RowMajorMatrix<F> sample_matrix(std::size_t height, std::size_t width) {
    std::vector<F> values(height * width);
    for (std::size_t r = 0; r < height; ++r) {
        for (std::size_t c = 0; c < width; ++c) {
            values[r * width + c] = F(static_cast<uint32_t>(3 + 11 * r + 17 * c));
        }
    }
    return p3_matrix::RowMajorMatrix<F>(std::move(values), width);
}

} // namespace

TEST(CircleDomain, GroupArithmeticMatchesReferenceConstants) {
    using p3_circle::CirclePoint;

    const auto g31 = CirclePoint<F>::generator(31);
    EXPECT_EQ(g31.x, F(uint32_t{311014874u}));
    EXPECT_EQ(g31.y, F(uint32_t{1584694829u}));

    const auto g3 = CirclePoint<F>::generator(3);
    EXPECT_EQ(g3.x, F(uint32_t{32768u}));
    EXPECT_EQ(g3.y, F(uint32_t{2147450879u}));

    EXPECT_EQ(g3 - g3, CirclePoint<F>::zero());
    EXPECT_EQ(g3 + g3, g3 * 2);
    EXPECT_EQ(g3 + g3 + g3, g3 * 3);
    EXPECT_EQ(g3 * 7, -g3);
    EXPECT_EQ(g3 * 8, CirclePoint<F>::zero());
}

TEST(CircleDomain, StandardDomainOrderingAndVanishing) {
    const auto domain = p3_circle::CircleDomain<F>::standard(4);
    ASSERT_EQ(domain.size(), 16u);

    auto t = domain.first_point();
    for (std::size_t i = 0; i < domain.size(); ++i) {
        EXPECT_EQ(p3_circle::CirclePoint<F>::from_projective_line(t), domain.at(i));
        auto next = domain.next_point(t);
        ASSERT_TRUE(next.has_value());
        t = *next;
    }
    EXPECT_EQ(t, domain.first_point());

    for (const auto& p : domain.points()) {
        EXPECT_EQ(domain.vanishing_poly(p), F::zero_val());
    }

    const auto disjoint = domain.create_disjoint_domain(domain.size());
    const auto selectors = domain.selectors_at_point(disjoint.at(0));
    EXPECT_NE(selectors.inv_vanishing, F::zero_val());
}

TEST(CircleFft, CfftIcfftRoundTrip) {
    p3_circle::CircleFft<F> fft;
    for (std::size_t log_n = 2; log_n <= 5; ++log_n) {
        const auto domain = p3_circle::CircleDomain<F>::standard(log_n);
        const auto coeffs = sample_matrix(domain.size(), 3);
        const auto evals = fft.cfft(coeffs, domain);
        const auto round_trip = fft.icfft(evals, domain);
        EXPECT_EQ(round_trip, coeffs);

        const auto pts = domain.points();
        for (std::size_t r = 0; r < domain.size(); ++r) {
            const auto basis = p3_circle::circle_basis(pts[r], log_n);
            for (std::size_t c = 0; c < coeffs.width(); ++c) {
                F acc = F::zero_val();
                for (std::size_t i = 0; i < coeffs.height(); ++i) {
                    acc += coeffs.get_unchecked(i, c) * basis[i];
                }
                EXPECT_EQ(evals.get_unchecked(r, c), acc);
            }
        }
    }
}

TEST(CircleFolding, FoldMatrixMatchesRows) {
    const std::size_t log_folded_height = 5;
    const std::size_t height = std::size_t{1} << log_folded_height;
    const auto evals = sample_matrix(height, 2);
    const F beta(uint32_t{19});

    const auto y_folded = p3_circle::fold_y(beta, evals);
    ASSERT_EQ(y_folded.size(), height);
    for (std::size_t i = 0; i < height; ++i) {
        EXPECT_EQ(y_folded[i],
                  p3_circle::fold_y_row(i, log_folded_height, beta,
                                         evals.get_unchecked(i, 0),
                                         evals.get_unchecked(i, 1)));
    }

    const auto x_folded = p3_circle::fold_x(beta, 1, evals);
    ASSERT_EQ(x_folded.size(), height);
    for (std::size_t i = 0; i < height; ++i) {
        EXPECT_EQ(x_folded[i],
                  p3_circle::fold_x_row(i, log_folded_height, 1, beta,
                                         evals.get_unchecked(i, 0),
                                         evals.get_unchecked(i, 1)));
    }

    const auto folded_vec = p3_circle::fold_circle(evals.values, beta);
    EXPECT_EQ(folded_vec, y_folded);
}

TEST(CirclePcs, CommitOpenVerifyRoundTripAndRejectsTamper) {
    using Pcs = p3_circle::CirclePcs<F>;
    Pcs pcs;
    const auto domain = Pcs::Domain::standard(3);
    auto evals = sample_matrix(domain.size(), 2);

    auto [commitment, pd] = pcs.commit({{domain, evals}});
    const auto z = p3_circle::CirclePoint<F>::from_projective_line(F(uint32_t{5}));

    CircleTestChallenger prover_challenger;
    auto [opened, proof] = pcs.open({{&pd, {{z}}}}, prover_challenger);
    ASSERT_EQ(opened.size(), 1u);
    ASSERT_EQ(opened[0].size(), 1u);
    ASSERT_EQ(opened[0][0].size(), 1u);
    ASSERT_EQ(opened[0][0][0].size(), 2u);

    Pcs::VerifyCommitment vc;
    vc.commitment = commitment;
    vc.domains = {domain};
    vc.points = {{z}};
    vc.opened_values = {opened[0][0]};

    CircleTestChallenger verifier_challenger;
    EXPECT_TRUE(pcs.verify({vc}, proof, verifier_challenger));

    vc.opened_values[0][0][0] += F::one_val();
    CircleTestChallenger tampered_challenger;
    EXPECT_FALSE(pcs.verify({vc}, proof, tampered_challenger));
}

TEST(CircleStark, ProveVerifyRoundTrip) {
    p3_circle::CircleStark<F> stark;
    const auto trace = sample_matrix(8, 2);

    CircleTestChallenger prover_challenger;
    auto proof = stark.prove(trace, prover_challenger);

    CircleTestChallenger verifier_challenger;
    EXPECT_TRUE(stark.verify(proof, verifier_challenger));

    proof.opened_values[0][0] += F::one_val();
    CircleTestChallenger tampered_challenger;
    EXPECT_FALSE(stark.verify(proof, tampered_challenger));
}

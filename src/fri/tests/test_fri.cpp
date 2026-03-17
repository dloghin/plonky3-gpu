/**
 * @file test_fri.cpp
 * @brief Google Test suite for the FRI core C++ port.
 *
 * Tests cover:
 *  1. compute_log_arity_for_round - scheduling logic
 *  2. TwoAdicFriFolding::fold_row - arity-2 and arity-4 folding correctness
 *  3. TwoAdicFriFolding::fold_matrix - multi-row folding
 *  4. FriParameters helpers
 *  5. Round-trip prove+verify using mock MMCS and mock Challenger
 */

#include <gtest/gtest.h>

#include "fri_params.hpp"
#include "fri_proof.hpp"
#include "fri_folding.hpp"
#include "fri_prover.hpp"
#include "fri_verifier.hpp"

#include "baby_bear.hpp"
#include "extension_field.hpp"
#include "p3_util/util.hpp"

#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <functional>
#include <unordered_map>

using namespace p3_fri;
using namespace p3_field;
using namespace p3_util;

// Convenience aliases
using BB  = BabyBear;
using BB4 = BabyBear4;   // BinomialExtensionField<BabyBear, 4, 11>

// ============================================================================
// Section 1: compute_log_arity_for_round
// ============================================================================

TEST(FriParams, ComputeLogArityNoNextInput) {
    // No next input: fold all the way to log_final_height
    // log_current = 10, log_final = 2, max_arity = 4
    // available = 10 - 2 = 8, capped at 4
    EXPECT_EQ(compute_log_arity_for_round(10, false, 0, 2, 4), 4u);
}

TEST(FriParams, ComputeLogArityWithNextInput) {
    // Next input at height 6, must fold from 10 down to 6 => arity = 4
    EXPECT_EQ(compute_log_arity_for_round(10, true, 6, 2, 4), 4u);
}

TEST(FriParams, ComputeLogArityNextInputCloser) {
    // Next input at height 9, from 10 => arity = 1
    EXPECT_EQ(compute_log_arity_for_round(10, true, 9, 2, 4), 1u);
}

TEST(FriParams, ComputeLogArityAlreadyAtFinal) {
    // Already at final height, no folding needed
    EXPECT_EQ(compute_log_arity_for_round(4, false, 0, 4, 4), 0u);
}

TEST(FriParams, ComputeLogAritySmallAvailable) {
    // Only 2 levels to fold, max_arity = 4 → returns 2
    EXPECT_EQ(compute_log_arity_for_round(6, false, 0, 4, 4), 2u);
}

TEST(FriParams, FriParametersHelpers) {
    // Trivial MMCS type placeholder for testing params helpers
    struct DummyMmcs {};
    FriParameters<DummyMmcs> p;
    p.log_blowup = 3;
    p.log_final_poly_len = 2;
    p.max_log_arity = 4;
    p.num_queries = 10;
    p.commit_proof_of_work_bits = 0;
    p.query_proof_of_work_bits  = 0;

    EXPECT_EQ(p.blowup(), 8u);
    EXPECT_EQ(p.final_poly_len(), 4u);
    EXPECT_EQ(p.log_final_height(), 5u);  // 3 + 2
}

// ============================================================================
// Section 2: TwoAdicFriFolding::fold_row — arity 2
// ============================================================================

// Manual check of fold_row for arity=2 using the closed-form formula:
//   folded = (f(x) + f(-x))/2 + beta * (f(x) - f(-x)) / (2*x)
// We verify that fold_row produces the same result.

TEST(FriFolding, FoldRowArity2Manual) {
    // Use BabyBear as both Val and Challenge (for simplicity, using base field beta)
    // We interpret BB4 as Challenge.  But to keep things simple, let's use
    // Challenge = BabyBear and Val = BabyBear.
    using Val = BB;
    using Challenge = BB;  // use base field as "challenge" for this unit test

    // For arity=2 and log_height=4 (height=16, folded height = 8):
    // omega = two_adic_generator(4) — primitive 16th root of unity
    // w     = two_adic_generator(1) = -1 (primitive 2nd root of unity)
    // At index=0: x = omega^(bit_rev(0,3)) = omega^0 = 1
    //   evals[0] = f(1), evals[1] = f(-1)
    //   folded = (f(1) + f(-1))/2 + beta*(f(1) - f(-1))/2

    size_t log_height = 4;
    size_t log_arity  = 1;
    size_t index      = 0;

    BB f0 = BB(7u);
    BB f1 = BB(3u);
    BB beta = BB(5u);

    // x = omega^(bit_rev(0,3)) = omega^0 = 1
    // folded = (f0+f1)/2 + beta*(f0-f1)/2 = (f0+f1+beta*(f0-f1))/2
    // Compute in BabyBear arithmetic:
    BB two_inv = (BB::one_val() + BB::one_val()).inv();
    BB expected = (f0 + f1 + beta * (f0 - f1)) * two_inv;

    std::vector<BB> evals = {f0, f1};
    BB result = TwoAdicFriFolding<Val, Challenge>::fold_row(
        index, log_height, log_arity, beta, evals
    );
    EXPECT_EQ(result, expected);
}

TEST(FriFolding, FoldRowArity2NonZeroIndex) {
    // For index=1, log_height=4 (folded height=8):
    // x = omega^(bit_rev(1,3)) where omega = 2-adic gen(4)
    // bit_rev(1,3) = 4 (100 in 3 bits reversed)
    // x = omega^4  (omega has order 16, so omega^4 has order 4)
    using Val = BB;
    using Challenge = BB;

    size_t log_height = 4;
    size_t log_arity  = 1;
    size_t index      = 1;

    // x = two_adic_generator(4)^(bit_rev(1,3))
    // bit_rev(1,3) = 4
    BB omega = BB::two_adic_generator(log_height);
    size_t x_pow = reverse_bits_len(index, log_height - log_arity);
    BB x = omega.exp_u64(static_cast<uint64_t>(x_pow));

    BB f0 = BB(9u);
    BB f1 = BB(2u);
    BB beta = BB(13u);

    BB two_inv = (BB::one_val() + BB::one_val()).inv();
    // folded = (f0+f1)/2 + beta*(f0-f1)/(2*x)
    BB expected = (f0 + f1) * two_inv
                + beta * (f0 - f1) * two_inv * x.inv();

    std::vector<BB> evals = {f0, f1};
    BB result = TwoAdicFriFolding<Val, Challenge>::fold_row(
        index, log_height, log_arity, beta, evals
    );
    EXPECT_EQ(result, expected);
}

TEST(FriFolding, FoldRowArity1Identity) {
    // log_arity = 0 means no folding: just return evals[0]
    using Val = BB;
    using Challenge = BB;

    BB f0 = BB(42u);
    BB beta = BB(7u);
    std::vector<BB> evals = {f0};
    BB result = TwoAdicFriFolding<Val, Challenge>::fold_row(0, 4, 0, beta, evals);
    EXPECT_EQ(result, f0);
}

// ============================================================================
// Section 3: TwoAdicFriFolding::fold_row — arity 4 consistency
// ============================================================================

// For arity=4, fold_row should give the same answer as manually doing two
// arity-2 folds with the same beta.  In plonky3 FRI, folding by arity 4 with
// challenge beta is NOT the same as two arity-2 folds — it uses a single
// degree-3 Lagrange interpolation.  We just check internal consistency:
//   fold_row is deterministic given the same inputs.

TEST(FriFolding, FoldRowArity4Deterministic) {
    using Val = BB;
    using Challenge = BB;

    size_t log_height = 5;
    size_t log_arity  = 2;  // arity = 4

    BB beta = BB(17u);
    std::vector<BB> evals = {BB(1u), BB(2u), BB(3u), BB(4u)};

    BB r1 = TwoAdicFriFolding<Val, Challenge>::fold_row(0, log_height, log_arity, beta, evals);
    BB r2 = TwoAdicFriFolding<Val, Challenge>::fold_row(0, log_height, log_arity, beta, evals);
    EXPECT_EQ(r1, r2);
}

TEST(FriFolding, FoldRowArity4VariesWithBeta) {
    using Val = BB;
    using Challenge = BB;

    size_t log_height = 5;
    size_t log_arity  = 2;

    std::vector<BB> evals = {BB(1u), BB(2u), BB(3u), BB(4u)};

    BB r1 = TwoAdicFriFolding<Val, Challenge>::fold_row(0, log_height, log_arity, BB(7u), evals);
    BB r2 = TwoAdicFriFolding<Val, Challenge>::fold_row(0, log_height, log_arity, BB(13u), evals);
    // Different betas should generally give different results
    // (unless special cancellation — very unlikely for generic values)
    EXPECT_NE(r1, r2);
}

// ============================================================================
// Section 4: fold_matrix
// ============================================================================

TEST(FriFolding, FoldMatrixArity2) {
    using Val = BB;
    using Challenge = BB;

    // A vector of length 4, arity=2 => 2 output values
    // current = [f0, f1, f2, f3]  (two rows: [f0,f1] and [f2,f3])
    size_t log_height = 2;  // height = 4
    size_t log_arity  = 1;  // arity = 2

    BB beta = BB(5u);
    std::vector<BB> current = {BB(10u), BB(20u), BB(30u), BB(40u)};

    auto folded = TwoAdicFriFolding<Val, Challenge>::fold_matrix(
        log_height, log_arity, beta, current
    );

    ASSERT_EQ(folded.size(), 2u);

    // Row 0: fold_row(0, 2, 1, beta, {10, 20})
    std::vector<BB> row0 = {BB(10u), BB(20u)};
    BB expected0 = TwoAdicFriFolding<Val, Challenge>::fold_row(0, log_height, log_arity, beta, row0);
    EXPECT_EQ(folded[0], expected0);

    // Row 1: fold_row(1, 2, 1, beta, {30, 40})
    std::vector<BB> row1 = {BB(30u), BB(40u)};
    BB expected1 = TwoAdicFriFolding<Val, Challenge>::fold_row(1, log_height, log_arity, beta, row1);
    EXPECT_EQ(folded[1], expected1);
}

TEST(FriFolding, FoldMatrixArity4) {
    using Val = BB;
    using Challenge = BB;

    // A vector of length 8, arity=4 => 2 output values
    size_t log_height = 3;   // height = 8
    size_t log_arity  = 2;   // arity = 4

    BB beta = BB(3u);
    std::vector<BB> current = {
        BB(1u), BB(2u), BB(3u), BB(4u),
        BB(5u), BB(6u), BB(7u), BB(8u)
    };

    auto folded = TwoAdicFriFolding<Val, Challenge>::fold_matrix(
        log_height, log_arity, beta, current
    );

    ASSERT_EQ(folded.size(), 2u);

    // Verify each row independently
    for (size_t i = 0; i < 2; ++i) {
        std::vector<BB> row(4);
        for (size_t j = 0; j < 4; ++j) row[j] = current[i*4+j];
        BB expected = TwoAdicFriFolding<Val, Challenge>::fold_row(
            i, log_height, log_arity, beta, row
        );
        EXPECT_EQ(folded[i], expected);
    }
}

// ============================================================================
// Section 5: Lagrange interpolation property of fold_row
// ============================================================================

// fold_row performs Lagrange interpolation.  If we evaluate the polynomial at
// one of its defining points t_i = x*w^i, we should recover evals[i].

TEST(FriFolding, FoldRowLagrangeAtEvalPoint) {
    using Val = BB;
    using Challenge = BB;

    // Arity 2: eval at t_0 = x should give evals[0]
    size_t log_height = 4;
    size_t log_arity  = 1;
    size_t index      = 2;

    BB omega = BB::two_adic_generator(log_height);
    size_t x_pow = reverse_bits_len(index, log_height - log_arity);
    BB x = omega.exp_u64(static_cast<uint64_t>(x_pow));

    BB f0 = BB(11u);
    BB f1 = BB(7u);

    std::vector<BB> evals = {f0, f1};

    // When beta = x, fold_row should return f0 (the interpolating poly at t_0 = x)
    BB result = TwoAdicFriFolding<Val, Challenge>::fold_row(
        index, log_height, log_arity, x, evals
    );
    EXPECT_EQ(result, f0);

    // When beta = -x (= x * w where w = -1), fold_row should return f1
    BB neg_x = BB::zero_val() - x;
    BB result2 = TwoAdicFriFolding<Val, Challenge>::fold_row(
        index, log_height, log_arity, neg_x, evals
    );
    EXPECT_EQ(result2, f1);
}

// ============================================================================
// Section 6: Round-trip prove+verify (mock infrastructure)
// ============================================================================

// ---- Mock MMCS ----
// Stores committed matrices and returns trivial proofs (row index only).
// The commitment is a "hash" (sum of all elements mod p) for simplicity.
struct MockMmcsOpeningProof {
    size_t row_index;
};

struct MockMmcsCommitment {
    uint32_t hash;   // sum of all values mod PRIME
    size_t height;   // number of rows
    size_t width;    // number of columns
    bool operator==(const MockMmcsCommitment& o) const {
        return hash == o.hash && height == o.height && width == o.width;
    }
};

struct MockMmcsProverData {
    std::vector<BB4> data;  // flat row-major storage
    size_t height;
    size_t width;
};

struct MockMmcs {
    using Commitment   = MockMmcsCommitment;
    using ProverData   = MockMmcsProverData;
    using OpeningProof = MockMmcsOpeningProof;

    // Commit a flat vector as a matrix of given width
    std::pair<Commitment, ProverData> commit_matrix(
        const std::vector<BB4>& vals,
        size_t width
    ) const {
        size_t height = vals.size() / width;
        // Compute a simple "hash": sum of all canonical values mod PRIME
        uint64_t acc = 0;
        for (const auto& v : vals) {
            for (size_t k = 0; k < 4; ++k) {
                acc = (acc + v[k].as_canonical_u64()) % BB::PRIME;
            }
        }
        Commitment c{static_cast<uint32_t>(acc), height, width};
        ProverData d{vals, height, width};
        return {c, d};
    }

    size_t log_height(const ProverData& d) const {
        return log2_strict_usize(d.height);
    }

    size_t log_width(const ProverData& d) const {
        return log2_strict_usize(d.width);
    }

    std::vector<BB4> get_row(const ProverData& d, size_t row) const {
        std::vector<BB4> row_vals(d.width);
        for (size_t j = 0; j < d.width; ++j) {
            row_vals[j] = d.data[row * d.width + j];
        }
        return row_vals;
    }

    void open_row(const ProverData& /*d*/, size_t row_index, OpeningProof& proof) const {
        proof.row_index = row_index;
    }

    bool verify_row(const Commitment& /*c*/, size_t /*row_index*/,
                    const std::vector<BB4>& /*row_vals*/,
                    const OpeningProof& /*proof*/) const {
        // Mock: always accept
        return true;
    }

    // For the input MMCS interface (used by verifier):
    // We reuse MockMmcs for the input oracle as well.
    void open(size_t /*query_index*/,
              const std::vector<ProverData>& /*data*/,
              OpeningProof& proof) const {
        proof.row_index = 0;
    }

    bool verify_query(size_t /*query_index*/, size_t /*log_height*/,
                      const std::vector<Commitment>& /*commits*/,
                      const OpeningProof& /*proof*/,
                      const BB4& /*claimed_eval*/) const {
        return true;
    }

    void observe_commitment(const Commitment& /*c*/) const {}
};

// ---- Mock Challenger ----
// Simple counter-based Fiat-Shamir transcript.
struct MockChallenger {
    uint64_t counter = 0;
    std::vector<uint64_t> observed;

    void observe_commitment(const MockMmcsCommitment& c) {
        observed.push_back(static_cast<uint64_t>(c.hash));
        counter += c.hash;
    }

    BB4 sample_challenge() {
        // Produce a non-trivial Challenge from counter
        counter = counter * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t v0 = static_cast<uint32_t>(counter       % BB::PRIME);
        uint32_t v1 = static_cast<uint32_t>((counter >> 8) % BB::PRIME);
        uint32_t v2 = static_cast<uint32_t>((counter >>16) % BB::PRIME);
        uint32_t v3 = static_cast<uint32_t>((counter >>24) % BB::PRIME);
        return BB4({BB(v0), BB(v1), BB(v2), BB(v3)});
    }

    size_t sample_bits(size_t bits) {
        counter = counter * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t mask = (bits >= 64) ? ~size_t(0) : ((size_t(1) << bits) - 1);
        return static_cast<size_t>(counter & mask);
    }

    uint64_t grind(size_t /*bits*/) {
        // Mock: PoW always succeeds trivially with witness 0.
        // We do NOT modify the counter here so that the prover and verifier
        // remain in sync (check_witness is a no-op and also does not modify state).
        return 0;
    }

    bool check_witness(size_t /*bits*/, uint64_t /*witness*/) {
        // Mock: always accept. Does not modify state so prover/verifier stay in sync.
        return true;
    }

    void observe_challenge(const BB4& c) {
        // Observe a Challenge element to keep prover/verifier in sync.
        for (size_t k = 0; k < 4; ++k) {
            counter += c[k].as_canonical_u64();
        }
    }

    void observe_arity(size_t la) {
        // Observe a log_arity value to keep prover/verifier in sync.
        counter += la;
    }
};

// ---- Minimal prove+verify round-trip test ----
//
// We exercise the prover and verifier with a very small polynomial so that
// the computation finishes quickly and we can verify the round-trip property.
//
// Polynomial: f(x) = 1 + 2x^2 (degree 2) over BabyBear.
// Evaluated on a two-adic domain of size N = 8 (2^3), blowup = 2 => domain 16.
// Then FRI: log_final_poly_len=1, log_blowup=1, max_log_arity=2, 1 query.

TEST(FriRoundTrip, SmallPolynomialConsistency) {
    using Val = BB;
    using Challenge = BB4;
    using Mmcs = MockMmcs;
    using Witness = uint64_t;
    using InputProof = MockMmcsOpeningProof;

    // Build a simple input evaluation vector (length 8 = 2^3)
    // We just use arbitrary values for the "evaluation of the polynomial":
    size_t log_n = 3;
    size_t n = size_t(1) << log_n;

    // Create evaluations as Challenge elements from a simple pattern
    std::vector<BB4> input(n);
    for (size_t i = 0; i < n; ++i) {
        BB4 v;
        v.coeffs[0] = BB(static_cast<uint32_t>((i * 7 + 3) % BB::PRIME));
        v.coeffs[1] = BB(static_cast<uint32_t>((i * 2 + 1) % BB::PRIME));
        v.coeffs[2] = BB::zero_val();
        v.coeffs[3] = BB::zero_val();
        input[i] = v;
    }

    MockMmcs mmcs_instance;
    auto [input_commit, input_pdata] = mmcs_instance.commit_matrix(input, 1);

    // FRI parameters
    FriParameters<Mmcs> params;
    params.log_blowup                = 1;  // blowup = 2
    params.log_final_poly_len        = 1;  // final_poly_len = 2
    params.max_log_arity             = 2;  // arity up to 4
    params.num_queries               = 1;
    params.commit_proof_of_work_bits = 0;
    params.query_proof_of_work_bits  = 0;
    params.mmcs                      = mmcs_instance;

    // log_final_height = 1 + 1 = 2, so fold from log_n=3 down to 2
    EXPECT_EQ(params.log_final_height(), 2u);
    // There should be at least one round of folding
    EXPECT_GT(log_n, params.log_final_height());

    MockChallenger prover_challenger;
    std::vector<std::vector<BB4>> inputs_copy = {input};
    std::vector<MockMmcsProverData> pdata = {input_pdata};

    // Run prover
    auto proof = prove_fri<Val, Challenge, Mmcs, MockChallenger, Witness, Mmcs, InputProof>(
        params,
        inputs_copy,
        prover_challenger,
        pdata,
        mmcs_instance
    );

    // Basic structural checks on the proof
    EXPECT_FALSE(proof.commit_phase_commits.empty());
    EXPECT_EQ(proof.commit_pow_witnesses.size(), proof.commit_phase_commits.size());
    EXPECT_EQ(proof.query_proofs.size(), params.num_queries);
    EXPECT_FALSE(proof.final_poly.empty());

    // The final polynomial should have length = final_poly_len = 2
    // (coefficients after truncation and IDFT)
    EXPECT_EQ(proof.final_poly.size(), params.final_poly_len());

    // Run verifier (reset challenger to same state)
    MockChallenger verifier_challenger;  // fresh challenger, same initial state

    // The verifier needs an eval_at_query callback.
    // For our mock, we just return the first element of input at the queried index.
    auto eval_fn = [&](size_t query_index, size_t log_height, const InputProof& /*ip*/) -> BB4 {
        size_t idx = query_index & (n - 1);  // wrap into [0, n)
        (void)log_height;
        return input[idx];
    };

    std::vector<MockMmcsCommitment> in_commits = {input_commit};

    bool ok = verify_fri<Val, Challenge, Mmcs, MockChallenger, Witness, Mmcs, InputProof>(
        params,
        in_commits,
        proof,
        verifier_challenger,
        mmcs_instance,
        eval_fn
    );

    // With the mock MMCS (always-accept proofs) the verifier should succeed
    EXPECT_TRUE(ok);
}

// ============================================================================
// Section 7: FriProof structure tests
// ============================================================================

TEST(FriProof, DefaultConstructible) {
    // Just verify the proof types compile and are default-constructible
    using Mmcs  = MockMmcs;
    FriProof<BB4, Mmcs, uint64_t, MockMmcsOpeningProof> proof;
    EXPECT_TRUE(proof.commit_phase_commits.empty());
    EXPECT_TRUE(proof.final_poly.empty());
    EXPECT_TRUE(proof.query_proofs.empty());

    CommitPhaseProofStep<BB4, Mmcs> step;
    EXPECT_EQ(step.log_arity, 0u);
    EXPECT_TRUE(step.sibling_values.empty());

    QueryProof<BB4, Mmcs, MockMmcsOpeningProof> qp;
    EXPECT_TRUE(qp.commit_phase_openings.empty());
}

// ============================================================================
// Section 8: Consistency — fold_row with extension field challenge
// ============================================================================

// Ensure fold_row works when Challenge = BB4 (the typical use case).
TEST(FriFolding, FoldRowExtensionChallenge) {
    using Val = BB;
    using Challenge = BB4;

    size_t log_height = 4;
    size_t log_arity  = 1;

    // Use extension field elements as evaluations and beta
    BB4 f0({BB(3u), BB(5u), BB::zero_val(), BB::zero_val()});
    BB4 f1({BB(7u), BB(2u), BB::zero_val(), BB::zero_val()});
    BB4 beta({BB(11u), BB(1u), BB::zero_val(), BB::zero_val()});

    std::vector<BB4> evals = {f0, f1};

    // Compute expected via the arity-2 formula: (f0+f1)/2 + beta*(f0-f1)/(2x)
    BB omega = BB::two_adic_generator(log_height);
    size_t x_pow = reverse_bits_len(0, log_height - log_arity);
    BB x_val = omega.exp_u64(static_cast<uint64_t>(x_pow));
    BB4 x = BB4::from_base(x_val);

    BB4 two_inv = BB4::from_base((BB::one_val() + BB::one_val()).inv());
    BB4 expected = (f0 + f1) * two_inv
                 + beta * (f0 - f1) * two_inv * x.inv();

    BB4 result = TwoAdicFriFolding<Val, Challenge>::fold_row(
        0, log_height, log_arity, beta, evals
    );
    EXPECT_EQ(result, expected);
}

// ============================================================================
// Main
// ============================================================================
// GTest provides its own main via GTest::gtest_main.

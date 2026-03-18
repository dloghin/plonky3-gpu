/**
 * @file test_two_adic_fri_pcs.cpp
 * @brief Google Test suite for TwoAdicFriPcs.
 *
 * Tests cover:
 *  1. natural_domain_for_degree
 *  2. commit (LDE + Merkle commitment)
 *  3. open / verify round-trip
 *  4. verify rejects tampered opened values
 */

#include <gtest/gtest.h>

#include "two_adic_fri_pcs.hpp"
#include "fri_params.hpp"

#include "baby_bear.hpp"
#include "extension_field.hpp"
#include "radix2_dit.hpp"
#include "p3_util/util.hpp"

#include <vector>
#include <memory>
#include <cstdint>
#include <type_traits>

using namespace p3_fri;
using namespace p3_field;
using namespace p3_util;

using BB  = BabyBear;
using BB4 = BabyBear4;

// ============================================================================
// PCS mock infrastructure
// ============================================================================

// A minimal MMCS for PCS tests. `TwoAdicFriPcs` only needs `commit_matrix()`
// for the input MMCS (Merkle commitments are not exercised by these unit tests).
struct PcsMmcsCommitment {
    uint32_t hash;
    bool operator==(const PcsMmcsCommitment& o) const { return hash == o.hash; }
};

struct PcsMmcsProverData {
    std::vector<BB> flat;
    size_t height = 0;
    size_t width  = 0;
};

// FRI commit-phase MMCS (BB4 Challenge elements, single-matrix commit_matrix interface)
struct FriMockMmcsOpeningProof { size_t row_index; };

struct FriMockMmcsCommitment {
    uint32_t hash;
    size_t   height;
    size_t   width;
    bool operator==(const FriMockMmcsCommitment& o) const {
        return hash == o.hash && height == o.height && width == o.width;
    }
};

struct FriMockMmcsProverData {
    std::vector<BB4> data;
    size_t height;
    size_t width;
};

struct FriMockMmcs {
    using Commitment   = FriMockMmcsCommitment;
    using ProverData   = FriMockMmcsProverData;
    using OpeningProof = FriMockMmcsOpeningProof;

    std::pair<Commitment, ProverData> commit_matrix(
        const std::vector<BB4>& vals, size_t width) const
    {
        size_t height = vals.size() / width;
        uint64_t acc = 0;
        for (const auto& v : vals)
            for (size_t k = 0; k < 4; ++k)
                acc = (acc + v[k].as_canonical_u64()) % BB::PRIME;
        Commitment c{ static_cast<uint32_t>(acc), height, width };
        ProverData d{ vals, height, width };
        return { c, d };
    }

    size_t log_width(const ProverData& d) const { return log2_strict_usize(d.width); }
    size_t log_height(const ProverData& d) const { return log2_strict_usize(d.height); }

    std::vector<BB4> get_row(const ProverData& d, size_t row) const {
        std::vector<BB4> r(d.width);
        for (size_t j = 0; j < d.width; ++j)
            r[j] = d.data[row * d.width + j];
        return r;
    }

    void open_row(const ProverData& /*pd*/, size_t row_index, OpeningProof& proof) const {
        proof.row_index = row_index;
    }

    bool verify_row(const Commitment& /*commit*/, size_t /*row*/,
                    const std::vector<BB4>& /*row_vals*/,
                    const OpeningProof& /*proof*/) const { return true; }

    bool verify_query(size_t /*query_index*/, size_t /*log_max_height*/,
                      const std::vector<Commitment>& /*commits*/,
                      const OpeningProof& /*proof*/,
                      const BB4& /*folded*/) const { return true; }

    void observe_commitment(const Commitment& /*commit*/) const {}
};

// The PCS input MMCS.  Handles multiple Val-typed matrices.
struct PcsInputMmcs {
    using Commitment   = PcsMmcsCommitment;
    using ProverData   = PcsMmcsProverData;

    void observe_commitment(const Commitment& /*commit*/) const {}

    std::pair<Commitment, ProverData> commit_matrix(
        const std::vector<BB>& vals, size_t width) const
    {
        uint64_t acc = 0;
        for (const auto& v : vals)
            acc = (acc + v.as_canonical_u64()) % BB::PRIME;
        Commitment c{ static_cast<uint32_t>(acc) };
        ProverData d;
        d.flat   = vals;
        d.width  = width;
        d.height = (width == 0) ? 0 : (vals.size() / width);
        return { c, d };
    }
};

// ============================================================================
// Mock Challenger (same as in test_fri.cpp but with BB4 challenge type)
// ============================================================================
struct PcsChallenger {
    uint64_t counter = 0;

private:
    static constexpr uint64_t MULT = 6364136223846793005ULL;
    static constexpr uint64_t INC = 1442695040888963407ULL;

    void mix_with_val(uint64_t val) {
        counter += val;
        counter = counter * MULT + INC;
    }

    void mix() {
        counter = counter * MULT + INC;
    }

public:
    void observe_commitment(const PcsMmcsCommitment& c) {
        mix_with_val(c.hash);
    }
    void observe_commitment(const FriMockMmcsCommitment& c) {
        mix_with_val(c.hash);
    }

    BB4 sample_challenge_bb4() {
        mix();
        uint32_t v0 = static_cast<uint32_t>(counter       % BB::PRIME);
        uint32_t v1 = static_cast<uint32_t>((counter >> 8) % BB::PRIME);
        uint32_t v2 = static_cast<uint32_t>((counter >>16) % BB::PRIME);
        uint32_t v3 = static_cast<uint32_t>((counter >>24) % BB::PRIME);
        return BB4({ BB(v0), BB(v1), BB(v2), BB(v3) });
    }

    template <typename EF = BB4>
    EF sample_challenge() {
        static_assert(std::is_same_v<EF, BB4>, "PcsChallenger only supports BB4 challenges");
        return sample_challenge_bb4();
    }

    size_t sample_bits(size_t bits) {
        mix();
        size_t mask = (bits >= 64) ? ~size_t(0) : ((size_t(1) << bits) - 1);
        return static_cast<size_t>(counter & mask);
    }

    uint64_t grind(size_t) { return 0; }
    bool check_witness(size_t, uint64_t) { return true; }

    void observe_challenge_bb4(const BB4& c) {
        for (size_t k = 0; k < 4; ++k) {
            mix_with_val(c[k].as_canonical_u64());
        }
    }

    template <typename EF>
    void observe_challenge(const EF& c) {
        static_assert(std::is_same_v<EF, BB4>, "PcsChallenger only supports BB4 challenges");
        observe_challenge_bb4(c);
    }

    void observe_arity(size_t la) { counter += la; }
};

// ============================================================================
// Type aliases
// ============================================================================
using Dft     = p3_dft::Radix2Dit<BB>;
using MyPcs   = TwoAdicFriPcs<BB, BB4, Dft, PcsInputMmcs, FriMockMmcs>;

// ============================================================================
// Test 1: natural_domain_for_degree
// ============================================================================
TEST(TwoAdicFriPcs, NaturalDomainForDegree) {
    Dft dft;
    PcsInputMmcs mmcs;
    FriMockMmcs fri_mmcs;
    FriParameters<FriMockMmcs> fri_params;
    fri_params.log_blowup = 1;
    fri_params.log_final_poly_len = 0;
    fri_params.max_log_arity = 1;
    fri_params.num_queries = 1;
    fri_params.commit_proof_of_work_bits = 0;
    fri_params.query_proof_of_work_bits = 0;
    fri_params.mmcs = fri_mmcs;

    MyPcs pcs(dft, mmcs, fri_params);

    auto coset = pcs.natural_domain_for_degree(8);
    EXPECT_EQ(coset.log_n, 3u);
    EXPECT_EQ(coset.shift, BB::one_val());

    auto coset2 = pcs.natural_domain_for_degree(4);
    EXPECT_EQ(coset2.log_n, 2u);
}

// ============================================================================
// Test 2: commit produces bit-reversed LDE of the right size
// ============================================================================
TEST(TwoAdicFriPcs, CommitProducesCorrectLDE) {
    Dft dft;
    PcsInputMmcs mmcs;
    FriMockMmcs fri_mmcs;
    FriParameters<FriMockMmcs> fri_params;
    fri_params.log_blowup = 1;  // blowup = 2
    fri_params.log_final_poly_len = 0;
    fri_params.max_log_arity = 1;
    fri_params.num_queries = 1;
    fri_params.commit_proof_of_work_bits = 0;
    fri_params.query_proof_of_work_bits = 0;
    fri_params.mmcs = fri_mmcs;

    MyPcs pcs(dft, mmcs, fri_params);

    size_t log_n = 3;  // domain size = 8
    size_t n     = size_t(1) << log_n;

    MyPcs::Domain domain{ log_n, BB::one_val() };

    // Simple polynomial: f(x) = 1 for all x (constant polynomial)
    p3_matrix::RowMajorMatrix<BB> eval_mat(n, 1, BB::one_val());

    auto [commit, pd] = pcs.commit({{ domain, eval_mat }});

    // The LDE should have height = n * blowup = 16
    (void)commit;
    ASSERT_EQ(pd.lde_matrices.size(), 1u);
    EXPECT_EQ(pd.lde_matrices[0].height(), n * 2);  // blowup = 2
    EXPECT_EQ(pd.lde_matrices[0].width(), 1u);

    // All LDE values of a constant polynomial are 1
    for (size_t i = 0; i < pd.lde_matrices[0].height(); ++i)
        EXPECT_EQ(pd.lde_matrices[0].get_unchecked(i, 0), BB::one_val());
}

// ============================================================================
// Test 3: commit -> open -> verify round-trip
// ============================================================================
TEST(TwoAdicFriPcs, OpenVerifyRoundTrip) {
    Dft dft;
    PcsInputMmcs mmcs;
    FriMockMmcs fri_mmcs;
    FriParameters<FriMockMmcs> fri_params;
    fri_params.log_blowup        = 1;  // blowup = 2
    fri_params.log_final_poly_len = 1;  // final poly len = 2
    fri_params.max_log_arity     = 1;
    fri_params.num_queries       = 1;
    fri_params.commit_proof_of_work_bits = 0;
    fri_params.query_proof_of_work_bits  = 0;
    fri_params.mmcs = fri_mmcs;

    MyPcs pcs(dft, mmcs, fri_params);

    // Polynomial: f = [1, 2, 3, 4] (evaluations in coefficient order,
    // but here we just treat them as LDE evaluations on a coset)
    size_t log_n = 2;
    size_t n     = size_t(1) << log_n;
    MyPcs::Domain domain{ log_n, BB::one_val() };

    std::vector<BB> evals_data(n);
    for (size_t i = 0; i < n; ++i)
        evals_data[i] = BB(static_cast<uint32_t>(i + 1));

    p3_matrix::RowMajorMatrix<BB> eval_mat(std::move(evals_data), 1);

    // Commit
    auto [commit, pd] = pcs.commit({{ domain, eval_mat }});

    // Prepare open data: open at one extension-field point
    BB4 z({ BB(5u), BB(3u), BB(1u), BB(0u) });

    PcsChallenger prover_challenger;
    std::vector<std::vector<BB4>> mat_points = { { z } };
    auto [opened_vals, proof] = pcs.open({ { &pd, mat_points } }, prover_challenger);

    // opened_vals[batch=0][point=0][col=0] is f(z)
    ASSERT_EQ(opened_vals.size(), 1u);
    ASSERT_EQ(opened_vals[0].size(), 1u);      // 1 matrix
    ASSERT_EQ(opened_vals[0][0].size(), 1u);   // 1 point
    ASSERT_EQ(opened_vals[0][0][0].size(), 1u); // 1 column

    // Verify
    MyPcs::VerifyCommitment vc;
    vc.commitment    = commit;
    vc.domains       = { domain };
    vc.points        = { { z } };
    vc.opened_values = { opened_vals[0][0] };  // [matrix][point][col]

    PcsChallenger verifier_challenger;  // fresh, same initial state
    bool ok = pcs.verify({ vc }, proof, verifier_challenger);
    EXPECT_TRUE(ok);
}

// ============================================================================
// Test 4: verify rejects tampered opened values
// ============================================================================
TEST(TwoAdicFriPcs, VerifyRejectsTamperedOpenedValues) {
    Dft dft;
    PcsInputMmcs mmcs;
    FriMockMmcs fri_mmcs;
    FriParameters<FriMockMmcs> fri_params;
    fri_params.log_blowup        = 1;
    fri_params.log_final_poly_len = 1;
    fri_params.max_log_arity     = 1;
    fri_params.num_queries       = 1;
    fri_params.commit_proof_of_work_bits = 0;
    fri_params.query_proof_of_work_bits  = 0;
    fri_params.mmcs = fri_mmcs;

    MyPcs pcs(dft, mmcs, fri_params);

    size_t log_n = 2;
    size_t n     = size_t(1) << log_n;
    MyPcs::Domain domain{ log_n, BB::one_val() };

    std::vector<BB> evals_data(n);
    for (size_t i = 0; i < n; ++i)
        evals_data[i] = BB(static_cast<uint32_t>(i + 1));

    p3_matrix::RowMajorMatrix<BB> eval_mat(std::move(evals_data), 1);
    auto [commit, pd] = pcs.commit({{ domain, eval_mat }});

    BB4 z({ BB(7u), BB(2u), BB(0u), BB(1u) });

    PcsChallenger prover_challenger;
    std::vector<std::vector<BB4>> mat_points = { { z } };
    auto [opened_vals, proof] = pcs.open({ { &pd, mat_points } }, prover_challenger);

    // Tamper: change the opened value
    auto tampered_vals = opened_vals;
    tampered_vals[0][0][0][0] = tampered_vals[0][0][0][0] + BB4::one_val();  // add 1

    MyPcs::VerifyCommitment vc;
    vc.commitment    = commit;
    vc.domains       = { domain };
    vc.points        = { { z } };
    vc.opened_values = { tampered_vals[0][0] };

    PcsChallenger verifier_challenger;
    bool ok = pcs.verify({ vc }, proof, verifier_challenger);
    // With tampered values the quotient won't match FRI
    EXPECT_FALSE(ok);
}

// ============================================================================
// Test 5: multi-column polynomial
// ============================================================================
TEST(TwoAdicFriPcs, MultiColumnRoundTrip) {
    Dft dft;
    PcsInputMmcs mmcs;
    FriMockMmcs fri_mmcs;
    FriParameters<FriMockMmcs> fri_params;
    fri_params.log_blowup        = 1;
    fri_params.log_final_poly_len = 1;
    fri_params.max_log_arity     = 1;
    fri_params.num_queries       = 1;
    fri_params.commit_proof_of_work_bits = 0;
    fri_params.query_proof_of_work_bits  = 0;
    fri_params.mmcs = fri_mmcs;

    MyPcs pcs(dft, mmcs, fri_params);

    size_t log_n = 2;
    size_t n     = size_t(1) << log_n;
    size_t w     = 2;
    MyPcs::Domain domain{ log_n, BB::one_val() };

    // Two polynomials: f0 = [1,2,3,4], f1 = [5,6,7,8]
    std::vector<BB> evals_data(n * w);
    for (size_t i = 0; i < n; ++i) {
        evals_data[i * w + 0] = BB(static_cast<uint32_t>(i + 1));
        evals_data[i * w + 1] = BB(static_cast<uint32_t>(i + 5));
    }
    p3_matrix::RowMajorMatrix<BB> eval_mat(std::move(evals_data), w);

    auto [commit, pd] = pcs.commit({{ domain, eval_mat }});

    BB4 z({ BB(3u), BB(1u), BB(0u), BB(0u) });

    PcsChallenger prover_challenger;
    std::vector<std::vector<BB4>> mat_points = { { z } };
    auto [opened_vals, proof] = pcs.open({ { &pd, mat_points } }, prover_challenger);

    ASSERT_EQ(opened_vals[0][0][0].size(), 2u);

    MyPcs::VerifyCommitment vc;
    vc.commitment    = commit;
    vc.domains       = { domain };
    vc.points        = { { z } };
    vc.opened_values = { opened_vals[0][0] };

    PcsChallenger verifier_challenger;
    bool ok = pcs.verify({ vc }, proof, verifier_challenger);
    EXPECT_TRUE(ok);
}

// ============================================================================
// Test 6: two opening points on same polynomial
// ============================================================================
TEST(TwoAdicFriPcs, TwoOpeningPointsRoundTrip) {
    Dft dft;
    PcsInputMmcs mmcs;
    FriMockMmcs fri_mmcs;
    FriParameters<FriMockMmcs> fri_params;
    fri_params.log_blowup        = 1;
    fri_params.log_final_poly_len = 1;
    fri_params.max_log_arity     = 1;
    fri_params.num_queries       = 1;
    fri_params.commit_proof_of_work_bits = 0;
    fri_params.query_proof_of_work_bits  = 0;
    fri_params.mmcs = fri_mmcs;

    MyPcs pcs(dft, mmcs, fri_params);

    size_t log_n = 2;
    size_t n     = size_t(1) << log_n;
    MyPcs::Domain domain{ log_n, BB::one_val() };

    std::vector<BB> evals_data(n);
    for (size_t i = 0; i < n; ++i)
        evals_data[i] = BB(static_cast<uint32_t>(i * 3 + 7));
    p3_matrix::RowMajorMatrix<BB> eval_mat(std::move(evals_data), 1);

    auto [commit, pd] = pcs.commit({{ domain, eval_mat }});

    BB4 z1({ BB(2u), BB(1u), BB(0u), BB(0u) });
    BB4 z2({ BB(9u), BB(0u), BB(1u), BB(0u) });

    PcsChallenger prover_challenger;
    std::vector<std::vector<BB4>> mat_points = { { z1, z2 } };
    auto [opened_vals, proof] = pcs.open({ { &pd, mat_points } }, prover_challenger);

    ASSERT_EQ(opened_vals[0][0].size(), 2u);

    MyPcs::VerifyCommitment vc;
    vc.commitment    = commit;
    vc.domains       = { domain };
    vc.points        = { { z1, z2 } };
    vc.opened_values = { opened_vals[0][0] };

    PcsChallenger verifier_challenger;
    bool ok = pcs.verify({ vc }, proof, verifier_challenger);
    EXPECT_TRUE(ok);
}

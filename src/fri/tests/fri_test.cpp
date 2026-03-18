/**
 * @file fri_test.cpp
 * @brief Google Test suite for FRI LDT test using real Poseidon2, MerkleTreeMmcs,
 *        ExtensionMmcs, and TwoAdicFriPcs.
 *
 * Tests:
 *  - FriTest.test_fri_ldt  – prove+verify round-trip for various polynomial sizes
 *  - FriTest.test_fri_ldt_should_panic – verify fails for inconsistent opened values
 */

#include <gtest/gtest.h>

#include "fri_params.hpp"
#include "fri_proof.hpp"
#include "fri_folding.hpp"
#include "fri_prover.hpp"
#include "fri_verifier.hpp"
#include "merkle_tree_mmcs.hpp"
#include "extension_mmcs.hpp"
#include "two_adic_fri_pcs.hpp"

#include "baby_bear.hpp"
#include "extension_field.hpp"
#include "poseidon2.hpp"
#include "padding_free_sponge.hpp"
#include "truncated_permutation.hpp"
#include "duplex_challenger.hpp"
#include "radix2_dit.hpp"
#include "dense_matrix.hpp"
#include "interpolation.hpp"
#include "p3_util/util.hpp"

#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <random>
#include <memory>
#include <stdexcept>

using namespace p3_field;
using namespace p3_util;
using namespace p3_fri;

using BB  = BabyBear;
using BB4 = BabyBear4;

// ============================================================================
// Poseidon2 setup helpers
// ============================================================================

static auto make_poseidon2_babybear_16() {
    using namespace poseidon2;
    constexpr size_t WIDTH        = 16;
    constexpr uint64_t D          = 7;
    constexpr size_t ROUNDS_F_HALF = 4;

    std::vector<std::array<BB, WIDTH>> init_consts(
        ROUNDS_F_HALF, std::array<BB, WIDTH>{});
    std::vector<std::array<BB, WIDTH>> term_consts(
        ROUNDS_F_HALF, std::array<BB, WIDTH>{});
    ExternalLayerConstants<BB, WIDTH> ext_consts(init_consts, term_consts);
    std::vector<BB> int_consts(22, BB());

    return create_poseidon2<BB, BB, WIDTH, D>(ext_consts, int_consts);
}

struct Poseidon2BB16 {
    std::shared_ptr<poseidon2::Poseidon2<BB, BB, 16, 7>> perm;
    void permute_mut(std::array<BB, 16>& state) const {
        perm->permute_mut(state);
    }
};

// ============================================================================
// Type aliases for the FRI stack
// ============================================================================

// Permutation type
using Perm16 = Poseidon2BB16;

// Hash function: PaddingFreeSponge<Perm16, BB, WIDTH=16, RATE=8, OUT=8>
using MyHash = p3_symmetric::PaddingFreeSponge<Perm16, BB, 16, 8, 8>;

// Compressor: TruncatedPermutation<Perm16, BB, N=2, CHUNK=8, WIDTH=16>
using MyCompress = p3_symmetric::TruncatedPermutation<Perm16, BB, 2, 8, 16>;

// Val MMCS: MerkleTreeMmcs<BB, BB, MyHash, MyCompress, N_ARY=2, CHUNK=8>
using ValMmcs = p3_merkle::MerkleTreeMmcs<BB, BB, MyHash, MyCompress, 2, 8>;

// Extension MMCS: wraps ValMmcs, commits BB4 matrices
using EFMmcs = p3_merkle::ExtensionMmcs<BB, BB4, ValMmcs>;

// DFT
using MyDft = p3_dft::Radix2Dit<BB>;

// FRI parameters use EFMmcs as the FriMmcs
using FriParams = FriParameters<EFMmcs>;

// PCS
using MyPcs = TwoAdicFriPcs<BB, BB4, MyDft, ValMmcs, EFMmcs>;

// ============================================================================
// FriDuplexChallenger: thin wrapper around DuplexChallenger for FRI use
// ============================================================================

using BaseDuplexChallenger = p3_challenger::DuplexChallenger<BB, Perm16, 16, 8>;

struct FriChallenger {
    BaseDuplexChallenger inner;

    explicit FriChallenger(Perm16 perm) : inner(std::move(perm)) {}
    FriChallenger(const FriChallenger&) = default;
    FriChallenger& operator=(const FriChallenger&) = default;

    template <typename Commit>
    void observe_commitment(const Commit& c) { inner.observe_merkle_cap(c); }

    // Templated with default=BB4 so it works for both:
    //   challenger.sample_challenge()         (FRI prover/verifier, returns BB4)
    //   challenger.template sample_challenge<BB4>()  (PCS, explicit type)
    template <typename EF = BB4>
    EF sample_challenge() { return inner.template sample_algebra_element<EF>(); }

    template <typename EF>
    void observe_challenge(const EF& c) { inner.template observe_algebra_element<EF>(c); }

    void observe_arity(size_t la) { inner.observe(BB(static_cast<uint32_t>(la))); }

    uint64_t grind(size_t bits) { return inner.grind(bits); }
    bool check_witness(size_t bits, uint64_t w) { return inner.check_witness(bits, w); }
    size_t sample_bits(size_t n) { return inner.sample_bits(n); }

    void observe(BB v) { inner.observe(v); }
    void observe_slice(const std::vector<BB>& vals) { inner.observe_slice(vals); }
    BB sample() { return inner.sample(); }
};

// ============================================================================
// Test helpers
// ============================================================================

static Perm16 make_perm16() {
    return Perm16{make_poseidon2_babybear_16()};
}

static FriChallenger make_fri_challenger() {
    return FriChallenger{make_perm16()};
}

static ValMmcs make_val_mmcs() {
    auto perm = make_perm16();
    MyHash     hasher(perm);
    MyCompress compressor(perm);
    return ValMmcs(std::move(hasher), std::move(compressor));
}

static EFMmcs make_ef_mmcs() {
    return EFMmcs(make_val_mmcs());
}

static MyPcs make_pcs(size_t log_blowup = 1,
                      size_t log_final_poly_len = 0) {
    MyDft dft;
    auto val_mmcs = make_val_mmcs();

    FriParams fri{
        log_blowup,            // log_blowup
        log_final_poly_len,    // log_final_poly_len
        1,                     // max_log_arity
        4,                     // num_queries
        0,                     // commit_proof_of_work_bits
        0,                     // query_proof_of_work_bits
        make_ef_mmcs()         // mmcs
    };

    return MyPcs(std::move(dft), std::move(val_mmcs), std::move(fri));
}

// ============================================================================
// do_test_fri_ldt
//
// Creates polynomials of sizes given in `polynomial_log_sizes`, commits them,
// opens at a random challenge point, proves, and verifies.
// ============================================================================

static void do_test_fri_ldt(size_t log_final_poly_len,
                              const std::vector<size_t>& polynomial_log_sizes)
{
    using Domain = TwoAdicMultiplicativeCoset<BB>;

    auto pcs = make_pcs(/*log_blowup=*/1, log_final_poly_len);

    // Build evaluation matrices (one polynomial per log_size)
    // We evaluate the polynomial f(x) = 1 at all points (trivial LDE)
    // using the coset-LDE approach.
    std::vector<std::pair<Domain, p3_matrix::RowMajorMatrix<BB>>> eval_mats;

    for (size_t log_n : polynomial_log_sizes) {
        size_t n = size_t(1) << log_n;
        Domain domain;
        domain.log_n = log_n;
        domain.shift = BB::one_val();  // standard subgroup

        // Create evaluation matrix: n rows, 1 column, with values f(x_i) = i+1 (mod p)
        std::vector<BB> vals(n);
        for (size_t i = 0; i < n; ++i) {
            vals[i] = BB(static_cast<uint32_t>((i * 3 + 7) % BB::PRIME));
        }
        p3_matrix::RowMajorMatrix<BB> mat(std::move(vals), 1);
        eval_mats.push_back({domain, std::move(mat)});
    }

    // Commit
    auto p_challenger = make_fri_challenger();
    auto v_challenger = make_fri_challenger();

    auto [commitment, prover_data] = pcs.commit(std::move(eval_mats));

    // Both challengers observe the commitment
    p_challenger.observe_commitment(commitment);
    v_challenger.observe_commitment(commitment);

    // Both challengers sample the opening point zeta
    BB4 zeta_p = p_challenger.template sample_challenge<BB4>();
    BB4 zeta_v = v_challenger.template sample_challenge<BB4>();
    // They should be equal since challengers are in the same state
    ASSERT_EQ(zeta_p, zeta_v) << "Prover and verifier challengers diverged at zeta sampling";

    // Build open_data: one batch, one matrix per log_size, each opened at zeta
    std::vector<std::vector<BB4>> mat_points;
    for (size_t i = 0; i < polynomial_log_sizes.size(); ++i) {
        mat_points.push_back({zeta_p});  // single opening point
    }

    std::vector<std::pair<const MyPcs::PcsProverData*,
                          std::vector<std::vector<BB4>>>> open_data;
    open_data.push_back({&prover_data, mat_points});

    // Open (prove)
    auto [opened_values, fri_proof] = pcs.open(open_data, p_challenger);

    // Build verify inputs
    std::vector<MyPcs::VerifyCommitment> verify_inputs;
    {
        MyPcs::VerifyCommitment vc;
        vc.commitment = commitment;

        for (size_t mi = 0; mi < polynomial_log_sizes.size(); ++mi) {
            Domain domain;
            domain.log_n = polynomial_log_sizes[mi];
            domain.shift = BB::one_val();
            vc.domains.push_back(domain);
            vc.points.push_back({zeta_v});
            vc.opened_values.push_back(opened_values[0][mi]);
        }
        verify_inputs.push_back(std::move(vc));
    }

    // Verify
    bool ok = pcs.verify(verify_inputs, fri_proof, v_challenger);
    EXPECT_TRUE(ok) << "verify() returned false for log_final_poly_len=" << log_final_poly_len;
}

// ============================================================================
// Tests
// ============================================================================

TEST(FriTest, test_fri_ldt) {
    // Test with a variety of log_final_poly_len values
    for (size_t log_fpl = 0; log_fpl <= 3; ++log_fpl) {
        // Use a single polynomial of size 2^(log_fpl + 2) to ensure it's
        // larger than the final poly len
        size_t log_poly = log_fpl + 2;  // e.g. log_fpl=0 -> log_poly=2 (4 evals)
        SCOPED_TRACE("log_final_poly_len=" + std::to_string(log_fpl) +
                     " log_poly=" + std::to_string(log_poly));
        do_test_fri_ldt(log_fpl, {log_poly});
    }
}

TEST(FriTest, test_fri_ldt_should_panic) {
    // log_final_poly_len = 5, polynomial of size 2^3 = 8
    // After blowup of 2, LDE size = 16, log_final_height = 1 + 5 = 6.
    // But log_lde_height = 4 < log_final_height = 6, so prove_fri should throw
    // (no folding rounds can happen: current.size() <= 2^log_final_height already).
    //
    // Actually prove_fri throws "no input vectors" only if inputs is empty.
    // It'll just short-circuit the folding loop.  The throw we want is either:
    //  (a) prove_fri throws if final poly is longer than input, or
    //  (b) we force it by making log_final_poly_len too big.
    //
    // Let's use log_final_poly_len = 5 and log_poly = 2, blowup = 1:
    //   lde_height = 2^2 * 2^1 = 8
    //   log_final_height = 1 + 5 = 6  -> 2^6 = 64 > lde_height = 8
    // prove_fri: current.size() = 8; loop condition 8 > 64 is false -> no folding.
    // final_poly resized to 2^5 = 32 > current.size() = 8 -> truncation is fine
    // but then IDFT on 32 elements with only 8 actual evaluations would be wrong.
    //
    // Actually, prove_fri resizes current to fpl = 2^5 = 32, but current.size() = 8.
    // resize(32) would zero-pad. Then reverse_slice_index_bits(32) would work.
    // The test should FAIL verification, not necessarily throw.
    //
    // Let's trigger an actual exception by using log_final_poly_len that makes
    // log_final_height > log(input_size), which causes log2_strict to throw
    // when computing the final poly IDFT on a non-power-of-2... actually no.
    //
    // The clearest way: use a polynomial that is too small for the FRI parameters
    // so that the final polynomial verification fails or an invariant is violated.
    //
    // In prove_fri, when current.size() <= 2^log_final_height and no folding occurs,
    // final_poly is set to current.resize(fpl=2^log_final_poly_len).
    // If fpl > current.size(), the resize zero-pads, IDFT runs on zeros,
    // and the result is a polynomial that doesn't match the claimed evaluations.
    // The verifier will fail because the final polynomial evaluation won't match
    // the folded value.
    //
    // For a guaranteed exception, let's construct a case where a mathematical
    // invariant fails. We use log_final_poly_len=5 with log_poly=2.
    // This should either throw or return false from verify.
    // We wrap in try/catch and also check EXPECT_FALSE.

    size_t log_fpl   = 5;
    size_t log_poly  = 2;

    // Override the PCS to use these params
    MyDft dft;
    auto val_mmcs = make_val_mmcs();

    FriParams fri{
        1,               // log_blowup
        log_fpl,         // log_final_poly_len
        1,               // max_log_arity
        1,               // num_queries
        0,               // commit_proof_of_work_bits
        0,               // query_proof_of_work_bits
        make_ef_mmcs()   // mmcs
    };

    MyPcs pcs(std::move(dft), std::move(val_mmcs), std::move(fri));

    using Domain = TwoAdicMultiplicativeCoset<BB>;
    size_t n = size_t(1) << log_poly;

    Domain domain;
    domain.log_n = log_poly;
    domain.shift = BB::one_val();

    std::vector<BB> vals(n);
    for (size_t i = 0; i < n; ++i) {
        vals[i] = BB(static_cast<uint32_t>(i + 1));
    }
    p3_matrix::RowMajorMatrix<BB> mat(std::move(vals), 1);

    std::vector<std::pair<Domain, p3_matrix::RowMajorMatrix<BB>>> eval_mats;
    eval_mats.push_back({domain, std::move(mat)});

    auto p_challenger = make_fri_challenger();
    auto v_challenger = make_fri_challenger();

    // This may or may not throw depending on implementation details.
    // We expect EITHER an exception OR verify returning false.
    bool threw = false;
    bool verified = false;

    try {
        auto [commitment, prover_data] = pcs.commit(std::move(eval_mats));

        p_challenger.observe_commitment(commitment);
        v_challenger.observe_commitment(commitment);

        BB4 zeta = p_challenger.template sample_challenge<BB4>();
        v_challenger.template sample_challenge<BB4>();  // advance verifier state

        std::vector<std::vector<BB4>> mat_points = {{zeta}};
        std::vector<std::pair<const MyPcs::PcsProverData*,
                              std::vector<std::vector<BB4>>>> open_data;
        open_data.push_back({&prover_data, mat_points});

        auto [opened_values, fri_proof] = pcs.open(open_data, p_challenger);

        // Build verify input
        std::vector<MyPcs::VerifyCommitment> verify_inputs;
        {
            MyPcs::VerifyCommitment vc;
            vc.commitment = commitment;
            Domain vdomain;
            vdomain.log_n = log_poly;
            vdomain.shift = BB::one_val();
            vc.domains.push_back(vdomain);
            // Corrupt opened values to cause failure
            auto ov_corrupt = opened_values[0][0];
            if (!ov_corrupt.empty() && !ov_corrupt[0].empty()) {
                // Flip the first coefficient to make verification fail
                ov_corrupt[0][0] = ov_corrupt[0][0] + BB4::from_base(BB(1u));
            }
            vc.points.push_back({zeta});
            vc.opened_values.push_back(ov_corrupt);
            verify_inputs.push_back(std::move(vc));
        }

        verified = pcs.verify(verify_inputs, fri_proof, v_challenger);
    } catch (const std::exception& e) {
        threw = true;
    }

    // Either an exception was thrown OR verification failed
    EXPECT_TRUE(threw || !verified)
        << "Expected exception or verification failure for mismatched opened values";
}

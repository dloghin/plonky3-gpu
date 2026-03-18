/**
 * @file fri_test.cpp
 * @brief Google Test suite for FRI LDT end-to-end test using real Poseidon2,
 *        MerkleTreeMmcs, ExtensionMmcs, and TwoAdicFriPcs.
 *
 * Mirrors plonky3/fri/tests/fri.rs.
 *
 * Tests:
 *  - FriTest.test_fri_ldt  – prove+verify round-trip for various polynomial
 *    sizes with log_final_poly_len in {0, 1, 2, 3, 4}.
 *  - FriTest.test_fri_ldt_should_panic – should panic/throw when
 *    log_final_poly_len is too large for the smallest polynomial.
 */

#include <gtest/gtest.h>

#include "fri_params.hpp"
#include "fri_proof.hpp"
#include "fri_folding.hpp"
#include "fri_prover.hpp"
#include "fri_verifier.hpp"
#include "fri_merkle_tree_mmcs.hpp"
#include "fri_extension_mmcs.hpp"
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

using Perm16      = Poseidon2BB16;
using MyHash      = p3_symmetric::PaddingFreeSponge<Perm16, BB, 16, 8, 8>;
using MyCompress  = p3_symmetric::TruncatedPermutation<Perm16, BB, 2, 8, 16>;
using ValMmcs     = p3_merkle::MerkleTreeMmcs<BB, BB, MyHash, MyCompress, 2, 8>;
using EFMmcs      = p3_merkle::ExtensionMmcs<BB, BB4, ValMmcs>;
using MyDft       = p3_dft::Radix2Dit<BB>;
using FriParams   = FriParameters<EFMmcs>;
using MyPcs       = TwoAdicFriPcs<BB, BB4, MyDft, ValMmcs, EFMmcs>;

// ============================================================================
// FriChallenger: thin wrapper around DuplexChallenger for FRI use
// ============================================================================

using BaseDuplexChallenger = p3_challenger::DuplexChallenger<BB, Perm16, 16, 8>;

struct FriChallenger {
    BaseDuplexChallenger inner;

    explicit FriChallenger(Perm16 perm) : inner(std::move(perm)) {}
    FriChallenger(const FriChallenger&) = default;
    FriChallenger& operator=(const FriChallenger&) = default;

    template <typename Commit>
    void observe_commitment(const Commit& c) { inner.observe_merkle_cap(c); }

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

// Create PCS matching the Rust test: log_blowup=1, max_log_arity=1,
// num_queries=10, query_pow_bits=8.
static MyPcs make_pcs(size_t log_final_poly_len) {
    MyDft dft;
    auto val_mmcs = make_val_mmcs();

    FriParams fri{
        1,                     // log_blowup
        log_final_poly_len,    // log_final_poly_len
        1,                     // max_log_arity  (matches Rust test)
        10,                    // num_queries    (matches Rust test)
        0,                     // commit_proof_of_work_bits
        8,                     // query_proof_of_work_bits (matches Rust test)
        make_ef_mmcs()         // mmcs
    };

    return MyPcs(std::move(dft), std::move(val_mmcs), std::move(fri));
}

// Generate a deterministic non-zero matrix of size (rows x cols).
static p3_matrix::RowMajorMatrix<BB> rand_nonzero_matrix(
    std::mt19937_64& rng, size_t rows, size_t cols)
{
    std::uniform_int_distribution<uint32_t> dist(1, BB::PRIME - 1);
    std::vector<BB> vals(rows * cols);
    for (auto& v : vals) {
        v = BB(dist(rng));
    }
    return p3_matrix::RowMajorMatrix<BB>(std::move(vals), cols);
}

// ============================================================================
// do_test_fri_ldt
//
// Creates polynomials of sizes given in `polynomial_log_sizes`, commits them,
// opens at a random challenge point, proves, and verifies.
// Mirrors plonky3/fri/tests/fri.rs::do_test_fri_ldt.
// ============================================================================

static void do_test_fri_ldt(uint64_t seed,
                            size_t log_final_poly_len,
                            const std::vector<uint8_t>& polynomial_log_sizes)
{
    using Domain = TwoAdicMultiplicativeCoset<BB>;

    auto pcs = make_pcs(log_final_poly_len);

    // Convert polynomial_log_sizes to field elements for observation
    std::vector<BB> val_sizes;
    val_sizes.reserve(polynomial_log_sizes.size());
    for (auto s : polynomial_log_sizes) {
        val_sizes.push_back(BB(static_cast<uint32_t>(s)));
    }

    // RNG for generating random evaluation matrices
    std::mt19937_64 rng(seed);

    // --- Prover World ---
    auto p_challenger = make_fri_challenger();
    p_challenger.observe_slice(val_sizes);

    // Generate random evaluation matrices (one per polynomial)
    std::vector<std::pair<Domain, p3_matrix::RowMajorMatrix<BB>>> eval_mats;
    for (auto deg_bits : polynomial_log_sizes) {
        size_t deg = size_t(1) << deg_bits;
        Domain domain;
        domain.log_n = deg_bits;
        domain.shift = BB::one_val();

        auto mat = rand_nonzero_matrix(rng, deg, 16);
        eval_mats.push_back({domain, std::move(mat)});
    }

    size_t num_evaluations = eval_mats.size();

    // Commit
    auto [commitment, prover_data] = pcs.commit(std::move(eval_mats));

    p_challenger.observe_commitment(commitment);

    // Sample opening point zeta
    BB4 zeta = p_challenger.template sample_challenge<BB4>();

    // Build open_data: one batch, every polynomial opened at zeta
    std::vector<std::vector<BB4>> mat_points;
    for (size_t i = 0; i < num_evaluations; ++i) {
        mat_points.push_back({zeta});
    }

    std::vector<std::pair<const MyPcs::PcsProverData*,
                          std::vector<std::vector<BB4>>>> open_data;
    open_data.push_back({&prover_data, mat_points});

    // Open (prove)
    auto [opened_values, fri_proof] = pcs.open(open_data, p_challenger);

    // --- Verifier World ---
    auto v_challenger = make_fri_challenger();
    v_challenger.observe_slice(val_sizes);
    v_challenger.observe_commitment(commitment);

    BB4 v_zeta = v_challenger.template sample_challenge<BB4>();
    ASSERT_EQ(zeta, v_zeta)
        << "Prover and verifier challengers diverged at zeta sampling";

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
            vc.points.push_back({v_zeta});
            vc.opened_values.push_back(opened_values[0][mi]);
        }
        verify_inputs.push_back(std::move(vc));
    }

    // Verify
    bool ok = pcs.verify(verify_inputs, fri_proof, v_challenger);
    EXPECT_TRUE(ok) << "verify() returned false for log_final_poly_len="
                    << log_final_poly_len << " seed=" << seed;

    // Check prover and verifier challengers agree
    EXPECT_EQ(p_challenger.sample_bits(8), v_challenger.sample_bits(8))
        << "Prover and verifier transcript mismatch after FRI "
        << "(log_final_poly_len=" << log_final_poly_len << " seed=" << seed << ")";
}

// ============================================================================
// Tests
// ============================================================================

TEST(FriTest, test_fri_ldt) {
    // Matches the Rust test: multiple polynomials of varying sizes,
    // including duplicates and out-of-order.
    std::vector<uint8_t> polynomial_log_sizes = {5, 8, 10, 7, 5, 5, 7};
    for (size_t i = 0; i < 5; ++i) {
        SCOPED_TRACE("log_final_poly_len=" + std::to_string(i) +
                     " seed=" + std::to_string(i));
        do_test_fri_ldt(i, i, polynomial_log_sizes);
    }
}

TEST(FriTest, test_fri_ldt_should_panic) {
    // log_final_poly_len = 5: smallest polynomial has degree 2^5 = 32.
    // After blowup (2x), its LDE height is 64 = 2^6.
    // log_final_height = log_blowup + log_final_poly_len = 1 + 5 = 6.
    // The smallest LDE (height 64) equals the final height, meaning no
    // folding is possible for that input.  The Rust implementation asserts
    // that log_min_height > log_final_poly_len + log_blowup, so this
    // should either throw or produce a proof that doesn't verify.
    std::vector<uint8_t> polynomial_log_sizes = {5, 8, 10, 7, 5, 5, 7};

    bool threw = false;
    bool verified = false;

    try {
        // This may throw during commit, open, or prove
        do_test_fri_ldt(5, 5, polynomial_log_sizes);
        // If do_test_fri_ldt completes without assertion failure, it means
        // the EXPECT_TRUE inside it passed.  We consider that a "verified"
        // result for the purpose of this test.
        verified = true;
    } catch (const std::exception&) {
        threw = true;
    }

    EXPECT_TRUE(threw || !verified)
        << "Expected exception or verification failure for "
           "log_final_poly_len=5 with polynomial_log_sizes containing 5";
}

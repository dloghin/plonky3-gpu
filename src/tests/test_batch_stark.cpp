/**
 * @file test_batch_stark.cpp
 * @brief End-to-end tests for the batch-STARK prover/verifier (task 28).
 *
 * Covers:
 *   - Multi-AIR batch prove + verify round-trip (Fibonacci + Square).
 *   - Rejection of tampered per-instance openings / commitments.
 *   - Rejection of invalid traces (e.g. broken Fibonacci recurrence).
 *   - Mixed trace sizes across instances.
 *   - Per-instance public values bind into the shared transcript.
 *   - Batch proof payload is materially smaller than the sum of uni-STARK proofs.
 */

#include <gtest/gtest.h>

#include "batch_stark_prover.hpp"
#include "batch_stark_verifier.hpp"
#include "batch_stark_proof.hpp"
#include "stark_config.hpp"
#include "stark_prover.hpp"
#include "stark_verifier.hpp"
#include "constraint_folder.hpp"

#include "two_adic_fri_pcs.hpp"
#include "fri_params.hpp"
#include "baby_bear.hpp"
#include "extension_field.hpp"
#include "radix2_dit.hpp"
#include "air.hpp"
#include "p3_util/util.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

using namespace p3_batch_stark;
using namespace p3_fri;
using namespace p3_field;

using BB  = BabyBear;
using BB4 = BabyBear4;

// ============================================================================
// Mock MMCS + Challenger (same pattern as test_uni_stark.cpp).
// ============================================================================

struct PcsMmcsCommitment {
    uint32_t hash = 0;
    bool operator==(const PcsMmcsCommitment& o) const { return hash == o.hash; }
};

struct PcsMmcsProverData {
    std::vector<BB> flat;
    size_t height = 0;
    size_t width  = 0;
};

struct FriMockMmcsOpeningProof { size_t row_index = 0; };

struct FriMockMmcsCommitment {
    uint32_t hash   = 0;
    size_t   height = 0;
    size_t   width  = 0;
    bool operator==(const FriMockMmcsCommitment& o) const {
        return hash == o.hash && height == o.height && width == o.width;
    }
};

struct FriMockMmcsProverData {
    std::vector<BB4> data;
    size_t height = 0;
    size_t width  = 0;
};

struct FriMockMmcs {
    using Commitment   = FriMockMmcsCommitment;
    using ProverData   = FriMockMmcsProverData;
    using OpeningProof = FriMockMmcsOpeningProof;

    std::pair<Commitment, ProverData> commit_matrix(
        const std::vector<BB4>& vals, size_t width) const
    {
        size_t height = (width == 0) ? 0 : vals.size() / width;
        uint64_t acc = 0;
        for (const auto& v : vals)
            for (size_t k = 0; k < 4; ++k)
                acc = (acc + v[k].as_canonical_u64()) % BB::PRIME;
        return { Commitment{ static_cast<uint32_t>(acc), height, width },
                 ProverData{ vals, height, width } };
    }

    size_t log_width (const ProverData& d) const { return p3_util::log2_strict_usize(d.width);  }
    size_t log_height(const ProverData& d) const { return p3_util::log2_strict_usize(d.height); }

    std::vector<BB4> get_row(const ProverData& d, size_t row) const {
        std::vector<BB4> r(d.width);
        for (size_t j = 0; j < d.width; ++j) r[j] = d.data[row * d.width + j];
        return r;
    }

    void open_row(const ProverData&, size_t row_index, OpeningProof& proof) const {
        proof.row_index = row_index;
    }
    bool verify_row(const Commitment&, size_t,
                    const std::vector<BB4>&, const OpeningProof&) const { return true; }
    bool verify_query(size_t, size_t,
                      const std::vector<Commitment>&,
                      const OpeningProof&, const BB4&) const { return true; }

    void observe_commitment(const Commitment&) const {}
};

struct PcsInputMmcs {
    using Commitment = PcsMmcsCommitment;
    using ProverData = PcsMmcsProverData;

    void observe_commitment(const Commitment&) const {}

    std::pair<Commitment, ProverData> commit_matrix(
        const std::vector<BB>& vals, size_t width) const
    {
        uint64_t acc = 0;
        for (const auto& v : vals) acc = (acc + v.as_canonical_u64()) % BB::PRIME;
        PcsMmcsProverData d;
        d.flat   = vals;
        d.width  = width;
        d.height = (width == 0) ? 0 : vals.size() / width;
        return { Commitment{ static_cast<uint32_t>(acc) }, d };
    }
};

// Mock Fiat-Shamir challenger. Deterministic LCG mixed with observations.
struct MockChallenger {
    uint64_t counter = 0;

private:
    static constexpr uint64_t MULT = 6364136223846793005ULL;
    static constexpr uint64_t INC  = 1442695040888963407ULL;
    void mix_with_val(uint64_t v) { counter += v; counter = counter * MULT + INC; }
    void mix() { counter = counter * MULT + INC; }

public:
    void observe_val(const BB& v) { mix_with_val(v.as_canonical_u64()); }

    void observe_commitment(const PcsMmcsCommitment& c)     { mix_with_val(c.hash); }
    void observe_commitment(const FriMockMmcsCommitment& c) { mix_with_val(c.hash); }

    BB4 sample_challenge_bb4() {
        mix();
        uint32_t v0 = static_cast<uint32_t>( counter        % BB::PRIME);
        uint32_t v1 = static_cast<uint32_t>((counter >>  8) % BB::PRIME);
        uint32_t v2 = static_cast<uint32_t>((counter >> 16) % BB::PRIME);
        uint32_t v3 = static_cast<uint32_t>((counter >> 24) % BB::PRIME);
        return BB4({ BB(v0), BB(v1), BB(v2), BB(v3) });
    }

    template <typename EF = BB4>
    EF sample_challenge() {
        static_assert(std::is_same_v<EF, BB4>, "MockChallenger: only BB4");
        return sample_challenge_bb4();
    }

    size_t sample_bits(size_t bits) {
        mix();
        size_t mask = (bits >= 64) ? ~size_t(0) : ((size_t(1) << bits) - 1);
        return static_cast<size_t>(counter & mask);
    }

    uint64_t grind(size_t) { return 0; }
    bool     check_witness(size_t, uint64_t) { return true; }

    void observe_challenge_bb4(const BB4& c) {
        for (size_t k = 0; k < 4; ++k) mix_with_val(c[k].as_canonical_u64());
    }
    template <typename EF>
    void observe_challenge(const EF& c) {
        static_assert(std::is_same_v<EF, BB4>, "MockChallenger: only BB4");
        observe_challenge_bb4(c);
    }

    void observe_arity(size_t la) { counter += la; }
};

using Dft    = p3_dft::Radix2Dit<BB>;
using MyPcs  = TwoAdicFriPcs<BB, BB4, Dft, PcsInputMmcs, FriMockMmcs>;
using MyCfg  = p3_uni_stark::StarkConfig<BB, BB4, MyPcs, MockChallenger>;

static FriParameters<FriMockMmcs> make_fri_params() {
    FriParameters<FriMockMmcs> p;
    p.log_blowup                 = 1;
    p.log_final_poly_len         = 1;
    p.max_log_arity              = 1;
    p.num_queries                = 2;
    p.commit_proof_of_work_bits  = 0;
    p.query_proof_of_work_bits   = 0;
    p.mmcs                       = FriMockMmcs{};
    return p;
}

static MyPcs make_pcs() {
    Dft dft;
    PcsInputMmcs mmcs;
    return MyPcs(dft, mmcs, make_fri_params());
}

// ============================================================================
// AIRs (duck-typed)
// ============================================================================

// Fibonacci: next(0) = cur(1); next(1) = cur(0) + cur(1); boundary cur(0)=cur(1)=1.
// Constraint polynomial degree = 2.
struct FibonacciAir {
    std::size_t width()             const { return 2; }
    std::size_t constraint_degree() const { return 2; }

    template <typename AB>
    void eval(AB& builder) const {
        using Expr = typename AB::ExprType;
        using Field = typename AB::Field;
        auto main = builder.main();
        {
            auto t = builder.when_transition();
            t.assert_eq(main.next(0), main.current(1));
            t.assert_eq(main.next(1), main.current(0) + main.current(1));
        }
        {
            auto f = builder.when_first_row();
            f.assert_eq(main.current(0), Expr(Field(1)));
            f.assert_eq(main.current(1), Expr(Field(1)));
        }
    }
};

// Squaring AIR: next(0) = cur(0)^2. Constraint polynomial degree = 3 → K = 2N.
struct SquareAir {
    std::size_t width()             const { return 1; }
    std::size_t constraint_degree() const { return 3; }

    template <typename AB>
    void eval(AB& builder) const {
        auto main = builder.main();
        auto t = builder.when_transition();
        t.assert_eq(main.next(0), main.current(0) * main.current(0));
    }
};

// Tagged-union AIR that can dispatch to either Fibonacci or Square constraint
// logic. Used to mix different AIR behaviors in a single batch (prove_batch is
// templated on a single AIR type).
struct MixedAir {
    enum Kind { Fib, Square };
    Kind   kind;
    std::size_t air_width;
    std::size_t air_cdeg;

    static MixedAir fibo()   { return MixedAir{Fib,    2, 2}; }
    static MixedAir square() { return MixedAir{Square, 1, 3}; }

    std::size_t width()             const { return air_width; }
    std::size_t constraint_degree() const { return air_cdeg; }

    template <typename AB>
    void eval(AB& builder) const {
        using Expr = typename AB::ExprType;
        using Field = typename AB::Field;
        auto main = builder.main();
        if (kind == Fib) {
            auto t = builder.when_transition();
            t.assert_eq(main.next(0), main.current(1));
            t.assert_eq(main.next(1), main.current(0) + main.current(1));
            auto f = builder.when_first_row();
            f.assert_eq(main.current(0), Expr(Field(1)));
            f.assert_eq(main.current(1), Expr(Field(1)));
        } else {
            auto t = builder.when_transition();
            t.assert_eq(main.next(0), main.current(0) * main.current(0));
        }
    }
};

// ============================================================================
// Trace builders
// ============================================================================

static p3_matrix::RowMajorMatrix<BB> build_fibonacci_trace(std::size_t n) {
    std::vector<BB> data(n * 2);
    BB a(1), b(1);
    for (std::size_t i = 0; i < n; ++i) {
        data[2 * i + 0] = a;
        data[2 * i + 1] = b;
        BB c = a + b;
        a = b;
        b = c;
    }
    return p3_matrix::RowMajorMatrix<BB>(std::move(data), 2);
}

static p3_matrix::RowMajorMatrix<BB> build_square_trace(std::size_t n, BB seed) {
    std::vector<BB> data(n);
    BB cur = seed;
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = cur;
        cur = cur * cur;
    }
    return p3_matrix::RowMajorMatrix<BB>(std::move(data), 1);
}

// Size (rough) of a FullFriProof: sum of commit_phase_commits + final_poly +
// sum over query proofs of (rows + siblings). Since our mock MMCS is trivial,
// we use the high-level fields as a proxy.
template <typename SC>
static std::size_t rough_proof_size(const p3_uni_stark::Proof<SC>& proof) {
    const auto& fp = proof.opening_proof;
    std::size_t sz = 0;
    sz += fp.commit_phase_commits.size();
    sz += fp.final_poly.size();
    sz += fp.query_proofs.size();
    for (const auto& qp : fp.query_proofs) {
        sz += qp.input_proof.row_values.size();
        for (const auto& step : qp.commit_phase_openings) {
            sz += step.sibling_values.size();
            (void)step;
        }
    }
    sz += proof.opened_values.trace_local.size();
    sz += proof.opened_values.trace_next.size();
    sz += proof.opened_values.quotient_chunks.size();
    return sz;
}

template <typename SC>
static std::size_t rough_proof_size(const BatchProof<SC>& proof) {
    const auto& fp = proof.opening_proof;
    std::size_t sz = 0;
    sz += fp.commit_phase_commits.size();
    sz += fp.final_poly.size();
    sz += fp.query_proofs.size();
    for (const auto& qp : fp.query_proofs) {
        sz += qp.input_proof.row_values.size();
        for (const auto& step : qp.commit_phase_openings) {
            sz += step.sibling_values.size();
            (void)step;
        }
    }
    for (const auto& inst : proof.opened_values.instances) {
        sz += inst.trace_local.size();
        sz += inst.trace_next.size();
        sz += inst.quotient_chunks.size();
    }
    return sz;
}

// ============================================================================
// Tests
// ============================================================================

TEST(BatchStark, TwoFibonacciBatchRoundTrip) {
    MyCfg config(make_pcs());
    FibonacciAir air;

    auto trace_a = build_fibonacci_trace(4);
    auto trace_b = build_fibonacci_trace(4);

    std::vector<StarkInstance<MyCfg, FibonacciAir>> instances = {
        StarkInstance<MyCfg, FibonacciAir>(air, trace_a),
        StarkInstance<MyCfg, FibonacciAir>(air, trace_b),
    };

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove_batch(config, instances, prover_ch, dft);

    EXPECT_EQ(proof.opened_values.instances.size(), 2u);
    EXPECT_EQ(proof.degree_bits.size(), 2u);

    MockChallenger verifier_ch;
    std::vector<const FibonacciAir*> airs = { &air, &air };
    EXPECT_TRUE(verify_batch(config, airs, verifier_ch, proof));
}

TEST(BatchStark, MixedFibonacciAndSquareRoundTrip) {
    MyCfg config(make_pcs());
    auto fib_trace = build_fibonacci_trace(4);
    auto sq_trace  = build_square_trace(4, BB(2));

    MixedAir air0 = MixedAir::fibo();
    MixedAir air1 = MixedAir::square();

    std::vector<StarkInstance<MyCfg, MixedAir>> instances = {
        StarkInstance<MyCfg, MixedAir>(air0, fib_trace),
        StarkInstance<MyCfg, MixedAir>(air1, sq_trace),
    };

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove_batch(config, instances, prover_ch, dft);

    ASSERT_EQ(proof.opened_values.instances.size(), 2u);
    // Fibonacci: constraint_degree=2 => log_num_quotient_chunks=0 (N chunks=1).
    EXPECT_EQ(proof.log_num_quotient_chunks[0], 0u);
    // Square: constraint_degree=3 => log_num_quotient_chunks=1 (K=2N).
    EXPECT_EQ(proof.log_num_quotient_chunks[1], 1u);

    MockChallenger verifier_ch;
    std::vector<const MixedAir*> airs = { &air0, &air1 };
    EXPECT_TRUE(verify_batch(config, airs, verifier_ch, proof));
}

TEST(BatchStark, RejectsTamperedTraceOpening) {
    MyCfg config(make_pcs());
    FibonacciAir air;
    auto trace_a = build_fibonacci_trace(4);
    auto trace_b = build_fibonacci_trace(4);

    std::vector<StarkInstance<MyCfg, FibonacciAir>> instances = {
        StarkInstance<MyCfg, FibonacciAir>(air, trace_a),
        StarkInstance<MyCfg, FibonacciAir>(air, trace_b),
    };

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove_batch(config, instances, prover_ch, dft);

    // Tamper instance 1's trace_local.
    auto tampered = proof;
    tampered.opened_values.instances[1].trace_local[0] =
        tampered.opened_values.instances[1].trace_local[0] + BB4::one_val();

    MockChallenger verifier_ch;
    std::vector<const FibonacciAir*> airs = { &air, &air };
    EXPECT_FALSE(verify_batch(config, airs, verifier_ch, tampered));
}

TEST(BatchStark, RejectsTamperedQuotientChunk) {
    MyCfg config(make_pcs());
    FibonacciAir air;
    auto trace_a = build_fibonacci_trace(4);
    auto trace_b = build_fibonacci_trace(4);

    std::vector<StarkInstance<MyCfg, FibonacciAir>> instances = {
        StarkInstance<MyCfg, FibonacciAir>(air, trace_a),
        StarkInstance<MyCfg, FibonacciAir>(air, trace_b),
    };

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove_batch(config, instances, prover_ch, dft);

    auto tampered = proof;
    tampered.opened_values.instances[0].quotient_chunks[0] =
        tampered.opened_values.instances[0].quotient_chunks[0] + BB4::one_val();

    MockChallenger verifier_ch;
    std::vector<const FibonacciAir*> airs = { &air, &air };
    EXPECT_FALSE(verify_batch(config, airs, verifier_ch, tampered));
}

TEST(BatchStark, RejectsTamperedMainCommitment) {
    MyCfg config(make_pcs());
    FibonacciAir air;
    auto trace_a = build_fibonacci_trace(4);
    auto trace_b = build_fibonacci_trace(4);

    std::vector<StarkInstance<MyCfg, FibonacciAir>> instances = {
        StarkInstance<MyCfg, FibonacciAir>(air, trace_a),
        StarkInstance<MyCfg, FibonacciAir>(air, trace_b),
    };

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove_batch(config, instances, prover_ch, dft);

    auto tampered = proof;
    tampered.commitments.main.hash ^= 0xABCD1234u;

    MockChallenger verifier_ch;
    std::vector<const FibonacciAir*> airs = { &air, &air };
    EXPECT_FALSE(verify_batch(config, airs, verifier_ch, tampered));
}

TEST(BatchStark, RejectsInvalidSingleInstanceTrace) {
    // If instance 1 contains a bogus Fibonacci trace, prove/verify must fail.
    MyCfg config(make_pcs());
    FibonacciAir air;

    auto good = build_fibonacci_trace(4);

    // Build a trace that violates the transition constraint at row 1.
    std::vector<BB> bad_data(4 * 2);
    BB a(1), b(1);
    for (std::size_t i = 0; i < 4; ++i) {
        bad_data[2 * i + 0] = a;
        bad_data[2 * i + 1] = b;
        BB c = a + b;
        if (i == 1) {
            c = c + BB(1);  // wrong
        }
        a = b;
        b = c;
    }
    p3_matrix::RowMajorMatrix<BB> bad(std::move(bad_data), 2);

    std::vector<StarkInstance<MyCfg, FibonacciAir>> instances = {
        StarkInstance<MyCfg, FibonacciAir>(air, good),
        StarkInstance<MyCfg, FibonacciAir>(air, bad),
    };

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove_batch(config, instances, prover_ch, dft);

    MockChallenger verifier_ch;
    std::vector<const FibonacciAir*> airs = { &air, &air };
    EXPECT_FALSE(verify_batch(config, airs, verifier_ch, proof));
}

TEST(BatchStark, PublicValuesBindPerInstance) {
    MyCfg config(make_pcs());
    FibonacciAir air;

    auto trace_a = build_fibonacci_trace(4);
    auto trace_b = build_fibonacci_trace(4);

    std::vector<StarkInstance<MyCfg, FibonacciAir>> instances = {
        StarkInstance<MyCfg, FibonacciAir>(air, trace_a, std::vector<BB>{BB(7), BB(8)}),
        StarkInstance<MyCfg, FibonacciAir>(air, trace_b, std::vector<BB>{BB(9)}),
    };

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove_batch(config, instances, prover_ch, dft);

    std::vector<const FibonacciAir*> airs = { &air, &air };
    std::vector<std::vector<BB>> pvs = { {BB(7), BB(8)}, {BB(9)} };

    {
        MockChallenger verifier_ch;
        EXPECT_TRUE(verify_batch(config, airs, verifier_ch, proof, pvs));
    }
    {
        MockChallenger verifier_ch;
        std::vector<std::vector<BB>> wrong = { {BB(7), BB(8)}, {BB(42)} };
        EXPECT_FALSE(verify_batch(config, airs, verifier_ch, proof, wrong));
    }
}

TEST(BatchStark, MixedDegreesRoundTrip) {
    MyCfg config(make_pcs());
    FibonacciAir air;

    auto trace_small = build_fibonacci_trace(4);
    auto trace_large = build_fibonacci_trace(8);

    std::vector<StarkInstance<MyCfg, FibonacciAir>> instances = {
        StarkInstance<MyCfg, FibonacciAir>(air, trace_small),
        StarkInstance<MyCfg, FibonacciAir>(air, trace_large),
    };

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove_batch(config, instances, prover_ch, dft);

    ASSERT_EQ(proof.degree_bits.size(), 2u);
    EXPECT_EQ(proof.degree_bits[0], 2u);
    EXPECT_EQ(proof.degree_bits[1], 3u);

    MockChallenger verifier_ch;
    std::vector<const FibonacciAir*> airs = { &air, &air };
    EXPECT_TRUE(verify_batch(config, airs, verifier_ch, proof));
}

TEST(BatchStark, BatchProofSmallerThanSumOfIndividual) {
    // The shared FRI proof is the dominant cost; proving N instances
    // independently should incur ~N copies of the FRI payload, whereas
    // batching produces a single FRI proof. We compare a rough payload
    // metric for the two.
    MyCfg config_batch(make_pcs());
    MyCfg config_solo0(make_pcs());
    MyCfg config_solo1(make_pcs());

    FibonacciAir air;
    auto trace_a = build_fibonacci_trace(8);
    auto trace_b = build_fibonacci_trace(8);

    Dft dft0, dft1, dft_batch;

    MockChallenger solo_ch0;
    auto solo_proof0 = p3_uni_stark::prove(config_solo0, air, solo_ch0, trace_a, dft0);
    MockChallenger solo_ch1;
    auto solo_proof1 = p3_uni_stark::prove(config_solo1, air, solo_ch1, trace_b, dft1);

    std::vector<StarkInstance<MyCfg, FibonacciAir>> instances = {
        StarkInstance<MyCfg, FibonacciAir>(air, trace_a),
        StarkInstance<MyCfg, FibonacciAir>(air, trace_b),
    };
    MockChallenger batch_ch;
    auto batch_proof = prove_batch(config_batch, instances, batch_ch, dft_batch);

    std::size_t solo_total = rough_proof_size(solo_proof0) + rough_proof_size(solo_proof1);
    std::size_t batch_total = rough_proof_size(batch_proof);

    // Batch proof should be strictly smaller than the sum of the two
    // uni-STARK proofs. The exact ratio depends on FRI parameters, but the
    // shared opening_proof (FRI commit-phase commits, final poly, and
    // query phases) is the dominant term and batch has one copy vs two.
    EXPECT_LT(batch_total, solo_total);
}

TEST(BatchStark, LookupDeclarationsBindAcrossInstances) {
    MyCfg config(make_pcs());
    FibonacciAir air;
    auto trace_a = build_fibonacci_trace(8);
    auto trace_b = build_fibonacci_trace(8);

    std::vector<StarkInstance<MyCfg, FibonacciAir>> instances = {
        StarkInstance<MyCfg, FibonacciAir>(air, trace_a),
        StarkInstance<MyCfg, FibonacciAir>(air, trace_b),
    };

    CommonData<MyCfg> common = CommonData<MyCfg>::empty(instances.size());
    common.lookups[0].push_back(Lookup<BB>{
        0, 0, 1, 0, LookupDirection::InputInTable
    });

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove_batch(config, instances, prover_ch, dft, common);

    MockChallenger verifier_ch_ok;
    std::vector<const FibonacciAir*> airs = { &air, &air };
    EXPECT_TRUE(verify_batch(config, airs, verifier_ch_ok, proof, {}, common));

    auto tampered = proof;
    tampered.opened_values.instances[1].trace_local[0] =
        tampered.opened_values.instances[1].trace_local[0] + BB4::one_val();

    MockChallenger verifier_ch_bad;
    EXPECT_FALSE(verify_batch(config, airs, verifier_ch_bad, tampered, {}, common));
}

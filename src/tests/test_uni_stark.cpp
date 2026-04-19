/**
 * @file test_uni_stark.cpp
 * @brief End-to-end tests for the uni-STARK prover/verifier (task 27).
 *
 * Covers:
 *   - Fibonacci prove + verify round-trip
 *   - Tampered trace/quotient opened values rejected
 *   - Tampered commitment rejected
 *   - Quadratic AIR (constraint_degree = 3) exercises quotient splitting
 *   - AIR with preprocessed trace round-trip
 */

#include <gtest/gtest.h>

#include "stark_config.hpp"
#include "stark_proof.hpp"
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
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

using namespace p3_uni_stark;
using namespace p3_fri;
using namespace p3_field;

using BB  = BabyBear;
using BB4 = BabyBear4;

// ============================================================================
// Mock MMCS + Challenger (same pattern as test_two_adic_fri_pcs.cpp).
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
using MyCfg  = StarkConfig<BB, BB4, MyPcs, MockChallenger>;

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

// ============================================================================
// AIRs (duck-typed: prover/verifier need width(), constraint_degree(), eval(AB&))
// ============================================================================

// Fibonacci: next(0) = cur(1); next(1) = cur(0) + cur(1); boundary cur(0)=cur(1)=1.
// Constraint polynomial degree = 2 (selector * linear).
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

// Squaring AIR: next(0) = cur(0)^2. Constraint polynomial degree = 3
// (selector degree 1 * quadratic expression degree 2), forcing K = 2N.
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

// AIR consuming a preprocessed column: main(0) must equal preprocessed(0) on every row.
// Constraint polynomial degree = 1 (no selector wrapping), so 1 quotient chunk.
struct CopyPreprocessedAir {
    std::size_t width()             const { return 1; }
    std::size_t constraint_degree() const { return 1; }

    template <typename AB>
    void eval(AB& builder) const {
        auto main = builder.main();
        auto pre  = builder.preprocessed();
        builder.assert_eq(main.current(0), pre.current(0));
    }
};

// ============================================================================
// Helpers
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

static MyPcs make_pcs() {
    Dft dft;
    PcsInputMmcs mmcs;
    return MyPcs(dft, mmcs, make_fri_params());
}

// ============================================================================
// Tests
// ============================================================================

TEST(UniStark, FibonacciProveVerifyRoundTrip) {
    MyCfg config(make_pcs());
    FibonacciAir air;
    auto trace = build_fibonacci_trace(4);

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove(config, air, prover_ch, trace, dft);

    MockChallenger verifier_ch;
    EXPECT_TRUE(verify(config, air, verifier_ch, proof));
}

TEST(UniStark, FibonacciRejectsTamperedTraceOpening) {
    MyCfg config(make_pcs());
    FibonacciAir air;
    auto trace = build_fibonacci_trace(4);

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove(config, air, prover_ch, trace, dft);

    auto tampered = proof;
    tampered.opened_values.trace_local[0] =
        tampered.opened_values.trace_local[0] + BB4::one_val();

    MockChallenger verifier_ch;
    EXPECT_FALSE(verify(config, air, verifier_ch, tampered));
}

TEST(UniStark, FibonacciRejectsTamperedQuotientChunk) {
    MyCfg config(make_pcs());
    FibonacciAir air;
    auto trace = build_fibonacci_trace(4);

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove(config, air, prover_ch, trace, dft);

    auto tampered = proof;
    tampered.opened_values.quotient_chunks[0] =
        tampered.opened_values.quotient_chunks[0] + BB4::one_val();

    MockChallenger verifier_ch;
    EXPECT_FALSE(verify(config, air, verifier_ch, tampered));
}

TEST(UniStark, FibonacciRejectsTamperedTraceCommitment) {
    MyCfg config(make_pcs());
    FibonacciAir air;
    auto trace = build_fibonacci_trace(4);

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove(config, air, prover_ch, trace, dft);

    auto tampered = proof;
    tampered.commitments.trace.hash ^= 0xDEADBEEFu;

    MockChallenger verifier_ch;
    EXPECT_FALSE(verify(config, air, verifier_ch, tampered));
}

TEST(UniStark, FibonacciRejectsShapeMismatch) {
    MyCfg config(make_pcs());
    FibonacciAir air;
    auto trace = build_fibonacci_trace(4);

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove(config, air, prover_ch, trace, dft);

    auto bad = proof;
    bad.opened_values.trace_local.pop_back();  // width now air.width() - 1

    MockChallenger verifier_ch;
    EXPECT_FALSE(verify(config, air, verifier_ch, bad));
}

TEST(UniStark, SquareAirExercisesQuotientSplitting) {
    MyCfg config(make_pcs());
    SquareAir air;
    auto trace = build_square_trace(4, BB(2));

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove(config, air, prover_ch, trace, dft);

    EXPECT_EQ(proof.log_num_quotient_chunks, 1u);

    MockChallenger verifier_ch;
    EXPECT_TRUE(verify(config, air, verifier_ch, proof));
}

TEST(UniStark, SquareAirRejectsTamperedProof) {
    MyCfg config(make_pcs());
    SquareAir air;
    auto trace = build_square_trace(4, BB(3));

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove(config, air, prover_ch, trace, dft);

    auto tampered = proof;
    tampered.opened_values.quotient_chunks[0] =
        tampered.opened_values.quotient_chunks[0] + BB4::from_base(BB(7));

    MockChallenger verifier_ch;
    EXPECT_FALSE(verify(config, air, verifier_ch, tampered));
}

TEST(UniStark, PreprocessedAirRoundTrip) {
    MyCfg config(make_pcs());
    CopyPreprocessedAir air;

    const std::size_t n = 4;
    std::vector<BB> main_data(n), pre_data(n);
    for (std::size_t i = 0; i < n; ++i) {
        main_data[i] = BB(static_cast<uint32_t>(i + 2));
        pre_data[i]  = main_data[i];
    }
    p3_matrix::RowMajorMatrix<BB> main_trace(std::move(main_data), 1);
    p3_matrix::RowMajorMatrix<BB> pre_trace (std::move(pre_data),  1);

    Dft dft;
    MockChallenger prover_ch;
    auto proof = prove(config, air, prover_ch, main_trace, dft, {}, &pre_trace);

    EXPECT_EQ(proof.preprocessed_width, 1u);
    EXPECT_TRUE(proof.commitments.has_preprocessed);

    MockChallenger verifier_ch;
    EXPECT_TRUE(verify(config, air, verifier_ch, proof));
}

TEST(UniStark, PublicValuesAreBoundIntoTranscript) {
    MyCfg config(make_pcs());
    FibonacciAir air;
    auto trace = build_fibonacci_trace(4);

    Dft dft;
    std::vector<BB> pvs = { BB(1), BB(2), BB(3) };

    MockChallenger prover_ch;
    auto proof = prove(config, air, prover_ch, trace, dft, pvs);

    // Same public values: accept.
    {
        MockChallenger verifier_ch;
        EXPECT_TRUE(verify(config, air, verifier_ch, proof, pvs));
    }
    // Different public values: reject (transcripts diverge).
    {
        MockChallenger verifier_ch;
        std::vector<BB> wrong = { BB(1), BB(2), BB(4) };
        EXPECT_FALSE(verify(config, air, verifier_ch, proof, wrong));
    }
}

#pragma once

/**
 * @file stark_prover.hpp
 * @brief Uni-STARK prover. See `plonky3/uni-stark/src/prover.rs`.
 *
 * Proves that an execution `trace` (optionally accompanied by a preprocessed
 * trace) satisfies the constraints declared by an `AIR`. The algorithm:
 *
 *   1. Commit to the trace via the PCS.
 *   2. Observe commitment + instance data, sample alpha.
 *   3. LDE the trace onto the quotient domain `gK` (disjoint coset of size
 *      `K = N * 2^log_num_chunks`).
 *   4. For each row of `gK`, evaluate the folded constraint polynomial and
 *      divide by `Z_H` to produce the quotient evaluations.
 *   5. Flatten the extension-field quotient evaluations to a width-D base
 *      matrix, convert evals-on-gK → evals-on-H_K via IDFT+DFT, and commit.
 *   6. Observe the quotient commitment, sample zeta.
 *   7. Open trace at {zeta, zeta*g}, quotient at {zeta}, and (optionally)
 *      preprocessed at {zeta, zeta*g}.
 *   8. Wrap openings + FRI proof into a `Proof<SC>`.
 */

#include "air.hpp"
#include "constraint_folder.hpp"
#include "dense_matrix.hpp"
#include "p3_util/util.hpp"
#include "stark_proof.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_uni_stark {

namespace detail {

/// Given a matrix of extension-field evaluations on `gK` (stored as a width-D
/// `Val` matrix with rows packing the D basis coefficients of each evaluation),
/// convert to evaluations on `H_K` (shift = 1). This lets us commit the
/// quotient chunks using `natural_domain_for_degree(K)` without relying on a
/// coset-shifted commit path.
template<typename Val, typename Dft>
inline p3_matrix::RowMajorMatrix<Val>
convert_quotient_to_natural_domain(p3_matrix::RowMajorMatrix<Val> qmat_on_gK,
                                   const Val& coset_shift,
                                   Dft& dft) {
    auto coefs = dft.coset_idft_batch(std::move(qmat_on_gK), coset_shift);
    return dft.dft_batch(std::move(coefs));
}

/// Batch multiplicative inverse via Montgomery's trick.
/// Preconditions: all values are non-zero.
template<typename F>
inline std::vector<F> batch_multiplicative_inverse(const std::vector<F>& values) {
    const std::size_t n = values.size();
    if (n == 0) {
        return {};
    }

    std::vector<F> invs(n);
    invs[0] = values[0];
    for (std::size_t i = 1; i < n; ++i) {
        invs[i] = invs[i - 1] * values[i];
    }

    F running_inv = invs[n - 1].inv();
    for (std::size_t i = n - 1; i > 0; --i) {
        invs[i] = invs[i - 1] * running_inv;
        running_inv = running_inv * values[i];
    }
    invs[0] = running_inv;
    return invs;
}

} // namespace detail

/// Prove that `trace` satisfies `air`. The caller-supplied `challenger` is
/// mutated as the Fiat-Shamir transcript is built.
///
/// Requirements:
///   - `air.width()` must equal `trace.width()`.
///   - `trace.height()` must be a power of two.
///   - `air.eval(folder)` must call `folder.assert_zero(...)` exactly
///     `air.num_constraints()` times; constraint ordering must be identical
///     between prover and verifier.
///   - If `preprocessed_trace` is non-null, its height must equal `trace.height()`.
template<typename SC, typename AIR, typename Dft>
Proof<SC> prove(SC& config,
                const AIR& air,
                typename SC::Challenger& challenger,
                p3_matrix::RowMajorMatrix<typename SC::Val> trace,
                Dft& dft,
                const std::vector<typename SC::Val>& public_values = {},
                const p3_matrix::RowMajorMatrix<typename SC::Val>* preprocessed_trace = nullptr) {
    using Val       = typename SC::Val;
    using Challenge = typename SC::Challenge;
    using Pcs       = typename SC::Pcs;
    using Domain    = typename Pcs::Domain;

    // ---- 1. Shape checks ---------------------------------------------------
    const std::size_t degree = trace.height();
    if (degree == 0) {
        throw std::invalid_argument("stark_prover::prove: empty trace");
    }
    if (trace.width() != air.width()) {
        throw std::invalid_argument("stark_prover::prove: trace width != air.width()");
    }
    const std::size_t log_degree = p3_util::log2_strict_usize(degree);

    const std::size_t preprocessed_width =
        preprocessed_trace != nullptr ? preprocessed_trace->width() : 0;
    if (preprocessed_trace != nullptr && preprocessed_trace->height() != degree) {
        throw std::invalid_argument("stark_prover::prove: preprocessed height mismatch");
    }

    // Number of quotient chunks derived from the AIR's constraint degree.
    const std::size_t constraint_degree = air.constraint_degree();
    const std::size_t log_num_quotient_chunks =
        p3_util::log2_ceil_usize(constraint_degree <= 1 ? 1 : constraint_degree - 1);
    const std::size_t num_quotient_chunks = std::size_t(1) << log_num_quotient_chunks;
    const std::size_t quotient_size = degree * num_quotient_chunks;
    const std::size_t log_quotient_size = log_degree + log_num_quotient_chunks;

    Pcs& pcs = config.pcs();

    // ---- 2. Commit to trace (and optional preprocessed) --------------------
    Domain trace_domain = pcs.natural_domain_for_degree(degree);

    // We need a copy of the trace for the auxiliary LDE onto gK.
    p3_matrix::RowMajorMatrix<Val> trace_copy = trace;

    auto [trace_commit, trace_pd] = pcs.commit({{trace_domain, std::move(trace)}});

    typename Pcs::InputCommitment preprocessed_commit{};
    typename Pcs::PcsProverData   preprocessed_pd_storage;
    const typename Pcs::PcsProverData* preprocessed_pd_ptr = nullptr;
    const bool has_preprocessed = preprocessed_width > 0;
    if (has_preprocessed) {
        // Single duplicate for PCS; quotient LDE copies from `*preprocessed_trace`
        // only when that pass runs (see below), so we never keep two prover buffers.
        p3_matrix::RowMajorMatrix<Val> tmp = *preprocessed_trace;
        auto [pc, ppd] = pcs.commit({{trace_domain, std::move(tmp)}});
        preprocessed_commit    = pc;
        preprocessed_pd_storage = std::move(ppd);
        preprocessed_pd_ptr     = &preprocessed_pd_storage;
    }

    // ---- 3. Observe instance + sample alpha --------------------------------
    challenger.observe_val(Val(static_cast<uint32_t>(log_degree)));
    challenger.observe_val(Val(static_cast<uint32_t>(preprocessed_width)));
    challenger.observe_commitment(trace_commit);
    if (has_preprocessed) {
        challenger.observe_commitment(preprocessed_commit);
    }
    for (const Val& v : public_values) {
        challenger.observe_val(v);
    }

    Challenge alpha = challenger.template sample_challenge<Challenge>();

    // ---- 4. LDE trace (and preprocessed) onto gK --------------------------
    const Val g = Val::generator();
    auto trace_on_gK = dft.coset_lde_batch(std::move(trace_copy), log_num_quotient_chunks, g);
    if (trace_on_gK.height() != quotient_size) {
        throw std::runtime_error("stark_prover::prove: unexpected LDE height");
    }

    p3_matrix::RowMajorMatrix<Val> preprocessed_on_gK;
    if (has_preprocessed) {
        // One matrix copy into a prvalue (no second long-lived `preprocessed_copy`).
        preprocessed_on_gK = dft.coset_lde_batch(
            p3_matrix::RowMajorMatrix<Val>(*preprocessed_trace),
            log_num_quotient_chunks, g);
    }

    // ---- 5. Compute quotient evaluations on gK ----------------------------
    // Precompute selectors.
    const Val omega_N    = Val::two_adic_generator(log_degree);
    const Val omega_N_inv = omega_N.inv();
    const Val omega_K    = Val::two_adic_generator(log_quotient_size);
    const Val g_pow_N    = g.exp_u64(static_cast<uint64_t>(degree));
    const Val omega_K_pow_N = omega_K.exp_u64(static_cast<uint64_t>(degree)); // = omega_{num_chunks}
    const Val one        = Val::one_val();

    std::vector<Challenge> inv_vanishing_ch(quotient_size);
    std::vector<Challenge> is_first_row_ch(quotient_size);
    std::vector<Challenge> is_last_row_ch(quotient_size);
    std::vector<Challenge> is_transition_ch(quotient_size);
    {
        std::vector<Val> inversion_inputs;
        inversion_inputs.resize(3 * quotient_size);

        Val xk  = g;            // x_k  = g * omega_K^k, start k=0
        Val zhc = g_pow_N;      // g_pow_N * omega_{num_chunks}^k, start k=0
        for (std::size_t k = 0; k < quotient_size; ++k) {
            const Val z_h = zhc - one;
            inversion_inputs[3 * k]     = z_h;
            inversion_inputs[3 * k + 1] = xk - one;
            inversion_inputs[3 * k + 2] = xk - omega_N_inv;
            is_transition_ch[k]     = Challenge::from_base(xk - omega_N_inv);

            xk  = xk  * omega_K;
            zhc = zhc * omega_K_pow_N;
        }

        // One inversion for all selector denominators over the quotient domain.
        // Safe: x_k ∈ gK \ H, so Z_H(x_k), x_k - 1, and x_k - omega_N^{-1} are non-zero.
        const std::vector<Val> inversion_outputs =
            detail::batch_multiplicative_inverse(inversion_inputs);

        for (std::size_t k = 0; k < quotient_size; ++k) {
            const Val z_h = inversion_inputs[3 * k];
            inv_vanishing_ch[k] = Challenge::from_base(inversion_outputs[3 * k]);
            is_first_row_ch[k] = Challenge::from_base(z_h * inversion_outputs[3 * k + 1]);
            is_last_row_ch[k] = Challenge::from_base(z_h * inversion_outputs[3 * k + 2]);
        }
    }

    using Folder = ConstraintFolder<Val, Challenge>;
    using MainWindow = typename Folder::MainWindow;

    const std::size_t trace_width = air.width();

    // Embed the LDE once into the extension field so each cell is lifted a
    // single time (the quotient loop pairs row k with k_next and would
    // otherwise call from_base twice per cell over the full domain).
    std::vector<Challenge> trace_on_gK_ch(quotient_size * trace_width);
    for (std::size_t k = 0; k < quotient_size; ++k) {
        const std::size_t row_off = k * trace_width;
        for (std::size_t c = 0; c < trace_width; ++c) {
            trace_on_gK_ch[row_off + c] =
                Challenge::from_base(trace_on_gK.get_unchecked(k, c));
        }
    }
    trace_on_gK = p3_matrix::RowMajorMatrix<Val>();

    std::vector<Challenge> preprocessed_on_gK_ch;
    if (has_preprocessed) {
        preprocessed_on_gK_ch.resize(quotient_size * preprocessed_width);
        for (std::size_t k = 0; k < quotient_size; ++k) {
            const std::size_t row_off = k * preprocessed_width;
            for (std::size_t c = 0; c < preprocessed_width; ++c) {
                preprocessed_on_gK_ch[row_off + c] =
                    Challenge::from_base(preprocessed_on_gK.get_unchecked(k, c));
            }
        }
        preprocessed_on_gK = p3_matrix::RowMajorMatrix<Val>();
    }

    std::vector<Challenge> quotient_values(quotient_size);
    auto eval_quotient_at = [&](std::size_t k) {
        const std::size_t k_next = (k + num_quotient_chunks) % quotient_size;

        const Challenge* cur_ptr = trace_on_gK_ch.data() + k * trace_width;
        const Challenge* nxt_ptr = trace_on_gK_ch.data() + k_next * trace_width;
        MainWindow main_win(
            p3_air::ConstRowView<Challenge>(cur_ptr, trace_width),
            p3_air::ConstRowView<Challenge>(nxt_ptr, trace_width));

        MainWindow pre_win;
        if (has_preprocessed) {
            const Challenge* pre_cur = preprocessed_on_gK_ch.data() + k * preprocessed_width;
            const Challenge* pre_nxt =
                preprocessed_on_gK_ch.data() + k_next * preprocessed_width;
            pre_win = MainWindow(
                p3_air::ConstRowView<Challenge>(pre_cur, preprocessed_width),
                p3_air::ConstRowView<Challenge>(pre_nxt, preprocessed_width));
        }

        // Each row evaluation is independent; keep folder state thread-local.
        Folder folder;
        folder.set_alpha(alpha);
        folder.set_windows(main_win, pre_win);
        folder.set_selectors(
            is_first_row_ch[k],
            is_last_row_ch[k],
            is_transition_ch[k]);
        folder.reset_accumulator();
        air.eval(folder);

        return folder.accumulator() * inv_vanishing_ch[k];
    };

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (std::int64_t k_i = 0; k_i < static_cast<std::int64_t>(quotient_size); ++k_i) {
        const std::size_t k = static_cast<std::size_t>(k_i);
        quotient_values[k] = eval_quotient_at(k);
    }
#else
    for (std::size_t k = 0; k < quotient_size; ++k) {
        quotient_values[k] = eval_quotient_at(k);
    }
#endif

    // ---- 6. Flatten Q evals to width-D Val matrix on gK, convert to H_K ---
    constexpr std::size_t D = Challenge::DEGREE;
    std::vector<Val> qflat(quotient_size * D);
    for (std::size_t k = 0; k < quotient_size; ++k) {
        for (std::size_t d = 0; d < D; ++d) {
            qflat[k * D + d] = quotient_values[k][d];
        }
    }
    p3_matrix::RowMajorMatrix<Val> qmat_on_gK(std::move(qflat), D);
    auto qmat_on_HK = detail::convert_quotient_to_natural_domain<Val, Dft>(
        std::move(qmat_on_gK), g, dft);

    Domain quotient_commit_domain{ log_quotient_size, Val::one_val() };
    auto [quotient_commit, quotient_pd] =
        pcs.commit({{quotient_commit_domain, std::move(qmat_on_HK)}});

    challenger.observe_commitment(quotient_commit);

    // ---- 7. Sample zeta, open all commitments -----------------------------
    Challenge zeta = challenger.template sample_challenge<Challenge>();
    Challenge zeta_next = zeta * Challenge::from_base(omega_N);

    std::vector<std::pair<const typename Pcs::PcsProverData*,
                          std::vector<std::vector<Challenge>>>> open_rounds;
    open_rounds.emplace_back(&trace_pd,
                             std::vector<std::vector<Challenge>>{{zeta, zeta_next}});
    open_rounds.emplace_back(&quotient_pd,
                             std::vector<std::vector<Challenge>>{{zeta}});
    if (has_preprocessed) {
        open_rounds.emplace_back(preprocessed_pd_ptr,
                                 std::vector<std::vector<Challenge>>{{zeta, zeta_next}});
    }

    auto [opened_values, opening_proof] = pcs.open(std::move(open_rounds), challenger);

    // ---- 8. Assemble proof -------------------------------------------------
    Proof<SC> proof;
    proof.degree_bits             = log_degree;
    proof.log_num_quotient_chunks = log_num_quotient_chunks;
    proof.preprocessed_width      = preprocessed_width;
    proof.commitments.trace           = std::move(trace_commit);
    proof.commitments.quotient_chunks = std::move(quotient_commit);
    if (has_preprocessed) {
        proof.commitments.preprocessed     = std::move(preprocessed_commit);
        proof.commitments.has_preprocessed = true;
    }
    // opened_values[batch][matrix][point][col]
    // batch 0 = trace: 1 matrix, 2 points
    proof.opened_values.trace_local = opened_values[0][0][0];
    proof.opened_values.trace_next  = opened_values[0][0][1];
    // batch 1 = quotient: 1 matrix, 1 point, D columns
    proof.opened_values.quotient_chunks = opened_values[1][0][0];
    if (has_preprocessed) {
        // batch 2 = preprocessed: 1 matrix, 2 points
        proof.opened_values.preprocessed_local = opened_values[2][0][0];
        proof.opened_values.preprocessed_next  = opened_values[2][0][1];
    }
    proof.opening_proof = std::move(opening_proof);
    return proof;
}

} // namespace p3_uni_stark

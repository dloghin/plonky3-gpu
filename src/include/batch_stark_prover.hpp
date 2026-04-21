#pragma once

/**
 * @file batch_stark_prover.hpp
 * @brief Batch-STARK prover. See `plonky3/batch-stark/src/prover.rs`.
 *
 * Proves that every `StarkInstance` in `instances` satisfies its AIR, in
 * a single proof that shares one FRI opening proof across all instances.
 *
 * Algorithm (mirrors `plonky3/batch-stark/src/prover.rs`):
 *   1. Commit all main traces in a single PCS commit (one matrix per
 *      instance; the underlying PCS supports mixed trace heights via
 *      its per-height LDE groups).
 *   2. Observe the batched main commitment, per-instance shape data, and
 *      public values.
 *   3. Sample a single shared folding challenge `alpha`.
 *   4. For each instance i:
 *        - LDE the trace onto the quotient domain `g_i * H_{K_i}`.
 *        - Evaluate the AIR constraints and divide by `Z_H_i` to obtain
 *          `Q_i` values on the quotient domain.
 *        - Convert Q_i values to a width-D Val matrix on `H_{K_i}`.
 *   5. Commit all quotient matrices in a single PCS commit.
 *   6. Observe the batched quotient commitment.
 *   7. Sample a shared out-of-domain point `zeta`.
 *   8. Open the main commitment at `{zeta, zeta * g_i}` for each instance
 *      and the quotient commitment at `{zeta}` for each instance, all
 *      under one `pcs.open(...)` call that produces a single FRI proof.
 *   9. Wrap everything into a `BatchProof<SC>`.
 */

#include "air.hpp"
#include "batch_stark_proof.hpp"
#include "constraint_folder.hpp"
#include "dense_matrix.hpp"
#include "p3_util/util.hpp"
#include "stark_prover.hpp"  // reuses detail::batch_multiplicative_inverse + convert_quotient_to_natural_domain

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_batch_stark {

/// Prove a batch of AIR instances.
///
/// Each instance must satisfy the same invariants as `p3_uni_stark::prove`:
///   - `inst.trace->height()` is a power of two.
///   - `inst.trace->width() == inst.air->width()`.
///   - `inst.air->eval(folder)` calls `folder.assert_zero(...)` the same
///     number of times as `inst.air->num_constraints()` (when the AIR
///     reports a constraint count).
///
/// `common` is reserved for future lookup/preprocessed support; an empty
/// `CommonData::empty(instances.size())` is the expected value for AIRs
/// without preprocessed columns or lookups.
template <typename SC, typename AIR, typename Dft>
BatchProof<SC> prove_batch(SC& config,
                           const std::vector<StarkInstance<SC, AIR>>& instances,
                           typename SC::Challenger& challenger,
                           Dft& dft,
                           const CommonData<SC>& common = CommonData<SC>{}) {
    using Val       = typename SC::Val;
    using Challenge = typename SC::Challenge;
    using Pcs       = typename SC::Pcs;
    using Domain    = typename Pcs::Domain;

    const std::size_t n_instances = instances.size();
    if (n_instances == 0) {
        throw std::invalid_argument("prove_batch: no instances");
    }
    (void)common;  // placeholder while lookup/preprocessed support is a TODO.

    Pcs& pcs = config.pcs();

    // ---- 1. Per-instance shape data ----------------------------------------
    std::vector<std::size_t> degrees(n_instances);
    std::vector<std::size_t> log_degrees(n_instances);
    std::vector<std::size_t> widths(n_instances);
    std::vector<std::size_t> log_num_chunks(n_instances);
    std::vector<std::size_t> num_chunks(n_instances);
    std::vector<std::size_t> quotient_sizes(n_instances);
    std::vector<std::size_t> log_quotient_sizes(n_instances);
    for (std::size_t i = 0; i < n_instances; ++i) {
        const auto& inst = instances[i];
        if (inst.air == nullptr || inst.trace == nullptr) {
            throw std::invalid_argument("prove_batch: instance has null air or trace");
        }
        const std::size_t deg = inst.trace->height();
        if (deg == 0) {
            throw std::invalid_argument("prove_batch: empty trace");
        }
        if (inst.trace->width() != inst.air->width()) {
            throw std::invalid_argument("prove_batch: trace width != air.width()");
        }
        degrees[i]         = deg;
        log_degrees[i]     = p3_util::log2_strict_usize(deg);
        widths[i]          = inst.air->width();
        const std::size_t cdeg = inst.air->constraint_degree();
        log_num_chunks[i]  = p3_util::log2_ceil_usize(cdeg <= 1 ? 1 : cdeg - 1);
        num_chunks[i]      = std::size_t(1) << log_num_chunks[i];
        quotient_sizes[i]  = degrees[i] * num_chunks[i];
        log_quotient_sizes[i] = log_degrees[i] + log_num_chunks[i];
    }

    // ---- 2. Commit all traces in a single PCS commit -----------------------
    std::vector<std::pair<Domain, p3_matrix::RowMajorMatrix<Val>>> trace_commit_inputs;
    trace_commit_inputs.reserve(n_instances);

    // Keep copies of each trace for the LDE step below (PCS commit consumes the input).
    std::vector<p3_matrix::RowMajorMatrix<Val>> trace_copies;
    trace_copies.reserve(n_instances);
    for (std::size_t i = 0; i < n_instances; ++i) {
        Domain dom = pcs.natural_domain_for_degree(degrees[i]);
        trace_copies.push_back(*instances[i].trace);
        trace_commit_inputs.emplace_back(dom, *instances[i].trace);
    }
    auto [main_commit, main_pd] = pcs.commit(std::move(trace_commit_inputs));

    // ---- 3. Observe instance-binding data + public values + main commitment
    challenger.observe_val(Val(static_cast<uint32_t>(n_instances)));
    for (std::size_t i = 0; i < n_instances; ++i) {
        challenger.observe_val(Val(static_cast<uint32_t>(log_degrees[i])));
        challenger.observe_val(Val(static_cast<uint32_t>(widths[i])));
        challenger.observe_val(Val(static_cast<uint32_t>(num_chunks[i])));
    }
    challenger.observe_commitment(main_commit);
    for (std::size_t i = 0; i < n_instances; ++i) {
        for (const Val& v : instances[i].public_values) {
            challenger.observe_val(v);
        }
    }

    // ---- 4. Shared alpha ---------------------------------------------------
    Challenge alpha = challenger.template sample_challenge<Challenge>();

    // ---- 5. Compute quotient evaluations for each instance -----------------
    // Each quotient is committed as a width-D base matrix on H_{K_i} (shift=1).
    std::vector<std::pair<Domain, p3_matrix::RowMajorMatrix<Val>>> quotient_commit_inputs;
    quotient_commit_inputs.reserve(n_instances);

    constexpr std::size_t D = Challenge::DEGREE;
    const Val g = Val::generator();
    const Val one = Val::one_val();

    using Folder = p3_uni_stark::ConstraintFolder<Val, Challenge>;
    using MainWindow = typename Folder::MainWindow;

    for (std::size_t i = 0; i < n_instances; ++i) {
        const auto& inst = instances[i];
        const std::size_t deg = degrees[i];
        const std::size_t log_deg = log_degrees[i];
        const std::size_t log_chunks_i = log_num_chunks[i];
        const std::size_t qsize = quotient_sizes[i];
        const std::size_t log_qsize = log_quotient_sizes[i];

        auto trace_on_gK = dft.coset_lde_batch(std::move(trace_copies[i]), log_chunks_i, g);
        if (trace_on_gK.height() != qsize) {
            throw std::runtime_error("prove_batch: unexpected LDE height");
        }

        // Selectors at each quotient-domain point.
        const Val omega_N     = Val::two_adic_generator(log_deg);
        const Val omega_N_inv = omega_N.inv();
        const Val omega_K     = Val::two_adic_generator(log_qsize);
        const Val g_pow_N     = g.exp_u64(static_cast<uint64_t>(deg));
        const Val omega_K_pow_N = omega_K.exp_u64(static_cast<uint64_t>(deg));
        const Val inv_degree  = Val(static_cast<uint32_t>(deg)).inv();
        const Val last_row_scale = omega_N_inv * inv_degree;

        std::vector<Challenge> inv_vanishing_ch(qsize);
        std::vector<Challenge> is_first_row_ch(qsize);
        std::vector<Challenge> is_last_row_ch(qsize);
        std::vector<Challenge> is_transition_ch(qsize);
        {
            std::vector<Val> inversion_inputs(3 * qsize);
            Val xk  = g;
            Val zhc = g_pow_N;
            for (std::size_t k = 0; k < qsize; ++k) {
                const Val z_h = zhc - one;
                inversion_inputs[3 * k]     = z_h;
                inversion_inputs[3 * k + 1] = xk - one;
                inversion_inputs[3 * k + 2] = xk - omega_N_inv;
                xk  = xk  * omega_K;
                zhc = zhc * omega_K_pow_N;
            }
            const auto inversion_outputs =
                p3_uni_stark::detail::batch_multiplicative_inverse(inversion_inputs);
            for (std::size_t k = 0; k < qsize; ++k) {
                const Val z_h = inversion_inputs[3 * k];
                inv_vanishing_ch[k] = Challenge::from_base(inversion_outputs[3 * k]);
                const Val is_first = z_h * inversion_outputs[3 * k + 1] * inv_degree;
                const Val is_last  = z_h * inversion_outputs[3 * k + 2] * last_row_scale;
                is_first_row_ch[k]  = Challenge::from_base(is_first);
                is_last_row_ch[k]   = Challenge::from_base(is_last);
                is_transition_ch[k] = Challenge::one_val() - is_last_row_ch[k];
            }
        }

        const std::size_t trace_width = widths[i];
        std::vector<Challenge> quotient_values(qsize);

        auto eval_quotient_at = [&](std::size_t k) {
            const std::size_t k_next = (k + num_chunks[i]) % qsize;
            thread_local std::vector<Challenge> trace_cur;
            thread_local std::vector<Challenge> trace_nxt;
            trace_cur.resize(trace_width);
            trace_nxt.resize(trace_width);
            for (std::size_t c = 0; c < trace_width; ++c) {
                trace_cur[c] = Challenge::from_base(trace_on_gK.get_unchecked(k, c));
                trace_nxt[c] = Challenge::from_base(trace_on_gK.get_unchecked(k_next, c));
            }
            MainWindow main_win(
                p3_air::ConstRowView<Challenge>(trace_cur.data(), trace_width),
                p3_air::ConstRowView<Challenge>(trace_nxt.data(), trace_width));
            MainWindow pre_win;  // no preprocessed support in this port.

            Folder folder;
            folder.set_alpha(alpha);
            folder.set_windows(main_win, pre_win);
            folder.set_selectors(is_first_row_ch[k], is_last_row_ch[k], is_transition_ch[k]);
            folder.reset_accumulator();
            inst.air->eval(folder);
            return folder.accumulator() * inv_vanishing_ch[k];
        };

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
        for (std::int64_t k_i = 0; k_i < static_cast<std::int64_t>(qsize); ++k_i) {
            quotient_values[static_cast<std::size_t>(k_i)] =
                eval_quotient_at(static_cast<std::size_t>(k_i));
        }
#else
        for (std::size_t k = 0; k < qsize; ++k) {
            quotient_values[k] = eval_quotient_at(k);
        }
#endif

        // Release LDE memory before allocating the flattened quotient matrix.
        trace_on_gK = p3_matrix::RowMajorMatrix<Val>();

        // Flatten to width-D Val matrix on gK, convert to H_K so the PCS commit
        // can use `natural_domain_for_degree(K_i)` (shift = 1).
        std::vector<Val> qflat(qsize * D);
        for (std::size_t k = 0; k < qsize; ++k) {
            for (std::size_t d = 0; d < D; ++d) {
                qflat[k * D + d] = quotient_values[k][d];
            }
        }
        p3_matrix::RowMajorMatrix<Val> qmat_on_gK(std::move(qflat), D);
        auto qmat_on_HK = p3_uni_stark::detail::convert_quotient_to_natural_domain<Val, Dft>(
            std::move(qmat_on_gK), g, dft);

        Domain quotient_commit_domain{ log_qsize, Val::one_val() };
        quotient_commit_inputs.emplace_back(quotient_commit_domain, std::move(qmat_on_HK));
    }

    // ---- 6. Commit all quotient matrices in a single PCS commit ------------
    auto [quotient_commit, quotient_pd] = pcs.commit(std::move(quotient_commit_inputs));
    challenger.observe_commitment(quotient_commit);

    // ---- 7. Shared zeta + per-instance zeta_next ---------------------------
    Challenge zeta = challenger.template sample_challenge<Challenge>();
    std::vector<Challenge> zeta_nexts(n_instances);
    for (std::size_t i = 0; i < n_instances; ++i) {
        const Val omega_N_i = Val::two_adic_generator(log_degrees[i]);
        zeta_nexts[i] = zeta * Challenge::from_base(omega_N_i);
    }

    // ---- 8. Open all commitments under a single FRI proof ------------------
    // Trace round: one matrix per instance, 2 points {zeta, zeta_next_i}.
    std::vector<std::vector<Challenge>> trace_points;
    trace_points.reserve(n_instances);
    for (std::size_t i = 0; i < n_instances; ++i) {
        trace_points.push_back({ zeta, zeta_nexts[i] });
    }
    // Quotient round: one matrix per instance, 1 point {zeta}.
    std::vector<std::vector<Challenge>> quotient_points;
    quotient_points.reserve(n_instances);
    for (std::size_t i = 0; i < n_instances; ++i) {
        quotient_points.push_back({ zeta });
    }

    std::vector<std::pair<const typename Pcs::PcsProverData*,
                          std::vector<std::vector<Challenge>>>> open_rounds;
    open_rounds.emplace_back(&main_pd, std::move(trace_points));
    open_rounds.emplace_back(&quotient_pd, std::move(quotient_points));

    auto [opened_values, opening_proof] = pcs.open(std::move(open_rounds), challenger);

    // ---- 9. Assemble proof -------------------------------------------------
    BatchProof<SC> proof;
    proof.commitments.main = std::move(main_commit);
    proof.commitments.quotient_chunks = std::move(quotient_commit);
    proof.opening_proof = std::move(opening_proof);
    proof.degree_bits = log_degrees;
    proof.log_num_quotient_chunks = log_num_chunks;

    // opened_values[round=0][matrix=i][point=0/1][col]   -> trace local/next
    // opened_values[round=1][matrix=i][point=0][col]     -> quotient D coefs
    proof.opened_values.instances.resize(n_instances);
    for (std::size_t i = 0; i < n_instances; ++i) {
        auto& out = proof.opened_values.instances[i];
        out.trace_local = opened_values[0][i][0];
        out.trace_next  = opened_values[0][i][1];
        out.quotient_chunks = opened_values[1][i][0];
    }
    return proof;
}

} // namespace p3_batch_stark

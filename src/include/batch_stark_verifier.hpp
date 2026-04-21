#pragma once

/**
 * @file batch_stark_verifier.hpp
 * @brief Batch-STARK verifier. Mirrors `plonky3/batch-stark/src/verifier.rs`.
 *
 * Algorithm:
 *   1. Shape-check the proof against the AIR list.
 *   2. Replay observations on a fresh challenger in the same order used by
 *      the prover: instance count, per-instance (log_degree, width, num
 *      quotient chunks), main commitment, public values, then quotient
 *      commitment. Sample shared `alpha` before observing the quotient
 *      commitment and shared `zeta` after.
 *   3. Call `pcs.verify(...)` once with all trace/quotient `VerifyCommitment`
 *      entries. This covers every matrix in the single FRI opening proof.
 *   4. For each instance i, recompose `Q_i(zeta)` from its D quotient
 *      extension coefficients and check
 *      `folded_constraints_i * (1 / Z_{H_i}(zeta)) == Q_i(zeta)`.
 *
 * Returns `true` iff all steps succeed.
 */

#include "air.hpp"
#include "batch_stark_proof.hpp"
#include "constraint_folder.hpp"
#include "p3_util/util.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace p3_batch_stark {

template <typename SC, typename AIR>
bool verify_batch(SC& config,
                  const std::vector<const AIR*>& airs,
                  typename SC::Challenger& challenger,
                  const BatchProof<SC>& proof,
                  const std::vector<std::vector<typename SC::Val>>& public_values = {},
                  const CommonData<SC>& common = CommonData<SC>{}) {
    using Val       = typename SC::Val;
    using Challenge = typename SC::Challenge;
    using Pcs       = typename SC::Pcs;
    using Domain    = typename Pcs::Domain;

    (void)common;

    const std::size_t n_instances = airs.size();
    if (n_instances == 0) return false;
    if (proof.degree_bits.size() != n_instances) return false;
    if (proof.log_num_quotient_chunks.size() != n_instances) return false;
    if (proof.opened_values.instances.size() != n_instances) return false;
    if (!public_values.empty() && public_values.size() != n_instances) return false;

    Pcs& pcs = config.pcs();

    // ---- 1. Shape checks ---------------------------------------------------
    constexpr std::size_t D = Challenge::DEGREE;
    for (std::size_t i = 0; i < n_instances; ++i) {
        if (airs[i] == nullptr) return false;

        const std::size_t cdeg = airs[i]->constraint_degree();
        const std::size_t expected_log_chunks =
            p3_util::log2_ceil_usize(cdeg <= 1 ? 1 : cdeg - 1);
        if (proof.log_num_quotient_chunks[i] != expected_log_chunks) return false;

        const auto& ov = proof.opened_values.instances[i];
        if (ov.trace_local.size() != airs[i]->width()) return false;
        if (ov.trace_next.size()  != airs[i]->width()) return false;
        if (ov.quotient_chunks.size() != D) return false;
    }

    // ---- 2. Replay challenger observations --------------------------------
    challenger.observe_val(Val(static_cast<uint32_t>(n_instances)));
    for (std::size_t i = 0; i < n_instances; ++i) {
        const std::size_t num_chunks_i = std::size_t(1) << proof.log_num_quotient_chunks[i];
        challenger.observe_val(Val(static_cast<uint32_t>(proof.degree_bits[i])));
        challenger.observe_val(Val(static_cast<uint32_t>(airs[i]->width())));
        challenger.observe_val(Val(static_cast<uint32_t>(num_chunks_i)));
    }
    challenger.observe_commitment(proof.commitments.main);
    for (std::size_t i = 0; i < n_instances; ++i) {
        if (!public_values.empty()) {
            for (const Val& v : public_values[i]) {
                challenger.observe_val(v);
            }
        }
    }

    Challenge alpha = challenger.template sample_challenge<Challenge>();
    challenger.observe_commitment(proof.commitments.quotient_chunks);
    Challenge zeta = challenger.template sample_challenge<Challenge>();

    // ---- 3. Build VerifyCommitments and run a single pcs.verify -----------
    using VC = typename Pcs::VerifyCommitment;
    std::vector<VC> to_verify;
    to_verify.reserve(2);

    // Trace: 1 commitment, n_instances matrices, 2 points each.
    std::vector<Challenge> zeta_nexts(n_instances);
    for (std::size_t i = 0; i < n_instances; ++i) {
        const Val omega_N_i = Val::two_adic_generator(proof.degree_bits[i]);
        zeta_nexts[i] = zeta * Challenge::from_base(omega_N_i);
    }

    {
        VC vc;
        vc.commitment = proof.commitments.main;
        vc.domains.reserve(n_instances);
        vc.points.reserve(n_instances);
        vc.opened_values.reserve(n_instances);
        for (std::size_t i = 0; i < n_instances; ++i) {
            const std::size_t degree_i = std::size_t(1) << proof.degree_bits[i];
            vc.domains.push_back(pcs.natural_domain_for_degree(degree_i));
            vc.points.push_back({ zeta, zeta_nexts[i] });
            vc.opened_values.push_back({
                proof.opened_values.instances[i].trace_local,
                proof.opened_values.instances[i].trace_next
            });
        }
        to_verify.push_back(std::move(vc));
    }
    {
        VC vc;
        vc.commitment = proof.commitments.quotient_chunks;
        vc.domains.reserve(n_instances);
        vc.points.reserve(n_instances);
        vc.opened_values.reserve(n_instances);
        for (std::size_t i = 0; i < n_instances; ++i) {
            const std::size_t log_qsize = proof.degree_bits[i] + proof.log_num_quotient_chunks[i];
            Domain qdom{ log_qsize, Val::one_val() };
            vc.domains.push_back(qdom);
            vc.points.push_back({ zeta });
            vc.opened_values.push_back({ proof.opened_values.instances[i].quotient_chunks });
        }
        to_verify.push_back(std::move(vc));
    }

    if (!pcs.verify(to_verify, proof.opening_proof, challenger)) {
        return false;
    }

    // ---- 4. Per-instance constraint check at zeta -------------------------
    using Folder = p3_uni_stark::ConstraintFolder<Val, Challenge>;
    using MainWindow = typename Folder::MainWindow;

    for (std::size_t i = 0; i < n_instances; ++i) {
        const auto& ov = proof.opened_values.instances[i];
        const std::size_t log_deg = proof.degree_bits[i];
        const std::size_t degree  = std::size_t(1) << log_deg;

        // Q(zeta) = sum_d basis_d * Q_d(zeta)
        Challenge Q_zeta = Challenge::zero_val();
        for (std::size_t d = 0; d < D; ++d) {
            Challenge basis_d = Challenge::zero_val();
            basis_d[d] = Val::one_val();
            Q_zeta = Q_zeta + basis_d * ov.quotient_chunks[d];
        }

        // Selectors at zeta (trace_domain shift = 1, so x = zeta).
        const Val omega_N   = Val::two_adic_generator(log_deg);
        Challenge zeta_pow_N = zeta.exp_u64(static_cast<uint64_t>(degree));
        Challenge one_ext    = Challenge::one_val();
        Challenge z_h        = zeta_pow_N - one_ext;
        Challenge omega_N_inv_c = Challenge::from_base(omega_N.inv());
        const Challenge inv_degree_c =
            Challenge::from_base(Val(static_cast<uint32_t>(degree)).inv());

        const Challenge zeta_minus_one         = zeta - one_ext;
        const Challenge zeta_minus_omega_N_inv = zeta - omega_N_inv_c;
        if (z_h == Challenge::zero_val()
            || zeta_minus_one == Challenge::zero_val()
            || zeta_minus_omega_N_inv == Challenge::zero_val()) {
            return false;
        }

        Challenge is_first_row  = z_h * zeta_minus_one.inv() * inv_degree_c;
        Challenge is_last_row   = z_h * zeta_minus_omega_N_inv.inv() * omega_N_inv_c * inv_degree_c;
        Challenge is_transition = one_ext - is_last_row;
        Challenge inv_vanishing = z_h.inv();

        MainWindow main_win(
            p3_air::ConstRowView<Challenge>(ov.trace_local.data(), ov.trace_local.size()),
            p3_air::ConstRowView<Challenge>(ov.trace_next.data(),  ov.trace_next.size()));
        MainWindow pre_win;

        Folder folder;
        folder.set_alpha(alpha);
        folder.set_windows(main_win, pre_win);
        folder.set_selectors(is_first_row, is_last_row, is_transition);
        folder.reset_accumulator();
        airs[i]->eval(folder);
        Challenge folded = folder.accumulator();
        if (folded * inv_vanishing != Q_zeta) {
            return false;
        }
    }

    return true;
}

} // namespace p3_batch_stark

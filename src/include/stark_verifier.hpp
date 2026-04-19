#pragma once

/**
 * @file stark_verifier.hpp
 * @brief Uni-STARK verifier. Mirrors plonky3/uni-stark/src/verifier.rs.
 *
 * Algorithm:
 *   1. Shape-check the proof against the AIR.
 *   2. Replay observations on a fresh challenger; sample alpha, then zeta.
 *   3. Call `pcs.verify(...)` with the trace/quotient/(preprocessed)
 *      commitments and their opened values.
 *   4. Reconstruct `Q(zeta) = sum_{d} basis_d * Q_d(zeta)` from the opened
 *      quotient chunks.
 *   5. Run the constraint folder on the opened trace rows at `zeta` and check
 *      `folded_constraints * (1 / Z_H(zeta)) == Q(zeta)`.
 *
 * Returns `true` if every step passes; `false` otherwise. Any structural
 * mismatch (bad widths, missing preprocessed opening, etc.) also produces
 * `false` rather than throwing, so a malformed proof cannot crash the
 * verifier.
 */

#include "air.hpp"
#include "constraint_folder.hpp"
#include "dense_matrix.hpp"
#include "stark_config.hpp"
#include "stark_proof.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace p3_uni_stark {

template<typename SC, typename AIR>
bool verify(SC& config,
            const AIR& air,
            typename SC::Challenger& challenger,
            const Proof<SC>& proof,
            const std::vector<typename SC::Val>& public_values = {}) {
    using Val       = typename SC::Val;
    using Challenge = typename SC::Challenge;
    using Pcs       = typename SC::Pcs;
    using Domain    = typename Pcs::Domain;

    Pcs& pcs = config.pcs();

    const std::size_t log_degree         = proof.degree_bits;
    const std::size_t degree             = std::size_t(1) << log_degree;
    const std::size_t log_num_chunks     = proof.log_num_quotient_chunks;
    const std::size_t log_quotient_size  = log_degree + log_num_chunks;
    const std::size_t preprocessed_width = proof.preprocessed_width;
    const bool has_preprocessed          = proof.commitments.has_preprocessed;

    // ---- 1. Shape checks --------------------------------------------------
    const std::size_t air_width = air.width();
    if (proof.opened_values.trace_local.size() != air_width) return false;
    if (proof.opened_values.trace_next.size()  != air_width) return false;
    constexpr std::size_t D = Challenge::DEGREE;
    if (proof.opened_values.quotient_chunks.size() != D) return false;
    if (has_preprocessed != (preprocessed_width > 0)) return false;
    if (has_preprocessed) {
        if (proof.opened_values.preprocessed_local.size() != preprocessed_width) return false;
        if (proof.opened_values.preprocessed_next.size()  != preprocessed_width) return false;
    } else {
        if (!proof.opened_values.preprocessed_local.empty()) return false;
        if (!proof.opened_values.preprocessed_next.empty())  return false;
    }

    // ---- 2. Replay challenger observations -------------------------------
    challenger.observe_val(Val(static_cast<uint32_t>(log_degree)));
    challenger.observe_val(Val(static_cast<uint32_t>(preprocessed_width)));
    challenger.observe_commitment(proof.commitments.trace);
    if (has_preprocessed) {
        challenger.observe_commitment(proof.commitments.preprocessed);
    }
    for (const Val& v : public_values) {
        challenger.observe_val(v);
    }

    Challenge alpha = challenger.template sample_challenge<Challenge>();
    challenger.observe_commitment(proof.commitments.quotient_chunks);
    Challenge zeta = challenger.template sample_challenge<Challenge>();
    const Val omega_N = Val::two_adic_generator(log_degree);
    Challenge zeta_next = zeta * Challenge::from_base(omega_N);

    // ---- 3. Build VerifyCommitments for the PCS --------------------------
    Domain trace_domain    = pcs.natural_domain_for_degree(degree);
    Domain quotient_domain{ log_quotient_size, Val::one_val() };

    using VC = typename Pcs::VerifyCommitment;

    std::vector<VC> to_verify;
    to_verify.reserve(has_preprocessed ? 3 : 2);
    {
        // Trace round: 1 matrix, 2 points (zeta, zeta_next)
        VC vc;
        vc.commitment   = proof.commitments.trace;
        vc.domains      = { trace_domain };
        vc.points       = { { zeta, zeta_next } };
        vc.opened_values = {
            { proof.opened_values.trace_local, proof.opened_values.trace_next }
        };
        to_verify.push_back(std::move(vc));
    }
    {
        // Quotient round: 1 matrix of width D, 1 point (zeta)
        VC vc;
        vc.commitment   = proof.commitments.quotient_chunks;
        vc.domains      = { quotient_domain };
        vc.points       = { { zeta } };
        vc.opened_values = { { proof.opened_values.quotient_chunks } };
        to_verify.push_back(std::move(vc));
    }
    if (has_preprocessed) {
        VC vc;
        vc.commitment   = proof.commitments.preprocessed;
        vc.domains      = { trace_domain };
        vc.points       = { { zeta, zeta_next } };
        vc.opened_values = {
            { proof.opened_values.preprocessed_local, proof.opened_values.preprocessed_next }
        };
        to_verify.push_back(std::move(vc));
    }

    if (!pcs.verify(to_verify, proof.opening_proof, challenger)) {
        return false;
    }

    // ---- 4. Recompose Q(zeta) from quotient chunks -----------------------
    // Q(zeta) = sum_{d=0}^{D-1} basis_d * Q_d(zeta)
    // basis_d = Challenge with 1 at position d (i.e. extension-field basis).
    Challenge Q_zeta = Challenge::zero_val();
    for (std::size_t d = 0; d < D; ++d) {
        Challenge basis_d = Challenge::zero_val();
        basis_d[d] = Val::one_val();
        Q_zeta = Q_zeta + basis_d * proof.opened_values.quotient_chunks[d];
    }

    // ---- 5. Evaluate folded constraints at zeta -------------------------
    // Selectors at zeta (trace_domain has shift = 1 so y = zeta):
    //   z_h           = zeta^N - 1
    //   is_first_row  = z_h / (zeta - 1)
    //   is_last_row   = z_h / (zeta - omega_N^{-1})
    //   is_transition = zeta - omega_N^{-1}
    //   inv_vanishing = 1 / z_h
    Challenge zeta_pow_N = zeta.exp_u64(static_cast<uint64_t>(degree));
    Challenge one_ext    = Challenge::one_val();
    Challenge z_h        = zeta_pow_N - one_ext;
    Challenge omega_N_inv_c = Challenge::from_base(omega_N.inv());

    Challenge is_first_row  = z_h * (zeta - one_ext).inv();
    Challenge is_last_row   = z_h * (zeta - omega_N_inv_c).inv();
    Challenge is_transition = zeta - omega_N_inv_c;
    Challenge inv_vanishing = z_h.inv();

    using Folder = ConstraintFolder<Val, Challenge>;
    using MainWindow = typename Folder::MainWindow;

    MainWindow main_win(
        p3_air::ConstRowView<Challenge>(proof.opened_values.trace_local.data(),
                                        proof.opened_values.trace_local.size()),
        p3_air::ConstRowView<Challenge>(proof.opened_values.trace_next.data(),
                                        proof.opened_values.trace_next.size()));

    MainWindow pre_win;
    if (has_preprocessed) {
        pre_win = MainWindow(
            p3_air::ConstRowView<Challenge>(proof.opened_values.preprocessed_local.data(),
                                            proof.opened_values.preprocessed_local.size()),
            p3_air::ConstRowView<Challenge>(proof.opened_values.preprocessed_next.data(),
                                            proof.opened_values.preprocessed_next.size()));
    }

    Folder folder;
    folder.set_alpha(alpha);
    folder.set_windows(main_win, pre_win);
    folder.set_selectors(is_first_row, is_last_row, is_transition);
    folder.reset_accumulator();
    air.eval(folder);

    Challenge folded = folder.accumulator();
    return folded * inv_vanishing == Q_zeta;
}

} // namespace p3_uni_stark

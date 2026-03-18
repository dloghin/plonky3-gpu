#pragma once

#include "fri_params.hpp"
#include "fri_proof.hpp"
#include "fri_folding.hpp"
#include "p3_util/util.hpp"

#include <vector>
#include <cstddef>
#include <functional>
#include <utility>

namespace p3_fri {

// FriOpenings: a list of (log_height, evaluation) pairs sorted by descending
// log_height.  Each entry corresponds to one FRI input vector at a specific
// domain height.  The first (largest) entry seeds the folding, and subsequent
// entries are mixed in when the folded domain reaches their height.
template <typename Challenge>
using FriOpenings = std::vector<std::pair<size_t, Challenge>>;

// verify_fri: run the FRI verifier.
//
// Template parameters match prove_fri.
//
// commitments: the input oracle commitments (from the original MMCS commit)
// proof: the FRI proof
// challenger: mutable Fiat-Shamir transcript
// params: FRI parameters
// input_mmcs: the input MMCS instance (used to verify input openings)
// open_input: callback that, given (query_index, log_max_height, input_proof),
//             returns FriOpenings<Challenge> — a vector of
//             (log_height, reduced_opening) pairs sorted descending by
//             log_height.  The first entry MUST have log_height ==
//             log_max_height (the largest FRI input).
//
// Returns true if the proof verifies, false otherwise.
template <
    typename Val,
    typename Challenge,
    typename FriMmcs,
    typename Challenger,
    typename Witness,
    typename InputMmcs,
    typename InputProof
>
bool verify_fri(
    const FriParameters<FriMmcs>& params,
    const std::vector<typename InputMmcs::Commitment>& input_commitments,
    const FriProof<Challenge, FriMmcs, Witness, InputProof>& proof,
    Challenger& challenger,
    const InputMmcs& input_mmcs,
    std::function<FriOpenings<Challenge>(size_t, size_t, const InputProof&)> open_input
) {
    using Folding = TwoAdicFriFolding<Val, Challenge>;

    // -------------------------------------------------------------------------
    // Replay commit phase (to reconstruct betas)
    // -------------------------------------------------------------------------
    if (proof.commit_pow_witnesses.size() != proof.commit_phase_commits.size()) {
        return false;
    }

    std::vector<Challenge> betas;
    betas.reserve(proof.commit_phase_commits.size());

    for (size_t r = 0; r < proof.commit_phase_commits.size(); ++r) {
        challenger.observe_commitment(proof.commit_phase_commits[r]);

        if (!challenger.check_witness(params.commit_proof_of_work_bits,
                                      proof.commit_pow_witnesses[r])) {
            return false;
        }

        betas.push_back(challenger.sample_challenge());
    }

    // -------------------------------------------------------------------------
    // Validate and observe the final polynomial
    // -------------------------------------------------------------------------
    if (proof.final_poly.size() != params.final_poly_len()) {
        return false;
    }

    for (const auto& c : proof.final_poly) {
        challenger.observe_challenge(c);
    }

    // Bind the folding arities into the transcript (must match prover)
    if (!proof.query_proofs.empty()) {
        for (const auto& step : proof.query_proofs[0].commit_phase_openings) {
            challenger.observe_arity(static_cast<size_t>(step.log_arity));
        }
    }

    // -------------------------------------------------------------------------
    // Verify query-phase PoW
    // -------------------------------------------------------------------------
    if (!challenger.check_witness(params.query_proof_of_work_bits,
                                  proof.query_pow_witness)) {
        return false;
    }

    // -------------------------------------------------------------------------
    // Per-query verification
    // -------------------------------------------------------------------------
    size_t log_final_height = params.log_final_height();

    // Compute log_max_height by summing arities over all rounds
    size_t log_max_height = log_final_height;
    if (!proof.query_proofs.empty()) {
        for (const auto& step : proof.query_proofs[0].commit_phase_openings) {
            log_max_height += step.log_arity;
        }
    }

    for (size_t q = 0; q < params.num_queries; ++q) {
        if (q >= proof.query_proofs.size()) return false;
        const auto& qp = proof.query_proofs[q];

        size_t query_index = challenger.sample_bits(log_max_height);

        // Get reduced openings for all height groups at this query index
        auto reduced_openings = open_input(query_index, log_max_height, qp.input_proof);

        if (reduced_openings.empty() ||
            reduced_openings[0].first != log_max_height) {
            return false;
        }

        // Seed folded value from the largest height group
        Challenge folded = reduced_openings[0].second;
        size_t ro_idx = 1;  // index into remaining reduced_openings

        // Verify input proof
        if (!input_mmcs.verify_query(query_index, log_max_height,
                                     input_commitments, qp.input_proof, folded)) {
            return false;
        }

        // Fold through each commit-phase opening
        size_t cur_index      = query_index;
        size_t cur_log_height = log_max_height;

        for (size_t r = 0; r < qp.commit_phase_openings.size(); ++r) {
            const auto& step = qp.commit_phase_openings[r];
            size_t log_arity = step.log_arity;
            size_t arity     = size_t(1) << log_arity;

            size_t pos_in_row = cur_index & (arity - 1);
            size_t row_index  = cur_index >> log_arity;

            if (step.sibling_values.size() != arity - 1) {
                return false;
            }

            std::vector<Challenge> row_evals(arity);
            size_t sib = 0;
            for (size_t j = 0; j < arity; ++j) {
                if (j == pos_in_row) {
                    row_evals[j] = folded;
                } else {
                    row_evals[j] = step.sibling_values[sib++];
                }
            }

            if (!params.mmcs.verify_row(proof.commit_phase_commits[r],
                                        row_index,
                                        row_evals,
                                        step.opening_proof)) {
                return false;
            }

            folded = Folding::fold_row(row_index, cur_log_height, log_arity,
                                       betas[r], row_evals);

            cur_index      = row_index;
            cur_log_height -= log_arity;

            // Mix in the next input if its height matches the folded height
            if (ro_idx < reduced_openings.size() &&
                reduced_openings[ro_idx].first == cur_log_height) {
                Challenge beta_pow = betas[r].exp_power_of_2(log_arity);
                folded = folded + beta_pow * reduced_openings[ro_idx].second;
                ++ro_idx;
            }
        }

        // Check that we reached the expected final height
        if (cur_log_height != log_final_height) {
            return false;
        }

        // All reduced openings should have been consumed
        if (ro_idx != reduced_openings.size()) {
            return false;
        }

        // -------------------------------------------------------------------------
        // Check against final polynomial (Horner evaluation)
        // -------------------------------------------------------------------------
        {
            size_t rev_idx = p3_util::reverse_bits_len(cur_index, log_max_height);
            Val x_val = Val::two_adic_generator(log_max_height)
                            .exp_u64(static_cast<uint64_t>(rev_idx));

            Challenge eval = Challenge::zero_val();
            for (size_t i = proof.final_poly.size(); i > 0; --i) {
                eval = eval * embed_base<Val, Challenge>(x_val)
                     + proof.final_poly[i - 1];
            }

            if (folded != eval) {
                return false;
            }
        }
    }

    return true;
}

} // namespace p3_fri

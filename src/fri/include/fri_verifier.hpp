#pragma once

#include "fri_params.hpp"
#include "fri_proof.hpp"
#include "fri_folding.hpp"
#include "p3_util/util.hpp"

#include <vector>
#include <cstddef>
#include <functional>
#include <stdexcept>

namespace p3_fri {

// verify_fri: run the FRI verifier.
//
// Template parameters match prove_fri.
//
// commitments: the input oracle commitments (from the original MMCS commit)
// proof: the FRI proof
// challenger: Fiat-Shamir transcript (must be in the same state as the prover's)
// params: FRI parameters
// input_mmcs: the input MMCS instance (used to verify input openings)
// eval_at_query: callback that, given (query_index, log_max_height, opened_values),
//                returns the Challenge evaluation that should equal the FRI evaluation.
//                Signature: Challenge(size_t index, size_t log_height, InputProof&)
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
    std::function<Challenge(size_t, size_t, const InputProof&)> eval_at_query
) {
    using Folding = TwoAdicFriFolding<Val, Challenge>;

    // -------------------------------------------------------------------------
    // Replay commit phase (to reconstruct betas)
    // -------------------------------------------------------------------------
    std::vector<Challenge> betas;
    betas.reserve(proof.commit_phase_commits.size());

    for (size_t r = 0; r < proof.commit_phase_commits.size(); ++r) {
        // Observe this round's commitment
        challenger.observe_commitment(proof.commit_phase_commits[r]);

        // Verify PoW witness for this round
        if (!challenger.check_witness(params.commit_proof_of_work_bits,
                                      proof.commit_pow_witnesses[r])) {
            return false;
        }

        // Sample beta
        betas.push_back(challenger.sample_challenge());
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
    // Reconstruct log_max_height from number of rounds and log_final_height.
    // Each round i has log_arity = proof.query_proofs[0].commit_phase_openings[i].log_arity
    // (all queries see the same structure).
    size_t log_final_height = params.log_final_height();

    // Compute log_max_height by summing arities
    size_t log_max_height = log_final_height;
    if (!proof.query_proofs.empty()) {
        for (const auto& step : proof.query_proofs[0].commit_phase_openings) {
            log_max_height += step.log_arity;
        }
    }

    for (size_t q = 0; q < params.num_queries; ++q) {
        if (q >= proof.query_proofs.size()) return false;
        const auto& qp = proof.query_proofs[q];

        // Sample query index
        size_t query_index = challenger.sample_bits(log_max_height);

        // Evaluate the input oracle at this query index
        Challenge folded = eval_at_query(query_index, log_max_height, qp.input_proof);

        // Verify input proof
        if (!input_mmcs.verify_query(query_index, log_max_height,
                                     input_commitments, qp.input_proof, folded)) {
            return false;
        }

        // Now fold through each commit-phase opening
        size_t cur_index     = query_index;
        size_t cur_log_height = log_max_height;

        for (size_t r = 0; r < qp.commit_phase_openings.size(); ++r) {
            const auto& step = qp.commit_phase_openings[r];
            size_t log_arity = step.log_arity;
            size_t arity     = size_t(1) << log_arity;

            // Position within the coset row
            size_t pos_in_row = cur_index & (arity - 1);
            size_t row_index  = cur_index >> log_arity;

            // Reconstruct the full row from sibling_values + folded
            std::vector<Challenge> row_evals(arity);
            {
                size_t sib = 0;
                for (size_t j = 0; j < arity; ++j) {
                    if (j == pos_in_row) {
                        row_evals[j] = folded;
                    } else {
                        if (sib >= step.sibling_values.size()) return false;
                        row_evals[j] = step.sibling_values[sib++];
                    }
                }
            }

            // Verify the Merkle opening for this row
            if (!params.mmcs.verify_row(proof.commit_phase_commits[r],
                                        row_index,
                                        row_evals,
                                        step.opening_proof)) {
                return false;
            }

            // Apply the fold to get the next level's evaluation
            folded = Folding::fold_row(row_index, cur_log_height, log_arity,
                                       betas[r], row_evals);

            cur_index      = row_index;
            cur_log_height -= log_arity;
        }

        // -------------------------------------------------------------------------
        // Check against final polynomial
        // -------------------------------------------------------------------------
        // Throughout FRI, evaluations are stored in bit-reversed order:
        // position i holds the evaluation at domain point omega^(bit_rev(i, log_height)).
        //
        // The final polynomial is stored as proof.final_poly, which is obtained
        // by calling reverse_slice_index_bits on the last `current` vector.
        // So: final_poly[bit_rev(i, log_final)] = current[i]
        // Equivalently: current[i] = final_poly[bit_rev(i, log_final)]
        //
        // After all folding rounds, `folded` equals current[cur_index] in the
        // pre-reversal ordering.  The corresponding entry in final_poly is at
        // index bit_rev(cur_index, cur_log_height).
        if (cur_index >= proof.final_poly.size()) {
            return false;
        }
        size_t fp_index = p3_util::reverse_bits_len(cur_index, cur_log_height);
        if (fp_index >= proof.final_poly.size()) {
            return false;
        }

        if (folded != proof.final_poly[fp_index]) {
            return false;
        }
    }

    return true;
}

} // namespace p3_fri

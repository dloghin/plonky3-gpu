#pragma once

#include "fri_params.hpp"
#include "fri_proof.hpp"
#include "fri_folding.hpp"
#include "p3_util/util.hpp"

#include <vector>
#include <cstddef>
#include <algorithm>
#include <stdexcept>

namespace p3_fri {

// prove_fri: run the FRI prover.
//
// Template parameters:
//   Val       - base prime field type (must provide two_adic_generator)
//   Challenge - extension field Challenge type
//   FriMmcs   - Merkle multi-commitment scheme type
//   Challenger - Fiat-Shamir transcript type
//   Witness    - type for PoW witnesses
//   InputMmcs  - MMCS for the input oracle (for query-phase openings)
//   InputProof - proof type returned by InputMmcs::open
//
// inputs: list of input evaluation vectors, sorted descending by length (largest first).
//         Each must have a power-of-two length.
// params: FRI parameters
// challenger: mutable Fiat-Shamir transcript
// input_data: prover data from InputMmcs (one entry per input vector, parallel to inputs)
// input_mmcs: the input MMCS instance (used to open during query phase)
//
// Returns a FriProof<Challenge, FriMmcs, Witness, InputProof>.
template <
    typename Val,
    typename Challenge,
    typename FriMmcs,
    typename Challenger,
    typename Witness,
    typename InputMmcs,
    typename InputProof
>
FriProof<Challenge, FriMmcs, Witness, InputProof> prove_fri(
    const FriParameters<FriMmcs>& params,
    std::vector<std::vector<Challenge>> inputs,           // by value; sorted desc by size
    Challenger& challenger,
    const std::vector<typename InputMmcs::ProverData>& input_data,
    const InputMmcs& input_mmcs
) {
    using Folding = TwoAdicFriFolding<Val, Challenge>;
    using Proof   = FriProof<Challenge, FriMmcs, Witness, InputProof>;

    if (inputs.empty()) {
        throw std::invalid_argument("prove_fri: no input vectors");
    }

    // The smallest input must be strictly larger than the final domain height
    // so that at least one folding round is possible (matching Rust's assert).
    if (params.log_final_poly_len > 0) {
        size_t log_min_height = p3_util::log2_strict_usize(inputs.back().size());
        size_t log_final_h = params.log_final_height();
        if (log_min_height <= log_final_h) {
            throw std::invalid_argument(
                "prove_fri: smallest input (log_height=" +
                std::to_string(log_min_height) +
                ") must be strictly larger than log_final_height=" +
                std::to_string(log_final_h));
        }
    }

    // -------------------------------------------------------------------------
    // Commit phase
    // -------------------------------------------------------------------------
    std::vector<Challenge> current = std::move(inputs[0]);
    size_t input_idx = 1;  // index into remaining inputs

    std::vector<typename FriMmcs::Commitment> commit_phase_commits;
    std::vector<typename FriMmcs::ProverData>  commit_phase_data;
    std::vector<Witness>                       commit_pow_witnesses;

    size_t log_final_height = params.log_final_height();

    while (current.size() > (size_t(1) << log_final_height)) {
        size_t log_height = p3_util::log2_strict_usize(current.size());

        // Determine the next input's log height (if any remains)
        bool has_next_input = (input_idx < inputs.size());
        size_t next_input_log_height = has_next_input
            ? p3_util::log2_strict_usize(inputs[input_idx].size())
            : 0;

        size_t log_arity = compute_log_arity_for_round(
            log_height,
            has_next_input,
            next_input_log_height,
            log_final_height,
            params.max_log_arity
        );

        if (log_arity == 0) {
            // Nothing to fold this round (shouldn't happen normally,
            // but guard against infinite loops)
            break;
        }

        size_t arity = size_t(1) << log_arity;

        // Commit the current evaluations as a matrix of width=arity.
        // The matrix has (current.size() / arity) rows and arity columns.
        auto [commit, cdata] = params.mmcs.commit_matrix(current, arity);
        challenger.observe_commitment(commit);
        commit_phase_commits.push_back(commit);
        commit_phase_data.push_back(std::move(cdata));

        // Commit-phase proof-of-work
        Witness commit_witness = challenger.grind(params.commit_proof_of_work_bits);
        commit_pow_witnesses.push_back(commit_witness);

        // Sample folding challenge beta
        Challenge beta = challenger.sample_challenge();

        // Fold: reshape current into [height/arity][arity] and fold each row
        current = Folding::fold_matrix(log_height, log_arity, beta, current);

        // Mix in the next input if its height now matches.
        // Multiply by beta^arity to maintain independence of the random
        // linear combination (matches Rust: beta_pow = beta.exp_power_of_2(log_arity)).
        if (input_idx < inputs.size()) {
            size_t nlen = inputs[input_idx].size();
            if (nlen == current.size()) {
                Challenge beta_pow = beta.exp_power_of_2(log_arity);
                const auto& next_input = inputs[input_idx];
                for (size_t k = 0; k < current.size(); ++k) {
                    current[k] = current[k] + beta_pow * next_input[k];
                }
                ++input_idx;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Final polynomial
    // -------------------------------------------------------------------------
    // `current` now has length 2^log_final_height = blowup * final_poly_len.
    // The polynomial has degree < final_poly_len, so we truncate, bit-reverse,
    // and run an IDFT to recover the coefficients (matching Rust's approach).
    size_t fpl = params.final_poly_len();
    current.resize(fpl);
    p3_util::reverse_slice_index_bits(current);

    // Naive IDFT: c_k = (1/n) * sum_{j=0}^{n-1} e_j * omega^(-j*k)
    // where omega is the n-th root of unity and e_j are the evaluations.
    // This is O(n^2) but final_poly_len is tiny (typically 1–4).
    {
        size_t n = current.size();
        Val omega = Val::two_adic_generator(p3_util::log2_strict_usize(n));
        Val omega_inv = omega.exp_u64(static_cast<uint64_t>(n - 1));  // omega^(-1)

        // Compute n_inv in the base field
        Val n_val(static_cast<uint32_t>(n));
        Val n_inv = n_val.inv();
        Challenge n_inv_c = embed_base<Val, Challenge>(n_inv);

        std::vector<Challenge> coeffs(n, Challenge::zero_val());
        Val omega_inv_k = Val::one_val();  // omega_inv^k
        for (size_t k = 0; k < n; ++k) {
            Challenge sum = Challenge::zero_val();
            Val omega_inv_kj = Val::one_val();  // omega_inv^(k*j)
            for (size_t j = 0; j < n; ++j) {
                sum = sum + current[j] * embed_base<Val, Challenge>(omega_inv_kj);
                omega_inv_kj = omega_inv_kj * omega_inv_k;
            }
            coeffs[k] = sum * n_inv_c;
            omega_inv_k = omega_inv_k * omega_inv;
        }
        current = std::move(coeffs);
    }
    std::vector<Challenge> final_poly = std::move(current);

    // Observe final polynomial coefficients to the challenger
    for (const auto& c : final_poly) {
        challenger.observe_challenge(c);
    }

    // Bind the chosen folding arities into the transcript
    for (size_t r = 0; r < commit_phase_data.size(); ++r) {
        size_t la = params.mmcs.log_width(commit_phase_data[r]);
        challenger.observe_arity(la);
    }

    // -------------------------------------------------------------------------
    // Query phase
    // -------------------------------------------------------------------------
    // PoW for query phase
    Witness query_pow_witness = challenger.grind(params.query_proof_of_work_bits);

    std::vector<QueryProof<Challenge, FriMmcs, InputProof>> query_proofs;
    query_proofs.reserve(params.num_queries);

    // Determine log_max_height from the commit_phase_data sizes once, since it is
    // constant across all queries. The first round's matrix has height =
    // original_height / arity_0, so log_max_height = log_height(commit_phase_data[0])
    // + log_arity_0. We reconstruct this by summing arities over all rounds and
    // adding log_final_height.
    size_t log_max_height = log_final_height;
    for (size_t r = 0; r < commit_phase_data.size(); ++r) {
        log_max_height += params.mmcs.log_width(commit_phase_data[r]);
    }

    for (size_t q = 0; q < params.num_queries; ++q) {
        size_t query_index = challenger.sample_bits(log_max_height);

        // Open input at this index
        InputProof inp_proof;
        input_mmcs.open(query_index, input_data, inp_proof);

        // Open each commit-phase matrix at the appropriate index
        std::vector<CommitPhaseProofStep<Challenge, FriMmcs>> commit_openings;
        size_t cur_index = query_index;

        for (size_t r = 0; r < commit_phase_data.size(); ++r) {
            size_t cur_log_arity  = params.mmcs.log_width(commit_phase_data[r]);
            size_t cur_arity = size_t(1) << cur_log_arity;

            // The row index in the committed matrix is cur_index / arity
            size_t row_index = cur_index >> cur_log_arity;
            // The position within the row (sibling positions)
            size_t pos_in_row = cur_index & (cur_arity - 1);

            // Get all arity values for this row from the MMCS
            auto row_vals = params.mmcs.get_row(commit_phase_data[r], row_index);

            // sibling_values = all values except pos_in_row
            std::vector<Challenge> siblings;
            siblings.reserve(cur_arity - 1);
            for (size_t j = 0; j < cur_arity; ++j) {
                if (j != pos_in_row) {
                    siblings.push_back(row_vals[j]);
                }
            }

            typename FriMmcs::OpeningProof op;
            params.mmcs.open_row(commit_phase_data[r], row_index, op);

            CommitPhaseProofStep<Challenge, FriMmcs> step;
            step.log_arity      = static_cast<uint8_t>(cur_log_arity);
            step.sibling_values = std::move(siblings);
            step.opening_proof  = std::move(op);
            commit_openings.push_back(std::move(step));

            // Advance to next level
            cur_index = row_index;
        }

        QueryProof<Challenge, FriMmcs, InputProof> qp;
        qp.input_proof           = std::move(inp_proof);
        qp.commit_phase_openings = std::move(commit_openings);
        query_proofs.push_back(std::move(qp));
    }

    Proof proof;
    proof.commit_phase_commits = std::move(commit_phase_commits);
    proof.commit_pow_witnesses = std::move(commit_pow_witnesses);
    proof.query_proofs         = std::move(query_proofs);
    proof.final_poly           = std::move(final_poly);
    proof.query_pow_witness    = query_pow_witness;
    return proof;
}

} // namespace p3_fri

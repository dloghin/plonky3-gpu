#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>

namespace p3_fri {

// One step in the commit-phase proof for a single query.
// Holds the sibling values at one folding level plus a Merkle opening proof.
template <typename Challenge, typename FriMmcs>
struct CommitPhaseProofStep {
    uint8_t log_arity = 0;
    std::vector<Challenge> sibling_values;  // arity - 1 sibling values
    typename FriMmcs::OpeningProof opening_proof;
};

// The query proof for one random query index.
template <typename Challenge, typename FriMmcs, typename InputProof>
struct QueryProof {
    InputProof input_proof;
    std::vector<CommitPhaseProofStep<Challenge, FriMmcs>> commit_phase_openings;
};

// Full FRI proof.
template <typename Challenge, typename FriMmcs, typename Witness, typename InputProof>
struct FriProof {
    std::vector<typename FriMmcs::Commitment> commit_phase_commits;
    std::vector<Witness> commit_pow_witnesses;
    std::vector<QueryProof<Challenge, FriMmcs, InputProof>> query_proofs;
    std::vector<Challenge> final_poly;  // Coefficients of final polynomial
    Witness query_pow_witness{};
};

} // namespace p3_fri

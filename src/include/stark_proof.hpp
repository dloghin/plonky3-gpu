#pragma once

/**
 * @file stark_proof.hpp
 * @brief Proof structure for the uni-STARK prover/verifier.
 *
 * Mirrors plonky3/uni-stark/src/proof.rs, tailored to our C++ PCS API.
 *
 * Layout:
 *   Commitments         : trace / quotient / (optional) preprocessed roots.
 *   OpenedValues        : opened polynomial evaluations at zeta and zeta*g.
 *   OpeningProof        : the PCS `FullFriProof` that certifies the openings.
 *   degree_bits         : log2 of the trace height.
 *   log_num_quotient_chunks : log2 of how many sub-chunks the quotient was
 *                             split into (quotient_domain_size = 2^(degree_bits
 *                             + log_num_quotient_chunks)). Per-chunk opened
 *                             values are bundled into a single width-D matrix;
 *                             this field is retained so the verifier can rebuild
 *                             the constraint-to-quotient relation.
 */

#include <cstddef>
#include <vector>

namespace p3_uni_stark {

template<typename SC>
struct Commitments {
    typename SC::Pcs::InputCommitment trace;
    typename SC::Pcs::InputCommitment quotient_chunks;
    // The preprocessed commitment is committed once per AIR instance and
    // could be held outside the proof; we include it here for simplicity.
    // Empty (default-constructed) when the AIR has no preprocessed trace.
    typename SC::Pcs::InputCommitment preprocessed;
    bool has_preprocessed = false;
};

template<typename SC>
struct OpenedValues {
    using Challenge = typename SC::Challenge;

    std::vector<Challenge> trace_local;
    std::vector<Challenge> trace_next;

    /// D Challenge values representing the quotient polynomial `Q(zeta)`
    /// decomposed as `Q = sum_d basis_d * Q_d`. Stored as a single vector of
    /// length `Challenge::DEGREE` for simplicity; this is equivalent to Rust's
    /// `quotient_chunks: Vec<Vec<Challenge>>` with a single outer chunk.
    std::vector<Challenge> quotient_chunks;

    std::vector<Challenge> preprocessed_local;  // empty when no preprocessed
    std::vector<Challenge> preprocessed_next;   // empty when no preprocessed
};

template<typename SC>
struct Proof {
    using FullFriProof = typename SC::Pcs::FullFriProof;

    Commitments<SC>    commitments;
    OpenedValues<SC>   opened_values;
    FullFriProof       opening_proof;
    std::size_t        degree_bits = 0;
    std::size_t        log_num_quotient_chunks = 0;
    std::size_t        preprocessed_width = 0;
};

} // namespace p3_uni_stark

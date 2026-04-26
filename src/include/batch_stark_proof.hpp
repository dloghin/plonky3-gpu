#pragma once

/**
 * @file batch_stark_proof.hpp
 * @brief Data structures for the batch-STARK prover/verifier.
 *
 * Mirrors `plonky3/batch-stark/src/proof.rs` and `prover.rs` (StarkInstance),
 * adapted to the C++ PCS API. A batch-STARK proves multiple AIR instances
 * in a single proof that shares one FRI opening proof across all of them.
 *
 * Design mirrors the flat-commitment pattern used by `stark_proof.hpp`:
 *   - All traces are committed in a single PCS commitment (one matrix per
 *     instance).
 *   - All quotient polynomials are committed in a single PCS commitment
 *     (one width-D matrix per instance, extension-field decomposed).
 *   - The FRI opening proof covers every matrix across both commitments at
 *     the shared out-of-domain point zeta.
 *
 * Lookups are not yet implemented; empty `Lookup` lists are preserved for
 * future extension.
 */

#include "air.hpp"
#include "dense_matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace p3_batch_stark {

enum class LookupDirection : uint8_t {
    InputInTable = 0,
    TableInInput = 1,
};

template <typename F>
struct Lookup {
    std::size_t     source_instance = 0;
    std::size_t     source_column = 0;
    std::size_t     table_instance = 0;
    std::size_t     table_column = 0;
    LookupDirection direction = LookupDirection::InputInTable;
};

/// A single AIR instance to be proven within a batch.
///
/// The AIR is referenced by pointer so that the caller retains ownership;
/// this matches the Rust `StarkInstance<'a, SC, A>` borrow pattern.
template <typename SC, typename AIR>
struct StarkInstance {
    using Val = typename SC::Val;

    const AIR*                            air = nullptr;
    const p3_matrix::RowMajorMatrix<Val>* trace = nullptr;
    std::vector<Val>                      public_values;

    StarkInstance() = default;
    StarkInstance(const AIR& a,
                  const p3_matrix::RowMajorMatrix<Val>& t,
                  std::vector<Val> pvs = {})
        : air(&a), trace(&t), public_values(std::move(pvs)) {}
};

/// Commitments for a batch-STARK proof.
template <typename SC>
struct BatchCommitments {
    typename SC::Pcs::InputCommitment main;             // all traces
    typename SC::Pcs::InputCommitment quotient_chunks;  // all quotient matrices
};

/// Opened values for a single instance inside a batch proof.
///
/// `trace_local` and `trace_next` are the per-instance opening of the main
/// trace at zeta and zeta * g_i (where g_i is the generator of the i-th
/// trace domain). `quotient_chunks` are the D extension-field coefficients
/// that recompose Q_i(zeta).
template <typename SC>
struct OpenedValuesPerInstance {
    using Challenge = typename SC::Challenge;

    std::vector<Challenge> trace_local;
    std::vector<Challenge> trace_next;
    std::vector<Challenge> quotient_chunks;  // length = Challenge::DEGREE
};

/// Opened values for all instances in a batch-STARK proof.
template <typename SC>
struct BatchOpenedValues {
    std::vector<OpenedValuesPerInstance<SC>> instances;
};

/// A proof of batched STARK instances.
template <typename SC>
struct BatchProof {
    using FullFriProof = typename SC::Pcs::FullFriProof;

    BatchCommitments<SC>      commitments;
    BatchOpenedValues<SC>     opened_values;
    FullFriProof              opening_proof;
    /// Per-instance log2 of the extended trace domain size.
    std::vector<std::size_t>  degree_bits;
    /// Per-instance log2 of the number of quotient chunks.
    std::vector<std::size_t>  log_num_quotient_chunks;
};

/// Data shared between prover and verifier that is not otherwise derivable
/// from the AIRs alone.
///
/// This is a placeholder for future lookup/preprocessed extensions so that
/// the verifier signature can stay stable. An `empty(n)` constructor returns
/// a `CommonData` suitable for AIRs without lookups/preprocessed columns.
template <typename SC>
struct CommonData {
    /// Number of instances this common data was built for.
    std::size_t num_instances = 0;
    /// Per-instance lookup declarations.
    std::vector<std::vector<Lookup<typename SC::Val>>> lookups;

    static CommonData<SC> empty(std::size_t n) {
        CommonData<SC> c;
        c.num_instances = n;
        c.lookups.resize(n);
        return c;
    }
};

} // namespace p3_batch_stark

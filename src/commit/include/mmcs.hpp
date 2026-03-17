#pragma once

/**
 * @file mmcs.hpp
 * @brief Mixed Matrix Commitment Scheme (MMCS) abstractions.
 *
 * Mirrors plonky3/commit/src/mmcs.rs.
 *
 * Provides:
 *   - BatchOpening<F, Proof>  -- opened values + Merkle/opening proof
 *   - MmcsBase<Derived, F>    -- CRTP base class with default helpers
 *
 * Required interface for any type satisfying the Mmcs concept:
 *
 *   Types (as public member type aliases):
 *     Commitment   -- commitment output (e.g. MerkleCap)
 *     ProverData   -- prover-side data (e.g. MerkleTree)
 *     Proof        -- opening proof type
 *
 *   Methods:
 *     std::pair<Commitment, ProverData>
 *         commit(std::vector<RowMajorMatrix<F>> matrices);
 *
 *     BatchOpening<F, Proof>
 *         open_batch(size_t index, const ProverData& data);
 *
 *     bool
 *         verify_batch(const Commitment& commit,
 *                      const std::vector<Dimensions>& dims,
 *                      size_t index,
 *                      const BatchOpening<F, Proof>& opening);
 *
 *     std::vector<const RowMajorMatrix<F>*>
 *         get_matrices(const ProverData& data) const;
 *
 * MmcsBase provides default implementations of:
 *   - commit_matrix(RowMajorMatrix<F>)  -- single-matrix convenience wrapper
 *   - get_max_height(ProverData)        -- max height across committed matrices
 */

#include "dense_matrix.hpp"
#include "domain.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace p3_commit {

// ---------------------------------------------------------------------------
// BatchOpening
//
// Holds the data produced by opening a batch of committed matrices at a
// single row index:
//   opened_values[i]  -- the row of the i-th matrix at the queried index
//   opening_proof     -- Merkle/FRI proof authenticating the opened rows
//
// Templated on:
//   F     -- field element type (base or extension field)
//   Proof -- proof type from the underlying MMCS implementation
//
// Mirrors Rust's BatchOpening<F, M: Mmcs<F>>.
// ---------------------------------------------------------------------------
template<typename F, typename Proof>
struct BatchOpening {
    /// opened_values[i] is the row of the i-th committed matrix at the
    /// queried row index.
    std::vector<std::vector<F>> opened_values;

    /// The opening proof authenticating the above values.
    Proof opening_proof;

    BatchOpening() = default;
    BatchOpening(std::vector<std::vector<F>> vals, Proof proof)
        : opened_values(std::move(vals)), opening_proof(std::move(proof)) {}
};

// ---------------------------------------------------------------------------
// MmcsBase  (CRTP base)
//
// Concrete MMCS implementations should inherit from this class and provide
// the required types and methods documented above.
//
// MmcsBase adds two convenience helpers on top of the required interface:
//   - commit_matrix  (commit a single matrix)
//   - get_max_height (maximum committed matrix height in ProverData)
//
// Usage:
//   class MyMmcs : public MmcsBase<MyMmcs, MyFieldType> { ... };
// ---------------------------------------------------------------------------
template<typename Derived, typename F>
class MmcsBase {
public:
    // -----------------------------------------------------------------------
    // commit_matrix  --  convenience wrapper around commit()
    //
    // Commits a single RowMajorMatrix by delegating to the derived class's
    // commit(vector<RowMajorMatrix<F>>) method.
    // -----------------------------------------------------------------------
    auto commit_matrix(p3_matrix::RowMajorMatrix<F> matrix) {
        std::vector<p3_matrix::RowMajorMatrix<F>> matrices;
        matrices.push_back(std::move(matrix));
        return static_cast<Derived*>(this)->commit(std::move(matrices));
    }

    // -----------------------------------------------------------------------
    // get_max_height  --  maximum row-count across all committed matrices
    //
    // Iterates the matrices stored in prover_data and returns the height of
    // the tallest one, or 0 if there are no matrices.
    //
    // ProverData is deduced from the argument so this method can be
    // instantiated even when the Derived type is still incomplete at the
    // point of inheritance.
    // -----------------------------------------------------------------------
    template<typename ProverData>
    size_t get_max_height(const ProverData& prover_data) const {
        auto mats = static_cast<const Derived*>(this)->get_matrices(prover_data);
        size_t max_h = 0;
        for (const auto* m : mats) {
            if (m && m->height() > max_h) {
                max_h = m->height();
            }
        }
        return max_h;
    }
};

} // namespace p3_commit

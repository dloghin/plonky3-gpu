#pragma once

/**
 * @file pcs.hpp
 * @brief Polynomial Commitment Scheme (PCS) abstractions.
 *
 * Mirrors plonky3/commit/src/pcs.rs.
 *
 * Provides:
 *   - OpenedValues<Challenge>  -- type alias for opened polynomial evaluations
 *   - PcsBase<Derived, Val, Challenge, Challenger>  -- CRTP base class
 *
 * Required interface for any type satisfying the Pcs concept:
 *
 *   Types (as public member type aliases):
 *     Domain       -- polynomial evaluation domain (e.g. TwoAdicMultiplicativeCoset)
 *     Commitment   -- commitment output
 *     ProverData   -- prover-side data
 *     OpeningProof -- proof of correct opening
 *
 *   Methods:
 *     Domain
 *         natural_domain_for_degree(size_t degree);
 *
 *     std::pair<Commitment, ProverData>
 *         commit(std::vector<std::pair<Domain, RowMajorMatrix<Val>>> evaluations);
 *
 *     std::pair<OpenedValues<Challenge>, OpeningProof>
 *         open(std::vector<OpenData> data, Challenger& challenger);
 *
 *     bool
 *         verify(std::vector<CommitmentsWithPoints> data,
 *                const OpeningProof& proof,
 *                Challenger& challenger);
 *
 * Where:
 *   OpenData           = std::pair<ProverData, std::vector<std::vector<Challenge>>>
 *   CommitmentsWithPoints = std::pair<Commitment, std::vector<std::pair<Challenge,
 *                                std::vector<std::vector<Challenge>>>>>
 */

#include "domain.hpp"
#include "dense_matrix.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace p3_commit {

// ---------------------------------------------------------------------------
// OpenedValues
//
// The result of opening a collection of committed polynomial batches at
// multiple challenge points.
//
// Layout (mirrors Rust's OpenedValues<Challenge>):
//   OpenedValues[batch_idx][matrix_idx][point_idx]  =  Challenge
//
//   - batch_idx  : which (Commitment, ProverData) pair was opened
//   - matrix_idx : which matrix within that commitment
//   - point_idx  : which challenge point the polynomial was evaluated at
// ---------------------------------------------------------------------------
template<typename Challenge>
using OpenedValues = std::vector<std::vector<std::vector<Challenge>>>;

// ---------------------------------------------------------------------------
// PcsBase  (CRTP base)
//
// Concrete PCS implementations inherit from this class and provide the types
// and methods documented in the file header above.
//
// PcsBase currently adds no additional default implementations — it serves as
// a documentation anchor and compile-time tagging mechanism.
//
// Usage:
//   class MyPcs : public PcsBase<MyPcs, BabyBear, BabyBear4, MyChallenger> {
//     ...
//   };
// ---------------------------------------------------------------------------
template<typename Derived, typename Val, typename Challenge, typename Challenger>
class PcsBase {
public:
    // -----------------------------------------------------------------------
    // Type aliases for Val and Challenge so downstream code can inspect them.
    // -----------------------------------------------------------------------
    using ValType       = Val;
    using ChallengeType = Challenge;

    // -----------------------------------------------------------------------
    // natural_domain_for_degree
    //
    // Default implementation delegates to the MMCS natural domain helper.
    // Derived classes may override this if the domain type differs.
    // -----------------------------------------------------------------------
    TwoAdicMultiplicativeCoset<Val> natural_domain_for_degree(size_t degree) {
        return TwoAdicMultiplicativeCoset<Val>::natural_domain_for_degree(degree);
    }
};

} // namespace p3_commit

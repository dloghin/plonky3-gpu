#pragma once

/**
 * @file circle_stark.hpp
 * @brief Small Circle-STARK proving facade.
 *
 * This header provides the Circle-domain pieces needed by tests and examples:
 * trace commitment over a standard Circle domain, opening at boundary points,
 * and verifier-side recomputation through `CirclePcs`.
 */

#include "circle_pcs.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_circle {

template <typename Val>
struct CircleStarkProof {
    using Pcs = CirclePcs<Val>;
    using Domain = typename Pcs::Domain;
    using Commitment = typename Pcs::Commitment;
    using OpeningProof = typename Pcs::OpeningProof;

    Domain domain;
    Commitment trace_commitment;
    std::vector<std::vector<Val>> opened_values;
    OpeningProof opening_proof;
};

template <typename Val>
class CircleStark {
public:
    using Matrix = p3_matrix::RowMajorMatrix<Val>;
    using Pcs = CirclePcs<Val>;
    using Proof = CircleStarkProof<Val>;

    template <typename Challenger>
    Proof prove(const Matrix& trace, Challenger& challenger) const {
        if (trace.height() == 0 || (trace.height() & (trace.height() - 1u)) != 0) {
            throw std::invalid_argument("CircleStark::prove: trace height must be a power of two");
        }
        if (trace.height() < 2) {
            throw std::invalid_argument("CircleStark::prove: trace height must be at least two");
        }

        const auto domain = CircleDomain<Val>::standard(p3_util::log2_strict_usize(trace.height()));
        auto [commitment, pd] = pcs_.commit({{domain, trace}});

        const auto first = domain.at(0);
        const auto last = domain.at(domain.size() - 1);
        auto [opened, opening_proof] = pcs_.open({{&pd, {{first, last}}}}, challenger);

        Proof proof;
        proof.domain = domain;
        proof.trace_commitment = commitment;
        proof.opened_values = std::move(opened[0][0]);
        proof.opening_proof = std::move(opening_proof);
        return proof;
    }

    template <typename Challenger>
    bool verify(const Proof& proof, Challenger& challenger) const {
        typename Pcs::VerifyCommitment vc;
        vc.commitment = proof.trace_commitment;
        vc.domains = {proof.domain};
        vc.points = {{proof.domain.at(0), proof.domain.at(proof.domain.size() - 1)}};
        vc.opened_values = {proof.opened_values};
        return pcs_.verify({vc}, proof.opening_proof, challenger);
    }

private:
    Pcs pcs_;
};

} // namespace p3_circle

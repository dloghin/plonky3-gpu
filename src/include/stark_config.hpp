#pragma once

/**
 * @file stark_config.hpp
 * @brief StarkGenericConfig: type bundle for the uni-STARK prover/verifier.
 *
 * Mirrors plonky3/uni-stark/src/config.rs but expressed with C++ templates
 * instead of a Rust trait. A concrete "config" in this codebase is any struct
 * that exposes the same member type aliases and provides a `pcs()` accessor
 * returning a reference to a PCS instance. Prover and verifier are templated
 * on the config type so they can be instantiated with any compatible bundle.
 *
 * Required member types:
 *   Val        : base prime field (e.g. BabyBear)
 *   Challenge  : extension field for challenges and opened values (e.g. BabyBear4)
 *   Pcs        : polynomial commitment scheme (e.g. TwoAdicFriPcs)
 *   Challenger : Fiat-Shamir transcript
 *
 * Required methods:
 *   Pcs& pcs()
 *
 * The caller is responsible for providing fresh Challenger instances to
 * `prove()` and `verify()` (typically default-constructed or cloned from a
 * configured initial-state instance).
 */

#include <utility>

namespace p3_uni_stark {

/// A plain config struct that bundles the four types plus a PCS instance.
/// The caller constructs and passes fresh `Challenger`s to `prove`/`verify`.
template<typename ValT, typename ChallengeT, typename PcsT, typename ChallengerT>
class StarkConfig {
public:
    using Val        = ValT;
    using Challenge  = ChallengeT;
    using Pcs        = PcsT;
    using Challenger = ChallengerT;

    explicit StarkConfig(Pcs pcs) : pcs_(std::move(pcs)) {}

    Pcs& pcs() { return pcs_; }
    const Pcs& pcs() const { return pcs_; }

private:
    Pcs pcs_;
};

} // namespace p3_uni_stark

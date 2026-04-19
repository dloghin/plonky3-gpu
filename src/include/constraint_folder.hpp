#pragma once

/**
 * @file constraint_folder.hpp
 * @brief Single-accumulator constraint folder used by both the uni-STARK
 *        prover and verifier.
 *
 * Mirrors the VerifierConstraintFolder from plonky3/uni-stark/src/folder.rs.
 * We intentionally use the same single-accumulator pattern for both sides so
 * prover and verifier agree on constraint ordering without packed SIMD. The
 * folder implements `AirBuilder<Val, Challenge, Challenge>`; the AIR's
 * `eval()` drives it by calling `assert_zero` for each constraint.
 */

#include "air.hpp"

#include <cstddef>
#include <vector>

namespace p3_uni_stark {

/// Constraint folder.
///
/// Evaluates AIR constraints at a single point in the extension field, folding
/// them into `accumulator` via `acc := acc*alpha + constraint`. After all
/// constraints are evaluated the accumulator equals
/// `sum_i alpha^(n-1-i) * c_i(...)`.
///
/// On the prover side this is instantiated per-row of the quotient domain
/// using lifted trace values (`Challenge::from_base(trace_val)`). On the
/// verifier side the same pattern is used with the opened values at `zeta`
/// and `zeta * g`.
template<typename Val, typename Challenge>
class ConstraintFolder : public p3_air::AirBuilder<Val, Challenge, Challenge> {
public:
    using Base         = p3_air::AirBuilder<Val, Challenge, Challenge>;
    using MainWindow   = typename Base::MainWindow;
    using Field        = Val;
    using ExprType     = Challenge;

    ConstraintFolder() : alpha_(Challenge::zero_val()), accumulator_(Challenge::zero_val()) {}

    /// Assign the windows for the current evaluation point.
    void set_windows(MainWindow main, MainWindow preprocessed) {
        main_window_         = std::move(main);
        preprocessed_window_ = std::move(preprocessed);
    }

    void set_selectors(const Challenge& is_first_row,
                       const Challenge& is_last_row,
                       const Challenge& is_transition) {
        is_first_row_   = is_first_row;
        is_last_row_    = is_last_row;
        is_transition_  = is_transition;
    }

    void set_alpha(const Challenge& alpha) { alpha_ = alpha; }

    void reset_accumulator() { accumulator_ = Challenge::zero_val(); }

    const Challenge& accumulator() const { return accumulator_; }

    MainWindow main() const override {
        return main_window_;
    }

    const MainWindow& preprocessed() const override {
        return preprocessed_window_;
    }

    Challenge is_first_row() const override { return is_first_row_; }
    Challenge is_last_row()  const override { return is_last_row_; }
    Challenge is_transition() const override { return is_transition_; }

    void assert_zero(const Challenge& expression) override {
        accumulator_ = accumulator_ * alpha_ + expression;
    }

private:
    MainWindow main_window_;
    MainWindow preprocessed_window_;
    Challenge  is_first_row_{Challenge::zero_val()};
    Challenge  is_last_row_{Challenge::zero_val()};
    Challenge  is_transition_{Challenge::zero_val()};
    Challenge  alpha_;
    Challenge  accumulator_;
};

} // namespace p3_uni_stark

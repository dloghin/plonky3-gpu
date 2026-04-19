#pragma once

#include "air.hpp"
#include "matrix.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace p3_air {

struct ConstraintViolation {
    size_t row;
    size_t constraint;
};

template<typename F>
class DebugConstraintBuilder : public AirBuilder<F, F, F> {
public:
    using MainWindow = typename AirBuilder<F, F, F>::MainWindow;

    DebugConstraintBuilder(
        const p3_matrix::Matrix<F>& trace,
        size_t row_index,
        std::vector<ConstraintViolation>* failures,
        const p3_matrix::Matrix<F>* preprocessed_trace = nullptr)
        : trace_(trace),
          row_index_(row_index),
          current_row_(trace.row(row_index)),
          next_row_(trace.row((row_index + 1) % trace.height())),
          preprocessed_current_(preprocessed_trace == nullptr ? std::vector<F>{} : preprocessed_trace->row(row_index)),
          preprocessed_next_(preprocessed_trace == nullptr ? std::vector<F>{} : preprocessed_trace->row((row_index + 1) % trace.height())),
          preprocessed_window_(preprocessed_trace ? &preprocessed_current_ : nullptr, preprocessed_trace ? &preprocessed_next_ : nullptr),
          failures_(failures) {}

    MainWindow main() const override {
        return MainWindow(&current_row_, &next_row_);
    }

    const MainWindow& preprocessed() const override {
        return preprocessed_window_;
    }

    F is_first_row() const override {
        return row_index_ == 0 ? F{1} : F{};
    }

    F is_last_row() const override {
        return row_index_ + 1 == trace_.height() ? F{1} : F{};
    }

    F is_transition() const override {
        return row_index_ + 1 == trace_.height() ? F{} : F{1};
    }

    void assert_zero(const F& expression) override {
        if (expression != F{}) {
            failures_->push_back(ConstraintViolation{
                row_index_,
                constraint_index_,
            });
        }
        ++constraint_index_;
    }

private:
    const p3_matrix::Matrix<F>& trace_;
    size_t row_index_;
    std::vector<F> current_row_;
    std::vector<F> next_row_;
    std::vector<F> preprocessed_current_;
    std::vector<F> preprocessed_next_;
    MainWindow preprocessed_window_;
    std::vector<ConstraintViolation>* failures_;
    size_t constraint_index_ = 0;
};

template<typename F, typename AIR>
std::vector<ConstraintViolation> check_constraints(const AIR& air, const p3_matrix::Matrix<F>& trace) {
    std::vector<ConstraintViolation> failures;
    if (trace.height() == 0) {
        return failures;
    }
    if (trace.width() != air.width()) {
        throw std::invalid_argument("trace width must match air width");
    }

    for (size_t row = 0; row < trace.height(); ++row) {
        DebugConstraintBuilder<F> builder(trace, row, &failures);
        air.eval(builder);
    }
    return failures;
}

template<typename F, typename AIR>
bool constraints_hold(const AIR& air, const p3_matrix::Matrix<F>& trace, const p3_matrix::Matrix<F>* preprocessed_trace = nullptr) {
    return check_constraints<F>(air, trace, preprocessed_trace).empty();
}

} // namespace p3_air

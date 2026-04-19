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
        std::vector<ConstraintViolation>* failures,
        const p3_matrix::Matrix<F>* preprocessed_trace = nullptr)
        : trace_(trace), preprocessed_trace_(preprocessed_trace), failures_(failures) {}

    /// Rebind row-local views and reset the per-row constraint counter. Fallback row buffers are
    /// reused across calls so checking every row does not repeatedly allocate internal vectors.
    void reset_for_row(size_t row_index) {
        row_index_ = row_index;
        constraint_index_ = 0;

        const size_t h = trace_.height();
        const size_t next_i = (row_index + 1) % h;
        ConstRowView<F> cur = matrix_row_view(trace_, row_index, &main_current_fallback_);
        ConstRowView<F> nxt = matrix_row_view(trace_, next_i, &main_next_fallback_);
        main_window_ = MainWindow(cur, nxt);

        if (preprocessed_trace_ != nullptr) {
            ConstRowView<F> pc = matrix_row_view(*preprocessed_trace_, row_index, &preprocessed_current_fallback_);
            ConstRowView<F> pn = matrix_row_view(*preprocessed_trace_, next_i, &preprocessed_next_fallback_);
            preprocessed_window_ = MainWindow(pc, pn);
        } else {
            preprocessed_window_ = MainWindow();
        }
    }

    MainWindow main() const override {
        return main_window_;
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
    static ConstRowView<F> matrix_row_view(
        const p3_matrix::Matrix<F>& m,
        size_t r,
        std::vector<F>* fallback) {
        const F* p = m.row_ptr(r);
        if (p != nullptr) {
            return ConstRowView<F>(p, m.width());
        }
        fallback->resize(m.width());
        for (size_t c = 0; c < m.width(); ++c) {
            (*fallback)[c] = m.get_unchecked(r, c);
        }
        return ConstRowView<F>(fallback->data(), fallback->size());
    }

    const p3_matrix::Matrix<F>& trace_;
    const p3_matrix::Matrix<F>* preprocessed_trace_;
    std::vector<ConstraintViolation>* failures_;
    size_t row_index_ = 0;
    std::vector<F> main_current_fallback_;
    std::vector<F> main_next_fallback_;
    std::vector<F> preprocessed_current_fallback_;
    std::vector<F> preprocessed_next_fallback_;
    MainWindow main_window_;
    MainWindow preprocessed_window_;
    size_t constraint_index_ = 0;
};

template<typename F, typename AIR>
std::vector<ConstraintViolation> check_constraints(
    const AIR& air,
    const p3_matrix::Matrix<F>& trace,
    const p3_matrix::Matrix<F>* preprocessed_trace = nullptr) {
    std::vector<ConstraintViolation> failures;
    if (trace.height() == 0) {
        return failures;
    }
    if (trace.width() != air.width()) {
        throw std::invalid_argument("trace width must match air width");
    }

    DebugConstraintBuilder<F> builder(trace, &failures, preprocessed_trace);
    for (size_t row = 0; row < trace.height(); ++row) {
        builder.reset_for_row(row);
        air.eval(builder);
    }
    return failures;
}

template<typename F, typename AIR>
bool constraints_hold(const AIR& air, const p3_matrix::Matrix<F>& trace, const p3_matrix::Matrix<F>* preprocessed_trace = nullptr) {
    return check_constraints<F>(air, trace, preprocessed_trace).empty();
}

} // namespace p3_air

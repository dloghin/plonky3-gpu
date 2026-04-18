#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace p3_air {

template<typename F, typename Expr = F, typename Var = Expr>
class AirBuilder;

template<typename F, typename Expr = F, typename Var = Expr>
class FilteredAirBuilder;

template<typename T>
class RowWindow {
public:
    RowWindow() = default;

    RowWindow(const std::vector<T>* current, const std::vector<T>* next)
        : current_(current), next_(next) {}

    const std::vector<T>& current_slice() const {
        if (current_ == nullptr) {
            throw std::logic_error("current row is not available");
        }
        return *current_;
    }

    const std::vector<T>& next_slice() const {
        if (next_ == nullptr) {
            throw std::logic_error("next row is not available");
        }
        return *next_;
    }

    const T& current(size_t column) const {
        const auto& row = current_slice();
        if (column >= row.size()) {
            throw std::out_of_range("current row column out of bounds");
        }
        return row[column];
    }

    const T& next(size_t column) const {
        const auto& row = next_slice();
        if (column >= row.size()) {
            throw std::out_of_range("next row column out of bounds");
        }
        return row[column];
    }

private:
    const std::vector<T>* current_ = nullptr;
    const std::vector<T>* next_ = nullptr;
};

template<typename AB>
class Air {
public:
    virtual ~Air() = default;
    virtual size_t width() const = 0;
    virtual void eval(AB& builder) const = 0;
};

template<typename F, typename Expr, typename Var>
class AirBuilder {
public:
    using Field = F;
    using ExprType = Expr;
    using VarType = Var;
    using PreprocessedWindow = RowWindow<Var>;
    using MainWindow = RowWindow<Var>;

    virtual ~AirBuilder() = default;

    virtual MainWindow main() const = 0;
    virtual const PreprocessedWindow& preprocessed() const = 0;
    virtual Expr is_first_row() const = 0;
    virtual Expr is_last_row() const = 0;
    virtual Expr is_transition() const = 0;
    virtual void assert_zero(const Expr& expression) = 0;

    void assert_one(const Expr& expression) {
        assert_eq(expression, Expr(F{1}));
    }

    void assert_eq(const Expr& lhs, const Expr& rhs) {
        assert_zero(lhs - rhs);
    }

    FilteredAirBuilder<F, Expr, Var> when(const Expr& condition) {
        return FilteredAirBuilder<F, Expr, Var>(*this, condition);
    }

    FilteredAirBuilder<F, Expr, Var> when_first_row() {
        return when(is_first_row());
    }

    FilteredAirBuilder<F, Expr, Var> when_last_row() {
        return when(is_last_row());
    }

    FilteredAirBuilder<F, Expr, Var> when_transition() {
        return when(is_transition());
    }
};

template<typename F, typename Expr, typename Var>
class FilteredAirBuilder : public AirBuilder<F, Expr, Var> {
public:
    using PreprocessedWindow = RowWindow<Var>;
    using MainWindow = RowWindow<Var>;

    FilteredAirBuilder(AirBuilder<F, Expr, Var>& inner, const Expr& condition)
        : inner_(inner), condition_(condition) {}

    MainWindow main() const override {
        return inner_.main();
    }

    const PreprocessedWindow& preprocessed() const override {
        return inner_.preprocessed();
    }

    Expr is_first_row() const override {
        return inner_.is_first_row();
    }

    Expr is_last_row() const override {
        return inner_.is_last_row();
    }

    Expr is_transition() const override {
        return inner_.is_transition();
    }

    void assert_zero(const Expr& expression) override {
        inner_.assert_zero(condition_ * expression);
    }

private:
    AirBuilder<F, Expr, Var>& inner_;
    Expr condition_;
};

} // namespace p3_air

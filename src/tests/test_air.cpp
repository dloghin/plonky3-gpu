#include <gtest/gtest.h>

#include "air.hpp"
#include "baby_bear.hpp"
#include "check_constraints.hpp"
#include "dense_matrix.hpp"
#include "symbolic_expression.hpp"
#include "virtual_column.hpp"

using p3_air::Air;
using p3_air::AirBuilder;
using p3_air::ConstRowView;
using p3_air::ConstraintViolation;
using p3_air::DebugConstraintBuilder;
using p3_air::SymbolicExpression;
using p3_air::SymbolicVariable;
using p3_air::VirtualColumn;
using p3_field::BabyBear;
using p3_matrix::RowMajorMatrix;

namespace {

class FibonacciAir : public Air<DebugConstraintBuilder<BabyBear>> {
public:
    size_t width() const override {
        return 2;
    }

    void eval(DebugConstraintBuilder<BabyBear>& builder) const override {
        auto main = builder.main();

        auto transition = builder.when_transition();
        transition.assert_eq(main.next(0), main.current(1));
        transition.assert_eq(main.next(1), main.current(0) + main.current(1));

        auto first = builder.when_first_row();
        first.assert_eq(main.current(0), BabyBear(1));
        first.assert_eq(main.current(1), BabyBear(1));
    }
};

class DoubleColumnAir : public Air<DebugConstraintBuilder<BabyBear>> {
public:
    size_t width() const override {
        return 2;
    }

    void eval(DebugConstraintBuilder<BabyBear>& builder) const override {
        auto main = builder.main();
        builder.assert_eq(main.current(1), main.current(0) + main.current(0));
    }
};

class SymbolicCollectingBuilder
    : public AirBuilder<BabyBear, SymbolicExpression<BabyBear>, SymbolicExpression<BabyBear>> {
public:
    using Expr = SymbolicExpression<BabyBear>;
    using Window = typename AirBuilder<BabyBear, Expr, Expr>::MainWindow;

    SymbolicCollectingBuilder() {
        current_vars_ = {Expr::variable(0, 0), Expr::variable(0, 1)};
        next_vars_ = {Expr::variable(1, 0), Expr::variable(1, 1)};
        main_window_ = Window(ConstRowView<Expr>(current_vars_), ConstRowView<Expr>(next_vars_));
        preprocessed_window_ = Window(ConstRowView<Expr>(empty_vars_), ConstRowView<Expr>(empty_vars_));
    }

    Window main() const override {
        return main_window_;
    }

    const Window& preprocessed() const override {
        return preprocessed_window_;
    }

    Expr is_first_row() const override {
        return Expr(BabyBear(1));
    }

    Expr is_last_row() const override {
        return Expr(BabyBear(0));
    }

    Expr is_transition() const override {
        return Expr(BabyBear(1));
    }

    void assert_zero(const Expr& expression) override {
        constraints_.push_back(expression);
    }

    const std::vector<Expr>& constraints() const {
        return constraints_;
    }

private:
    std::vector<Expr> current_vars_;
    std::vector<Expr> next_vars_;
    std::vector<Expr> empty_vars_;
    Window main_window_;
    Window preprocessed_window_;
    std::vector<Expr> constraints_;
};

class GenericExpressionAir : public Air<SymbolicCollectingBuilder> {
public:
    size_t width() const override {
        return 2;
    }

    void eval(SymbolicCollectingBuilder& builder) const override {
        auto main = builder.main();
        auto transition = builder.when_transition();
        transition.assert_eq(main.next(0), main.current(1));
        transition.assert_eq(main.next(1), main.current(0) + main.current(1));
    }
};

TEST(AirFrameworkTest, FibonacciAirAcceptsValidTrace) {
    FibonacciAir air;
    RowMajorMatrix<BabyBear> trace({
        BabyBear(1), BabyBear(1),
        BabyBear(1), BabyBear(2),
        BabyBear(2), BabyBear(3),
        BabyBear(3), BabyBear(5),
    }, 2);

    const auto failures = p3_air::check_constraints<BabyBear>(air, trace);
    EXPECT_TRUE(failures.empty());
}

TEST(AirFrameworkTest, FibonacciAirRejectsInvalidTrace) {
    FibonacciAir air;
    RowMajorMatrix<BabyBear> trace({
        BabyBear(1), BabyBear(1),
        BabyBear(1), BabyBear(2),
        BabyBear(2), BabyBear(9), // invalid step
        BabyBear(3), BabyBear(5),
    }, 2);

    const auto failures = p3_air::check_constraints<BabyBear>(air, trace);
    EXPECT_FALSE(failures.empty());
    EXPECT_EQ(failures.front().row, 1u);
}

TEST(AirFrameworkTest, SecondAirUsesGenericBuilderInterface) {
    DoubleColumnAir air;
    RowMajorMatrix<BabyBear> good({
        BabyBear(1), BabyBear(2),
        BabyBear(3), BabyBear(6),
    }, 2);
    RowMajorMatrix<BabyBear> bad({
        BabyBear(1), BabyBear(3),
        BabyBear(3), BabyBear(6),
    }, 2);

    EXPECT_TRUE(p3_air::constraints_hold<BabyBear>(air, good));
    EXPECT_FALSE(p3_air::constraints_hold<BabyBear>(air, bad));
}

TEST(AirFrameworkTest, SymbolicExpressionSupportsOperatorsAndEvaluation) {
    using Expr = SymbolicExpression<BabyBear>;
    const Expr x = Expr::variable(0, 0);
    const Expr y = Expr::variable(1, 1);
    const Expr expr = (x + y) * x - (-y);

    const BabyBear value = expr.evaluate([](const SymbolicVariable& var) {
        if (var.row_offset == 0 && var.column == 0) {
            return BabyBear(3);
        }
        return BabyBear(4);
    });

    EXPECT_EQ(value, (BabyBear(3) + BabyBear(4)) * BabyBear(3) + BabyBear(4));
}

TEST(AirFrameworkTest, VirtualColumnComputesLinearCombination) {
    VirtualColumn<BabyBear> column(BabyBear(3));
    column.add_main_term(0, BabyBear(2));
    column.add_preprocessed_term(1, BabyBear(5));

    const std::vector<BabyBear> preprocessed = {BabyBear(13), BabyBear(17)};
    const std::vector<BabyBear> main = {BabyBear(2), BabyBear(11)};
    const BabyBear value = column.evaluate(preprocessed, main);
    EXPECT_EQ(value, BabyBear(3) + BabyBear(2) * BabyBear(2) + BabyBear(5) * BabyBear(17));
}

TEST(AirFrameworkTest, GenericExpressionBuilderCollectsSymbolicConstraints) {
    GenericExpressionAir air;
    SymbolicCollectingBuilder builder;
    air.eval(builder);

    ASSERT_EQ(builder.constraints().size(), 2u);
    const auto assignment = [](const SymbolicVariable& var) {
        if (var.row_offset == 0 && var.column == 0) return BabyBear(2);
        if (var.row_offset == 0 && var.column == 1) return BabyBear(3);
        if (var.row_offset == 1 && var.column == 0) return BabyBear(3);
        return BabyBear(5);
    };
    EXPECT_EQ(builder.constraints()[0].evaluate(assignment), BabyBear(0));
    EXPECT_EQ(builder.constraints()[1].evaluate(assignment), BabyBear(0));
}

TEST(AirFrameworkTest, CheckConstraintsRejectsWidthMismatch) {
    FibonacciAir air;
    RowMajorMatrix<BabyBear> bad_width_trace({
        BabyBear(1), BabyBear(1), BabyBear(1),
        BabyBear(1), BabyBear(2), BabyBear(3),
    }, 3);

    EXPECT_THROW((p3_air::check_constraints<BabyBear>(air, bad_width_trace)), std::invalid_argument);
}

} // namespace

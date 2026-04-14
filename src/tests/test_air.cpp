#include <gtest/gtest.h>

#include "air.hpp"
#include "baby_bear.hpp"
#include "check_constraints.hpp"
#include "dense_matrix.hpp"
#include "symbolic_expression.hpp"
#include "virtual_column.hpp"

using p3_air::Air;
using p3_air::AirBuilder;
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
    RowMajorMatrix<BabyBear> trace({
        BabyBear(2), BabyBear(7),
        BabyBear(5), BabyBear(11),
    }, 2);

    std::vector<ConstraintViolation> failures;
    DebugConstraintBuilder<BabyBear> builder(trace, 0, &failures);
    const auto window = builder.main();

    VirtualColumn<BabyBear> column(BabyBear(3));
    column.add_term(0, 0, BabyBear(2));
    column.add_term(1, 1, BabyBear(1));

    const BabyBear value = column.evaluate(window);
    EXPECT_EQ(value, BabyBear(3) + BabyBear(2) * BabyBear(2) + BabyBear(11));
}

} // namespace

#include <gtest/gtest.h>

#include "baby_bear.hpp"
#include "dense_matrix.hpp"
#include "extension_field.hpp"
#include "matrix_extension.hpp"
#include "matrix_views.hpp"

using namespace p3_matrix;

namespace {

RowMajorMatrix<int> sample_matrix_5x3() {
    return RowMajorMatrix<int>(
        {
            10, 11, 12,
            20, 21, 22,
            30, 31, 32,
            40, 41, 42,
            50, 51, 52
        },
        3
    );
}

} // namespace

TEST(MatrixViewsTest, VerticallyStridedMatrixViewSelectsEveryKthRow) {
    auto matrix = sample_matrix_5x3();
    VerticallyStridedMatrixView<int> view(matrix, 2, 1);

    EXPECT_EQ(view.width(), 3);
    EXPECT_EQ(view.height(), 2);
    EXPECT_EQ(view.get(0, 0), 20);
    EXPECT_EQ(view.get(1, 2), 42);
    EXPECT_THROW(view.get(2, 0), std::out_of_range);
}

TEST(MatrixViewsTest, VerticallyStridedMatrixViewHandlesLargeStride) {
    auto matrix = sample_matrix_5x3();
    VerticallyStridedMatrixView<int> view(matrix, 10, 4);
    EXPECT_EQ(view.height(), 1);
    EXPECT_EQ(view.get(0, 1), 51);

    VerticallyStridedMatrixView<int> empty_view(matrix, 10, 6);
    EXPECT_EQ(empty_view.height(), 0);
    EXPECT_THROW(empty_view.get(0, 0), std::out_of_range);
}

TEST(MatrixViewsTest, BitReversedMatrixViewMapsRowsByBitReverse) {
    RowMajorMatrix<int> matrix({0, 1, 2, 3, 4, 5, 6, 7}, 1);
    BitReversedMatrixView<int> view(matrix);

    const std::vector<int> expected = {0, 4, 2, 6, 1, 5, 3, 7};
    for (size_t r = 0; r < expected.size(); ++r) {
        EXPECT_EQ(view.get(r, 0), expected[r]);
    }
}

TEST(MatrixViewsTest, VerticallyStackedMatricesConcatenatesInputs) {
    RowMajorMatrix<int> top({1, 2, 3, 4}, 2);      // 2x2
    RowMajorMatrix<int> mid({5, 6}, 2);            // 1x2
    RowMajorMatrix<int> bottom({7, 8, 9, 10}, 2);  // 2x2

    VerticallyStackedMatrices<int> stacked({
        std::cref(static_cast<const Matrix<int>&>(top)),
        std::cref(static_cast<const Matrix<int>&>(mid)),
        std::cref(static_cast<const Matrix<int>&>(bottom))
    });

    EXPECT_EQ(stacked.width(), 2);
    EXPECT_EQ(stacked.height(), 5);
    EXPECT_EQ(stacked.get(0, 1), 2);
    EXPECT_EQ(stacked.get(2, 0), 5);
    EXPECT_EQ(stacked.get(4, 1), 10);
}

TEST(MatrixViewsTest, HorizontallyTruncatedViewLimitsColumns) {
    auto matrix = sample_matrix_5x3();
    HorizontallyTruncatedView<int> view(matrix, 2);

    EXPECT_EQ(view.width(), 2);
    EXPECT_EQ(view.height(), 5);
    EXPECT_EQ(view.get(3, 1), 41);
    EXPECT_THROW(view.get(0, 2), std::out_of_range);
}

TEST(MatrixViewsTest, RowIndexMappedViewSupportsArbitraryMapping) {
    auto matrix = sample_matrix_5x3();
    auto reverse_rows = [&matrix](size_t r) { return matrix.height() - 1 - r; };
    RowIndexMappedView<int, decltype(reverse_rows)> view(matrix, matrix.height(), reverse_rows);

    EXPECT_EQ(view.width(), 3);
    EXPECT_EQ(view.height(), 5);
    EXPECT_EQ(view.get(0, 0), 50);
    EXPECT_EQ(view.get(4, 2), 12);
}

TEST(MatrixViewsTest, ComposedViewsBitReverseAfterStriding) {
    RowMajorMatrix<int> matrix({
        0, 1,
        10, 11,
        20, 21,
        30, 31,
        40, 41,
        50, 51,
        60, 61,
        70, 71
    }, 2);

    VerticallyStridedMatrixView<int> strided(matrix, 2, 0); // rows 0,2,4,6
    BitReversedMatrixView<int> bitrev(strided);             // order 0,2,1,3

    EXPECT_EQ(bitrev.height(), 4);
    EXPECT_EQ(bitrev.get(0, 0), 0);
    EXPECT_EQ(bitrev.get(1, 0), 40);
    EXPECT_EQ(bitrev.get(2, 0), 20);
    EXPECT_EQ(bitrev.get(3, 0), 60);
}

TEST(MatrixViewsTest, ExtensionFlattenAndUnflattenRoundTrip) {
    using Base = p3_field::BabyBear;
    using Ext = p3_field::BabyBear4;

    std::vector<Ext> ext_values = {
        Ext({Base(1), Base(2), Base(3), Base(4)}),
        Ext({Base(5), Base(6), Base(7), Base(8)}),
        Ext({Base(9), Base(10), Base(11), Base(12)}),
        Ext({Base(13), Base(14), Base(15), Base(16)})
    };
    RowMajorMatrix<Ext> ext_matrix(ext_values, 2); // 2x2 in extension field

    auto flattened = flatten_to_base<Base, 4, 11>(ext_matrix);
    EXPECT_EQ(flattened.height(), 2);
    EXPECT_EQ(flattened.width(), 8);
    EXPECT_EQ(flattened.get(0, 0), Base(1));
    EXPECT_EQ(flattened.get(0, 7), Base(8));
    EXPECT_EQ(flattened.get(1, 3), Base(12));

    auto roundtrip = unflatten_from_base<Base, 4, 11>(flattened);
    EXPECT_EQ(roundtrip.width(), ext_matrix.width());
    EXPECT_EQ(roundtrip.height(), ext_matrix.height());
    for (size_t r = 0; r < ext_matrix.height(); ++r) {
        for (size_t c = 0; c < ext_matrix.width(); ++c) {
            EXPECT_EQ(roundtrip.get(r, c), ext_matrix.get(r, c));
        }
    }
}

TEST(MatrixViewsTest, FlattenedViewRowPtrZeroCopy) {
    using Base = p3_field::BabyBear;
    using Ext = p3_field::BabyBear4;

    std::vector<Ext> ext_values = {
        Ext({Base(1), Base(2), Base(3), Base(4)}),
        Ext({Base(5), Base(6), Base(7), Base(8)}),
        Ext({Base(9), Base(10), Base(11), Base(12)}),
        Ext({Base(13), Base(14), Base(15), Base(16)})
    };
    RowMajorMatrix<Ext> ext_matrix(ext_values, 2);

    auto flattened = flatten_to_base<Base, 4, 11>(ext_matrix);

    const Base* row0 = flattened.row_ptr(0);
    ASSERT_NE(row0, nullptr);
    for (size_t c = 0; c < flattened.width(); ++c) {
        EXPECT_EQ(row0[c], flattened.get(0, c));
    }

    const Base* row1 = flattened.row_ptr(1);
    ASSERT_NE(row1, nullptr);
    EXPECT_EQ(row1[0], Base(9));
    EXPECT_EQ(row1[7], Base(16));

    // row_ptr must alias the inner matrix's storage — no copy
    EXPECT_EQ(row0, reinterpret_cast<const Base*>(ext_matrix.row_ptr(0)));
}

TEST(MatrixViewsTest, UnflattenedViewRowPtrZeroCopy) {
    using Base = p3_field::BabyBear;
    using Ext = p3_field::BabyBear4;

    std::vector<Base> base_values = {
        Base(1), Base(2), Base(3), Base(4), Base(5), Base(6), Base(7), Base(8),
        Base(9), Base(10), Base(11), Base(12), Base(13), Base(14), Base(15), Base(16)
    };
    RowMajorMatrix<Base> base_matrix(base_values, 8);

    auto unflattened = unflatten_from_base<Base, 4, 11>(base_matrix);
    EXPECT_EQ(unflattened.width(), 2);
    EXPECT_EQ(unflattened.height(), 2);

    const Ext* row0 = unflattened.row_ptr(0);
    ASSERT_NE(row0, nullptr);
    EXPECT_EQ(row0[0], Ext({Base(1), Base(2), Base(3), Base(4)}));
    EXPECT_EQ(row0[1], Ext({Base(5), Base(6), Base(7), Base(8)}));

    // row_ptr must alias the inner matrix's storage — no copy
    EXPECT_EQ(reinterpret_cast<const Base*>(row0), base_matrix.row_ptr(0));
}

TEST(MatrixViewsTest, FlattenedViewToRowMajorMatrix) {
    using Base = p3_field::BabyBear;
    using Ext = p3_field::BabyBear4;

    std::vector<Ext> ext_values = {
        Ext({Base(10), Base(20), Base(30), Base(40)}),
        Ext({Base(50), Base(60), Base(70), Base(80)})
    };
    RowMajorMatrix<Ext> ext_matrix(ext_values, 2);

    auto view = flatten_to_base<Base, 4, 11>(ext_matrix);
    auto materialized = view.to_row_major_matrix();

    EXPECT_EQ(materialized.width(), 8);
    EXPECT_EQ(materialized.height(), 1);
    for (size_t c = 0; c < 8; ++c) {
        EXPECT_EQ(materialized.get(0, c), view.get(0, c));
    }
}

TEST(MatrixViewsTest, UnflattenedViewToRowMajorMatrix) {
    using Base = p3_field::BabyBear;

    std::vector<Base> base_values = {
        Base(1), Base(2), Base(3), Base(4), Base(5), Base(6), Base(7), Base(8)
    };
    RowMajorMatrix<Base> base_matrix(base_values, 8);

    auto view = unflatten_from_base<Base, 4, 11>(base_matrix);
    auto materialized = view.to_row_major_matrix();

    EXPECT_EQ(materialized.width(), 2);
    EXPECT_EQ(materialized.height(), 1);
    for (size_t c = 0; c < 2; ++c) {
        EXPECT_EQ(materialized.get(0, c), view.get(0, c));
    }
}

TEST(MatrixViewsTest, UnflattenWidthNotDivisibleThrows) {
    using Base = p3_field::BabyBear;

    RowMajorMatrix<Base> bad_matrix({Base(1), Base(2), Base(3)}, 3);
    EXPECT_THROW((unflatten_from_base<Base, 4, 11>(bad_matrix)), std::invalid_argument);
}


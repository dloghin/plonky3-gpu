#include <gtest/gtest.h>
#include "matrix.hpp"
#include "dense_matrix.hpp"

using namespace p3_matrix;

TEST(MatrixTest, Dimensions) {
    Dimensions dims(3, 5);
    EXPECT_EQ(dims.width, 3);
    EXPECT_EQ(dims.height, 5);

    Dimensions dims2(3, 5);
    EXPECT_EQ(dims, dims2);

    Dimensions dims3(4, 5);
    EXPECT_NE(dims, dims3);
}

TEST(MatrixTest, BasicProperties) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> mat(vals, 3);  // 2x3 matrix

    EXPECT_EQ(mat.width(), 3);
    EXPECT_EQ(mat.height(), 2);
    EXPECT_EQ(mat.size(), 6);
    EXPECT_FALSE(mat.empty());
}

TEST(MatrixTest, EmptyMatrix) {
    RowMajorMatrix<int> mat;
    EXPECT_TRUE(mat.empty());
    EXPECT_EQ(mat.size(), 0);
}

TEST(MatrixTest, GetElement) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> mat(vals, 3);

    EXPECT_EQ(mat.get(0, 0), 1);
    EXPECT_EQ(mat.get(0, 1), 2);
    EXPECT_EQ(mat.get(0, 2), 3);
    EXPECT_EQ(mat.get(1, 0), 4);
    EXPECT_EQ(mat.get(1, 1), 5);
    EXPECT_EQ(mat.get(1, 2), 6);

    EXPECT_THROW(mat.get(2, 0), std::out_of_range);
    EXPECT_THROW(mat.get(0, 3), std::out_of_range);
}

TEST(MatrixTest, GetUnchecked) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> mat(vals, 3);

    EXPECT_EQ(mat.get_unchecked(0, 0), 1);
    EXPECT_EQ(mat.get_unchecked(1, 2), 6);
}

TEST(MatrixTest, RowAccess) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    RowMajorMatrix<int> mat(vals, 3);

    auto row0 = mat.row(0);
    EXPECT_EQ(row0, std::vector<int>({1, 2, 3}));

    auto row1 = mat.row(1);
    EXPECT_EQ(row1, std::vector<int>({4, 5, 6}));

    auto row2 = mat.row(2);
    EXPECT_EQ(row2, std::vector<int>({7, 8, 9}));

    EXPECT_THROW(mat.row(3), std::out_of_range);
}

TEST(MatrixTest, FirstLastRow) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> mat(vals, 3);

    auto first = mat.first_row();
    EXPECT_EQ(first, std::vector<int>({1, 2, 3}));

    auto last = mat.last_row();
    EXPECT_EQ(last, std::vector<int>({4, 5, 6}));
}

TEST(MatrixTest, AllRows) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> mat(vals, 2);

    auto all_rows = mat.rows();
    ASSERT_EQ(all_rows.size(), 3);
    EXPECT_EQ(all_rows[0], std::vector<int>({1, 2}));
    EXPECT_EQ(all_rows[1], std::vector<int>({3, 4}));
    EXPECT_EQ(all_rows[2], std::vector<int>({5, 6}));
}

TEST(MatrixTest, RowPtr) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> mat(vals, 3);

    const int* row0 = mat.row_ptr(0);
    ASSERT_NE(row0, nullptr);
    EXPECT_EQ(row0[0], 1);
    EXPECT_EQ(row0[1], 2);
    EXPECT_EQ(row0[2], 3);

    const int* row1 = mat.row_ptr(1);
    ASSERT_NE(row1, nullptr);
    EXPECT_EQ(row1[0], 4);
    EXPECT_EQ(row1[1], 5);
    EXPECT_EQ(row1[2], 6);
}


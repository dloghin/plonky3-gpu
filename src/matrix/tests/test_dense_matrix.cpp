#include <gtest/gtest.h>
#include "dense_matrix.hpp"
#include <random>

using namespace p3_matrix;

TEST(DenseMatrixTest, Construction) {
    // Default construction
    RowMajorMatrix<int> mat1;
    EXPECT_EQ(mat1.width(), 0);
    EXPECT_EQ(mat1.height(), 0);

    // Construction with vector
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> mat2(vals, 3);
    EXPECT_EQ(mat2.width(), 3);
    EXPECT_EQ(mat2.height(), 2);

    // Construction with size and default value
    RowMajorMatrix<int> mat3(3, 4, 7);
    EXPECT_EQ(mat3.width(), 4);
    EXPECT_EQ(mat3.height(), 3);
    EXPECT_EQ(mat3.get(0, 0), 7);
    EXPECT_EQ(mat3.get(2, 3), 7);
}

TEST(DenseMatrixTest, NewRowAndCol) {
    // Single row
    auto row_mat = RowMajorMatrix<int>::new_row({1, 2, 3});
    EXPECT_EQ(row_mat.width(), 3);
    EXPECT_EQ(row_mat.height(), 1);
    EXPECT_EQ(row_mat.get(0, 0), 1);
    EXPECT_EQ(row_mat.get(0, 2), 3);

    // Single column
    auto col_mat = RowMajorMatrix<int>::new_col({1, 2, 3});
    EXPECT_EQ(col_mat.width(), 1);
    EXPECT_EQ(col_mat.height(), 3);
    EXPECT_EQ(col_mat.get(0, 0), 1);
    EXPECT_EQ(col_mat.get(2, 0), 3);
}

TEST(DenseMatrixTest, SetElement) {
    RowMajorMatrix<int> mat(3, 3, 0);

    mat.set(0, 0, 1);
    mat.set(1, 1, 5);
    mat.set(2, 2, 9);

    EXPECT_EQ(mat.get(0, 0), 1);
    EXPECT_EQ(mat.get(1, 1), 5);
    EXPECT_EQ(mat.get(2, 2), 9);

    EXPECT_THROW(mat.set(3, 0, 10), std::out_of_range);
}

TEST(DenseMatrixTest, CopyFrom) {
    RowMajorMatrix<int> mat1(2, 3, 0);
    RowMajorMatrix<int> mat2(std::vector<int>{1, 2, 3, 4, 5, 6}, 3);

    mat1.copy_from(mat2);
    EXPECT_EQ(mat1.values, mat2.values);

    RowMajorMatrix<int> mat3(3, 3, 0);
    EXPECT_THROW(mat1.copy_from(mat3), std::invalid_argument);
}

TEST(DenseMatrixTest, SplitRows) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    RowMajorMatrix<int> mat(vals, 3);

    auto [top, bottom] = mat.split_rows(1);

    EXPECT_EQ(top.height(), 1);
    EXPECT_EQ(top.width(), 3);
    EXPECT_EQ(top.get(0, 0), 1);

    EXPECT_EQ(bottom.height(), 2);
    EXPECT_EQ(bottom.width(), 3);
    EXPECT_EQ(bottom.get(0, 0), 4);
    EXPECT_EQ(bottom.get(1, 2), 9);
}

TEST(DenseMatrixTest, PadToHeight) {
    RowMajorMatrix<int> mat(std::vector<int>{1, 2, 3, 4, 5, 6}, 3);
    EXPECT_EQ(mat.height(), 2);

    mat.pad_to_height(4, 9);
    EXPECT_EQ(mat.height(), 4);
    EXPECT_EQ(mat.get(3, 2), 9);

    // Check original values unchanged
    EXPECT_EQ(mat.get(0, 0), 1);
    EXPECT_EQ(mat.get(1, 2), 6);
}

TEST(DenseMatrixTest, Scale) {
    RowMajorMatrix<int> mat(std::vector<int>{1, 2, 3, 4, 5, 6}, 3);

    mat.scale(2);

    EXPECT_EQ(mat.get(0, 0), 2);
    EXPECT_EQ(mat.get(0, 1), 4);
    EXPECT_EQ(mat.get(0, 2), 6);
    EXPECT_EQ(mat.get(1, 0), 8);
    EXPECT_EQ(mat.get(1, 1), 10);
    EXPECT_EQ(mat.get(1, 2), 12);
}

TEST(DenseMatrixTest, ScaleRow) {
    RowMajorMatrix<int> mat(std::vector<int>{1, 2, 3, 4, 5, 6}, 3);

    mat.scale_row(1, 3);

    // First row unchanged
    EXPECT_EQ(mat.get(0, 0), 1);
    EXPECT_EQ(mat.get(0, 1), 2);
    EXPECT_EQ(mat.get(0, 2), 3);

    // Second row scaled
    EXPECT_EQ(mat.get(1, 0), 12);
    EXPECT_EQ(mat.get(1, 1), 15);
    EXPECT_EQ(mat.get(1, 2), 18);
}

TEST(DenseMatrixTest, Transpose) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> mat(vals, 3);  // 2x3

    auto transposed = mat.transpose();

    EXPECT_EQ(transposed.width(), 2);
    EXPECT_EQ(transposed.height(), 3);

    EXPECT_EQ(transposed.get(0, 0), 1);
    EXPECT_EQ(transposed.get(0, 1), 4);
    EXPECT_EQ(transposed.get(1, 0), 2);
    EXPECT_EQ(transposed.get(1, 1), 5);
    EXPECT_EQ(transposed.get(2, 0), 3);
    EXPECT_EQ(transposed.get(2, 1), 6);
}

TEST(DenseMatrixTest, TransposeSquare) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    RowMajorMatrix<int> mat(vals, 3);

    auto transposed = mat.transpose();

    EXPECT_EQ(transposed.width(), 3);
    EXPECT_EQ(transposed.height(), 3);
    EXPECT_EQ(transposed.values, std::vector<int>({1, 4, 7, 2, 5, 8, 3, 6, 9}));
}

TEST(DenseMatrixTest, TransposeInto) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> mat(vals, 3);  // 2x3

    RowMajorMatrix<int> dest(3, 2, 0);  // 3x2
    mat.transpose_into(dest);

    EXPECT_EQ(dest.get(0, 0), 1);
    EXPECT_EQ(dest.get(0, 1), 4);
    EXPECT_EQ(dest.get(1, 0), 2);
    EXPECT_EQ(dest.get(1, 1), 5);
    EXPECT_EQ(dest.get(2, 0), 3);
    EXPECT_EQ(dest.get(2, 1), 6);
}

TEST(DenseMatrixTest, TransposeIntoLargeMatrix) {
    // Test from Rust: 512x256 matrix
    const size_t HEIGHT = 512;
    const size_t WIDTH = 256;

    std::vector<int> vals(HEIGHT * WIDTH);
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<int>(i + 1);
    }

    RowMajorMatrix<int> mat(vals, WIDTH);
    RowMajorMatrix<int> transposed(WIDTH, HEIGHT, 0);

    mat.transpose_into(transposed);

    EXPECT_EQ(transposed.width(), HEIGHT);
    EXPECT_EQ(transposed.height(), WIDTH);

    // Verify correctness
    for (size_t r = 0; r < HEIGHT; ++r) {
        for (size_t c = 0; c < WIDTH; ++c) {
            EXPECT_EQ(mat.get_unchecked(r, c), transposed.get_unchecked(c, r));
        }
    }
}

TEST(DenseMatrixTest, SwapRows) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    RowMajorMatrix<int> mat(vals, 3);

    mat.swap_rows(0, 2);

    EXPECT_EQ(mat.get(0, 0), 7);
    EXPECT_EQ(mat.get(0, 1), 8);
    EXPECT_EQ(mat.get(0, 2), 9);
    EXPECT_EQ(mat.get(2, 0), 1);
    EXPECT_EQ(mat.get(2, 1), 2);
    EXPECT_EQ(mat.get(2, 2), 3);

    // Middle row unchanged
    EXPECT_EQ(mat.get(1, 0), 4);
    EXPECT_EQ(mat.get(1, 1), 5);
    EXPECT_EQ(mat.get(1, 2), 6);
}

TEST(DenseMatrixTest, Rand) {
    std::mt19937 rng(42);
    auto mat = RowMajorMatrix<int>::rand(rng, 10, 5);

    EXPECT_EQ(mat.width(), 5);
    EXPECT_EQ(mat.height(), 10);
    EXPECT_EQ(mat.size(), 50);
}

TEST(DenseMatrixTest, Equality) {
    RowMajorMatrix<int> mat1(std::vector<int>{1, 2, 3, 4}, 2);
    RowMajorMatrix<int> mat2(std::vector<int>{1, 2, 3, 4}, 2);
    RowMajorMatrix<int> mat3(std::vector<int>{1, 2, 3, 5}, 2);

    EXPECT_EQ(mat1, mat2);
    EXPECT_NE(mat1, mat3);
}

TEST(DenseMatrixTest, RowMut) {
    RowMajorMatrix<int> mat(std::vector<int>{1, 2, 3, 4, 5, 6}, 3);

    int* row = mat.row_mut(1);
    row[0] = 10;
    row[1] = 20;
    row[2] = 30;

    EXPECT_EQ(mat.get(1, 0), 10);
    EXPECT_EQ(mat.get(1, 1), 20);
    EXPECT_EQ(mat.get(1, 2), 30);

    // First row unchanged
    EXPECT_EQ(mat.get(0, 0), 1);
}


#include <gtest/gtest.h>

#include "matrix_cuda.hpp"

#include <stdexcept>
#include <vector>

using p3_matrix::CudaMatrix;
using p3_matrix::RowMajorMatrix;
using p3_matrix::columnwise_dot_product_cuda;
using p3_matrix::matrix_multiply_cuda;
using p3_matrix::matrix_vector_mul_cuda;

TEST(MatrixCuda, TransposeMatchesCpuShapeAndValues) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> host(vals, 3); // 2x3

    CudaMatrix<int> dmat(host);
    auto dtran = dmat.transpose();

    RowMajorMatrix<int> got;
    dtran.copy_to_host(got);

    EXPECT_EQ(got.height(), 3u);
    EXPECT_EQ(got.width(), 2u);
    EXPECT_EQ(got.get(0, 0), 1);
    EXPECT_EQ(got.get(1, 0), 2);
    EXPECT_EQ(got.get(2, 1), 6);
}

TEST(MatrixCuda, MatrixVectorMulMatchesExpected) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> host(vals, 3); // [[1,2,3],[4,5,6]]
    CudaMatrix<int> dmat(host);
    const std::vector<int> v = {1, 2, 3};

    const auto got = matrix_vector_mul_cuda(dmat, v);
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0], 14); // 1+4+9
    EXPECT_EQ(got[1], 32); // 4+10+18
}

TEST(MatrixCuda, ColumnwiseDotProductMatchesExpected) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> host(vals, 3); // 2x3
    CudaMatrix<int> dmat(host);
    const std::vector<int> v = {1, 2};

    const auto got = columnwise_dot_product_cuda(dmat, v);
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0], 9);  // 1*1 + 2*4
    EXPECT_EQ(got[1], 12); // 1*2 + 2*5
    EXPECT_EQ(got[2], 15); // 1*3 + 2*6
}

TEST(MatrixCuda, ScaleAndScaleRowMatchExpected) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> host(vals, 3); // 2x3
    CudaMatrix<int> dmat(host);

    dmat.scale(2);
    dmat.scale_row(1, 3);

    RowMajorMatrix<int> got;
    dmat.copy_to_host(got);
    EXPECT_EQ(got.get(0, 0), 2);
    EXPECT_EQ(got.get(0, 2), 6);
    EXPECT_EQ(got.get(1, 0), 24);
    EXPECT_EQ(got.get(1, 2), 36);
}

TEST(MatrixCuda, MatrixMultiplyMatchesExpected) {
    RowMajorMatrix<int> a(std::vector<int>{1, 2, 3, 4, 5, 6}, 2); // 3x2
    RowMajorMatrix<int> b(std::vector<int>{7, 8, 9, 10}, 2);      // 2x2
    CudaMatrix<int> da(a);
    CudaMatrix<int> db(b);

    auto dc = matrix_multiply_cuda(da, db);
    RowMajorMatrix<int> got;
    dc.copy_to_host(got);

    ASSERT_EQ(got.height(), 3u);
    ASSERT_EQ(got.width(), 2u);
    EXPECT_EQ(got.get(0, 0), 25);
    EXPECT_EQ(got.get(0, 1), 28);
    EXPECT_EQ(got.get(1, 0), 57);
    EXPECT_EQ(got.get(1, 1), 64);
    EXPECT_EQ(got.get(2, 0), 89);
    EXPECT_EQ(got.get(2, 1), 100);
}

TEST(MatrixCuda, ScaleRowOutOfBoundsThrows) {
    RowMajorMatrix<int> host(std::vector<int>{1, 2, 3, 4}, 2);
    CudaMatrix<int> dmat(host);
    EXPECT_THROW(dmat.scale_row(2, 2), std::out_of_range);
}


#include <gtest/gtest.h>
#include "util.hpp"

using namespace p3_matrix;

TEST(UtilTest, ReverseBitsLen) {
    // 3 bits
    EXPECT_EQ(reverse_bits_len(0b000, 3), 0b000);  // 0 -> 0
    EXPECT_EQ(reverse_bits_len(0b001, 3), 0b100);  // 1 -> 4
    EXPECT_EQ(reverse_bits_len(0b010, 3), 0b010);  // 2 -> 2
    EXPECT_EQ(reverse_bits_len(0b011, 3), 0b110);  // 3 -> 6
    EXPECT_EQ(reverse_bits_len(0b100, 3), 0b001);  // 4 -> 1
    EXPECT_EQ(reverse_bits_len(0b101, 3), 0b101);  // 5 -> 5
    EXPECT_EQ(reverse_bits_len(0b110, 3), 0b011);  // 6 -> 3
    EXPECT_EQ(reverse_bits_len(0b111, 3), 0b111);  // 7 -> 7
}

TEST(UtilTest, Log2Strict) {
    EXPECT_EQ(log2_strict(1), 0);
    EXPECT_EQ(log2_strict(2), 1);
    EXPECT_EQ(log2_strict(4), 2);
    EXPECT_EQ(log2_strict(8), 3);
    EXPECT_EQ(log2_strict(16), 4);
    EXPECT_EQ(log2_strict(1024), 10);

    EXPECT_THROW(log2_strict(0), std::invalid_argument);
    EXPECT_THROW(log2_strict(3), std::invalid_argument);
    EXPECT_THROW(log2_strict(5), std::invalid_argument);
}

TEST(UtilTest, ReverseMatrixIndexBits) {
    std::vector<int> vals = {
        0, 1,   // row 0
        2, 3,   // row 1
        4, 5,   // row 2
        6, 7,   // row 3
        8, 9,   // row 4
        10, 11, // row 5
        12, 13, // row 6
        14, 15  // row 7
    };
    RowMajorMatrix<int> mat(vals, 2);

    reverse_matrix_index_bits(mat);

    // Expected after bit reversal:
    // row 0 (0b000) -> stays at 0b000 (0)
    // row 1 (0b001) -> goes to 0b100 (4)
    // row 2 (0b010) -> stays at 0b010 (2)
    // row 3 (0b011) -> goes to 0b110 (6)
    // row 4 (0b100) -> goes to 0b001 (1)
    // row 5 (0b101) -> stays at 0b101 (5)
    // row 6 (0b110) -> goes to 0b011 (3)
    // row 7 (0b111) -> stays at 0b111 (7)

    std::vector<int> expected = {
        0, 1,   // row 0 (unchanged)
        8, 9,   // row 1 (was row 4)
        4, 5,   // row 2 (unchanged)
        12, 13, // row 3 (was row 6)
        2, 3,   // row 4 (was row 1)
        10, 11, // row 5 (unchanged)
        6, 7,   // row 6 (was row 3)
        14, 15  // row 7 (unchanged)
    };

    EXPECT_EQ(mat.values, expected);
}

TEST(UtilTest, ReverseMatrixIndexBitsHeight1) {
    std::vector<int> vals = {42, 43};
    RowMajorMatrix<int> mat(vals, 2);

    reverse_matrix_index_bits(mat);

    // Single row should be unchanged
    EXPECT_EQ(mat.values, std::vector<int>({42, 43}));
}

TEST(UtilTest, ReverseMatrixIndexBitsNonPowerOfTwo) {
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> mat(vals, 2);  // height = 3 (not power of 2)

    EXPECT_THROW(reverse_matrix_index_bits(mat), std::invalid_argument);
}

TEST(UtilTest, BitReversedZeroPad) {
    std::vector<int> vals = {1, 2, 3, 4};
    RowMajorMatrix<int> mat(vals, 2);  // 2x2

    auto padded = bit_reversed_zero_pad(mat, 1);

    // Should double height with zeros interleaved
    EXPECT_EQ(padded.width(), 2);
    EXPECT_EQ(padded.height(), 4);

    std::vector<int> expected = {
        1, 2,  // original row 0
        0, 0,  // zero padding
        3, 4,  // original row 1
        0, 0   // zero padding
    };

    EXPECT_EQ(padded.values, expected);
}

TEST(UtilTest, BitReversedZeroPadNoChange) {
    std::vector<int> vals = {1, 2, 3, 4};
    RowMajorMatrix<int> mat(vals, 2);

    auto padded = bit_reversed_zero_pad(mat, 0);

    EXPECT_EQ(padded.width(), 2);
    EXPECT_EQ(padded.height(), 2);
    EXPECT_EQ(padded.values, vals);
}

TEST(UtilTest, DotProduct) {
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};

    int result = dot_product(a, b);
    EXPECT_EQ(result, 1*4 + 2*5 + 3*6);  // 4 + 10 + 18 = 32
}

TEST(UtilTest, DotProductMismatch) {
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5};

    EXPECT_THROW(dot_product(a, b), std::invalid_argument);
}

TEST(UtilTest, MatrixVectorMul) {
    std::vector<int> vals = {
        1, 2, 3,
        4, 5, 6
    };
    RowMajorMatrix<int> mat(vals, 3);  // 2x3
    std::vector<int> vec = {1, 2, 3};

    auto result = matrix_vector_mul(mat, vec);

    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 1*1 + 2*2 + 3*3);  // 14
    EXPECT_EQ(result[1], 4*1 + 5*2 + 6*3);  // 32
}

TEST(UtilTest, ColumnwiseDotProduct) {
    std::vector<int> vals = {
        1, 2, 3,
        4, 5, 6
    };
    RowMajorMatrix<int> mat(vals, 3);  // 2x3
    std::vector<int> v = {2, 3};  // scale factors for rows

    auto result = columnwise_dot_product(mat, v);

    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 1*2 + 4*3);  // 2 + 12 = 14
    EXPECT_EQ(result[1], 2*2 + 5*3);  // 4 + 15 = 19
    EXPECT_EQ(result[2], 3*2 + 6*3);  // 6 + 18 = 24
}

TEST(UtilTest, MatrixMultiply) {
    // 2x3 matrix
    std::vector<int> a_vals = {
        1, 2, 3,
        4, 5, 6
    };
    RowMajorMatrix<int> a(a_vals, 3);

    // 3x2 matrix
    std::vector<int> b_vals = {
        7, 8,
        9, 10,
        11, 12
    };
    RowMajorMatrix<int> b(b_vals, 2);

    auto result = matrix_multiply(a, b);

    EXPECT_EQ(result.width(), 2);
    EXPECT_EQ(result.height(), 2);

    // Result should be 2x2:
    // [1*7+2*9+3*11,  1*8+2*10+3*12]   [58,  64]
    // [4*7+5*9+6*11,  4*8+5*10+6*12] = [139, 154]

    EXPECT_EQ(result.get(0, 0), 58);
    EXPECT_EQ(result.get(0, 1), 64);
    EXPECT_EQ(result.get(1, 0), 139);
    EXPECT_EQ(result.get(1, 1), 154);
}

TEST(UtilTest, MatrixMultiplyIncompatibleDimensions) {
    RowMajorMatrix<int> a(std::vector<int>{1, 2, 3, 4}, 2);  // 2x2
    RowMajorMatrix<int> b(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3);  // 3x3

    EXPECT_THROW(matrix_multiply(a, b), std::invalid_argument);
}


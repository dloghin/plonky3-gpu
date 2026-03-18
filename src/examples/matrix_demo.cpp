#include "dense_matrix.hpp"
#include "util.hpp"
#include <iostream>
#include <iomanip>

using namespace p3_matrix;

void print_matrix(const RowMajorMatrix<int>& mat, const std::string& name) {
    std::cout << "\n" << name << " (" << mat.height() << "x" << mat.width() << "):\n";
    for (size_t r = 0; r < mat.height(); ++r) {
        for (size_t c = 0; c < mat.width(); ++c) {
            std::cout << std::setw(4) << mat.get_unchecked(r, c);
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout << "=== C++ Matrix Library Demo ===\n";

    // 1. Create a simple matrix
    std::cout << "\n1. Creating matrices:\n";
    std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    RowMajorMatrix<int> mat(vals, 4);  // 3x4 matrix
    print_matrix(mat, "Original Matrix");

    // 2. Access elements
    std::cout << "\n2. Accessing elements:\n";
    std::cout << "   mat.get(0, 0) = " << mat.get(0, 0) << "\n";
    std::cout << "   mat.get(1, 2) = " << mat.get(1, 2) << "\n";
    std::cout << "   mat.get(2, 3) = " << mat.get(2, 3) << "\n";

    // 3. Transpose
    std::cout << "\n3. Matrix transpose:\n";
    auto transposed = mat.transpose();
    print_matrix(transposed, "Transposed Matrix");

    // 4. Matrix operations
    std::cout << "\n4. Matrix operations:\n";

    // Scale matrix
    RowMajorMatrix<int> scaled(vals, 4);
    scaled.scale(2);
    print_matrix(scaled, "Scaled by 2");

    // Scale single row
    RowMajorMatrix<int> row_scaled(vals, 4);
    row_scaled.scale_row(1, 3);
    print_matrix(row_scaled, "Second row scaled by 3");

    // 5. Matrix-vector multiplication
    std::cout << "\n5. Matrix-vector multiplication:\n";
    std::vector<int> vec = {1, 2, 3, 4};
    auto result = matrix_vector_mul(mat, vec);
    std::cout << "   Vector: [1, 2, 3, 4]\n";
    std::cout << "   Result: [";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // 6. Columnwise dot product
    std::cout << "\n6. Columnwise dot product (M^T * v):\n";
    std::vector<int> col_vec = {1, 2, 3};
    auto col_result = columnwise_dot_product(mat, col_vec);
    std::cout << "   Vector: [1, 2, 3]\n";
    std::cout << "   Result: [";
    for (size_t i = 0; i < col_result.size(); ++i) {
        std::cout << col_result[i];
        if (i < col_result.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // 7. Matrix multiplication
    std::cout << "\n7. Matrix multiplication:\n";
    std::vector<int> a_vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> a(a_vals, 3);  // 2x3

    std::vector<int> b_vals = {7, 8, 9, 10, 11, 12};
    RowMajorMatrix<int> b(b_vals, 2);  // 3x2

    auto product = matrix_multiply(a, b);
    print_matrix(a, "Matrix A");
    print_matrix(b, "Matrix B");
    print_matrix(product, "A * B");

    // 8. Split and pad
    std::cout << "\n8. Split and pad operations:\n";
    auto [top, bottom] = mat.split_rows(2);
    print_matrix(top, "Top half (rows 0-1)");
    print_matrix(bottom, "Bottom half (rows 2-)");

    RowMajorMatrix<int> padded(vals, 4);
    padded.pad_to_height(5, 99);
    print_matrix(padded, "Padded to height 5");

    // 9. Bit reversal operations (for FFT/NTT applications)
    std::cout << "\n9. Bit reversal operations:\n";
    std::vector<int> br_vals = {0, 1, 2, 3, 4, 5, 6, 7};
    RowMajorMatrix<int> br_mat(br_vals, 1);  // 8x1
    reverse_matrix_index_bits(br_mat);
    print_matrix(br_mat, "After bit-reversal");

    std::cout << "\n=== Demo Complete ===\n";

    return 0;
}


#include "matrix_cuda.hpp"
#include "dense_matrix.hpp"
#include <iostream>
#include <vector>
#include <chrono>

using namespace p3_matrix;

void print_matrix(const RowMajorMatrix<int>& mat, const std::string& name) {
    std::cout << name << " (" << mat.height() << "x" << mat.width() << "):\n";
    for (size_t r = 0; r < mat.height(); ++r) {
        for (size_t c = 0; c < mat.width(); ++c) {
            std::cout << mat.get(r, c) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "CUDA Matrix Library Example\n";
    std::cout << "========================================\n\n";

    // Example 1: Matrix transpose
    std::cout << "Example 1: Matrix Transpose\n";
    std::cout << "----------------------------\n";
    std::vector<int> vals = {1, 2, 3, 4, 5, 6};
    RowMajorMatrix<int> host_mat(vals, 3);  // 2x3 matrix
    print_matrix(host_mat, "Original matrix");

    CudaMatrix<int> cuda_mat(host_mat);
    auto cuda_transposed = cuda_mat.transpose();
    
    RowMajorMatrix<int> host_transposed(3, 2);
    cuda_transposed.copy_to_host(host_transposed);
    print_matrix(host_transposed, "Transposed matrix (CUDA)");

    // Example 2: Matrix-vector multiplication
    std::cout << "\nExample 2: Matrix-Vector Multiplication\n";
    std::cout << "----------------------------------------\n";
    std::vector<int> vec = {1, 2, 3};
    std::cout << "Vector: ";
    for (auto v : vec) std::cout << v << " ";
    std::cout << "\n\n";

    auto result = matrix_vector_mul_cuda(cuda_mat, vec);
    std::cout << "Result (M * v): ";
    for (auto v : result) std::cout << v << " ";
    std::cout << "\n\n";

    // Example 3: Columnwise dot product
    std::cout << "Example 3: Columnwise Dot Product (M^T * v)\n";
    std::cout << "-------------------------------------------\n";
    std::vector<int> scale_vec = {1, 2};
    std::cout << "Scale vector: ";
    for (auto v : scale_vec) std::cout << v << " ";
    std::cout << "\n\n";

    auto col_result = columnwise_dot_product_cuda(cuda_mat, scale_vec);
    std::cout << "Result (M^T * v): ";
    for (auto v : col_result) std::cout << v << " ";
    std::cout << "\n\n";

    // Example 4: Matrix scaling
    std::cout << "Example 4: Matrix Scaling\n";
    std::cout << "-------------------------\n";
    CudaMatrix<int> scaled_mat(host_mat);
    scaled_mat.scale(2);
    
    RowMajorMatrix<int> host_scaled(2, 3);
    scaled_mat.copy_to_host(host_scaled);
    print_matrix(host_scaled, "Scaled matrix (x2)");

    // Example 5: Matrix multiplication
    std::cout << "Example 5: Matrix Multiplication\n";
    std::cout << "---------------------------------\n";
    std::vector<int> a_vals = {1, 2, 3, 4, 5, 6};
    std::vector<int> b_vals = {1, 2, 3, 4};
    RowMajorMatrix<int> host_a(a_vals, 3);  // 2x3
    RowMajorMatrix<int> host_b(b_vals, 2);  // 2x2
    print_matrix(host_a, "Matrix A");
    print_matrix(host_b, "Matrix B");

    CudaMatrix<int> cuda_a(host_a);
    CudaMatrix<int> cuda_b(host_b);
    auto cuda_product = matrix_multiply_cuda(cuda_a, cuda_b);
    
    RowMajorMatrix<int> host_product(2, 2);
    cuda_product.copy_to_host(host_product);
    print_matrix(host_product, "Product (A * B)");

    // Example 6: Performance comparison
    std::cout << "Example 6: Performance Comparison\n";
    std::cout << "-----------------------------------\n";
    const size_t large_size = 1024;
    RowMajorMatrix<int> large_mat(large_size, large_size, 1);
    
    auto start = std::chrono::high_resolution_clock::now();
    CudaMatrix<int> large_cuda(large_mat);
    auto cuda_trans = large_cuda.transpose();
    RowMajorMatrix<int> result_host(large_size, large_size);
    cuda_trans.copy_to_host(result_host);
    auto end = std::chrono::high_resolution_clock::now();
    auto cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    start = std::chrono::high_resolution_clock::now();
    auto cpu_trans = large_mat.transpose();
    end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    std::cout << "Transpose of " << large_size << "x" << large_size << " matrix:\n";
    std::cout << "  CUDA time: " << cuda_time << " ms\n";
    std::cout << "  CPU time:  " << cpu_time << " ms\n";
    std::cout << "  Speedup:   " << (double)cpu_time / cuda_time << "x\n";

    std::cout << "\n========================================\n";
    std::cout << "All examples completed successfully!\n";
    std::cout << "========================================\n";

    return 0;
}

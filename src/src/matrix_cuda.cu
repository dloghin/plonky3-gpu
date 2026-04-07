#include "matrix_cuda.hpp"
#include "cuda_compat.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace p3_matrix {

// CUDA kernel for matrix transpose
template<typename T>
P3_GLOBAL void transpose_kernel(
    const T* input,
    T* output,
    size_t height,
    size_t width
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = height * width;
    
    if (idx < total) {
        size_t r = idx / width;
        size_t c = idx % width;
        // Transpose: output[c][r] = input[r][c]
        output[c * height + r] = input[r * width + c];
    }
}

// CUDA kernel for matrix-vector multiplication
template<typename T>
P3_GLOBAL void matrix_vector_mul_kernel(
    const T* matrix,
    const T* vector,
    T* result,
    size_t height,
    size_t width
) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height) {
        T sum = T();
        const T* row_ptr = matrix + row * width;
        for (size_t c = 0; c < width; ++c) {
            sum = sum + (row_ptr[c] * vector[c]);
        }
        result[row] = sum;
    }
}

// CUDA kernel for columnwise dot product (M^T * v)
template<typename T, typename S>
P3_GLOBAL void columnwise_dot_product_kernel(
    const T* matrix,
    const S* vector,
    S* result,
    size_t height,
    size_t width
) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < width) {
        S sum = S();
        for (size_t r = 0; r < height; ++r) {
            size_t idx = r * width + col;
            sum = sum + (vector[r] * matrix[idx]);
        }
        result[col] = sum;
    }
}

// CUDA kernel for matrix scaling
template<typename T, typename Scalar>
P3_GLOBAL void scale_kernel(
    T* matrix,
    Scalar scalar,
    size_t total_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        matrix[idx] = matrix[idx] * scalar;
    }
}

// CUDA kernel for row scaling
template<typename T, typename Scalar>
P3_GLOBAL void scale_row_kernel(
    T* matrix,
    Scalar scalar,
    size_t row,
    size_t width
) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < width) {
        size_t idx = row * width + col;
        matrix[idx] = matrix[idx] * scalar;
    }
}

// CUDA kernel for matrix multiplication
template<typename T>
P3_GLOBAL void matrix_multiply_kernel(
    const T* a,
    const T* b,
    T* c,
    size_t m,  // A height
    size_t n,  // B width
    size_t k   // A width = B height
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        T sum = T();
        for (size_t p = 0; p < k; ++p) {
            sum = sum + (a[row * k + p] * b[p * n + col]);
        }
        c[row * n + col] = sum;
    }
}

// Explicit instantiations for common types
template P3_GLOBAL void transpose_kernel<int>(const int*, int*, size_t, size_t);
template P3_GLOBAL void transpose_kernel<float>(const float*, float*, size_t, size_t);
template P3_GLOBAL void transpose_kernel<double>(const double*, double*, size_t, size_t);

template P3_GLOBAL void matrix_vector_mul_kernel<int>(const int*, const int*, int*, size_t, size_t);
template P3_GLOBAL void matrix_vector_mul_kernel<float>(const float*, const float*, float*, size_t, size_t);
template P3_GLOBAL void matrix_vector_mul_kernel<double>(const double*, const double*, double*, size_t, size_t);

template P3_GLOBAL void columnwise_dot_product_kernel<int, int>(const int*, const int*, int*, size_t, size_t);
template P3_GLOBAL void columnwise_dot_product_kernel<float, float>(const float*, const float*, float*, size_t, size_t);
template P3_GLOBAL void columnwise_dot_product_kernel<double, double>(const double*, const double*, double*, size_t, size_t);

template P3_GLOBAL void scale_kernel<int, int>(int*, int, size_t);
template P3_GLOBAL void scale_kernel<float, float>(float*, float, size_t);
template P3_GLOBAL void scale_kernel<double, double>(double*, double, size_t);

template P3_GLOBAL void scale_row_kernel<int, int>(int*, int, size_t, size_t);
template P3_GLOBAL void scale_row_kernel<float, float>(float*, float, size_t, size_t);
template P3_GLOBAL void scale_row_kernel<double, double>(double*, double, size_t, size_t);

template P3_GLOBAL void matrix_multiply_kernel<int>(const int*, const int*, int*, size_t, size_t, size_t);
template P3_GLOBAL void matrix_multiply_kernel<float>(const float*, const float*, float*, size_t, size_t, size_t);
template P3_GLOBAL void matrix_multiply_kernel<double>(const double*, const double*, double*, size_t, size_t, size_t);

// CudaMatrix method implementations
template<typename T>
CudaMatrix<T> CudaMatrix<T>::transpose() const {
    CudaMatrix<T> result(width_, height_);
    
    P3_CONSTEXPR size_t BLOCK_SIZE = 256;
    size_t total = size_;
    size_t num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    transpose_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_data, result.d_data, height_, width_);
    P3_CUDA_CHECK(cudaDeviceSynchronize());
    
    return result;
}

template<typename T>
template<typename Scalar>
void CudaMatrix<T>::scale(const Scalar& scalar) {
    if (size_ == 0) return;
    
    P3_CONSTEXPR size_t BLOCK_SIZE = 256;
    size_t num_blocks = (size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    scale_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, scalar, size_);
    P3_CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
template<typename Scalar>
void CudaMatrix<T>::scale_row(size_t row, const Scalar& scalar) {
    if (row >= height_) {
        throw std::out_of_range("Row index out of bounds");
    }
    
    P3_CONSTEXPR size_t BLOCK_SIZE = 256;
    size_t num_blocks = (width_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    scale_row_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_data, scalar, row, width_);
    P3_CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit instantiations for CudaMatrix methods
template CudaMatrix<int> CudaMatrix<int>::transpose() const;
template CudaMatrix<float> CudaMatrix<float>::transpose() const;
template CudaMatrix<double> CudaMatrix<double>::transpose() const;

template void CudaMatrix<int>::scale<int>(const int&);
template void CudaMatrix<float>::scale<float>(const float&);
template void CudaMatrix<double>::scale<double>(const double&);

template void CudaMatrix<int>::scale_row<int>(size_t, const int&);
template void CudaMatrix<float>::scale_row<float>(size_t, const float&);
template void CudaMatrix<double>::scale_row<double>(size_t, const double&);

// Standalone function implementations
template<typename T>
std::vector<T> matrix_vector_mul_cuda(
    const CudaMatrix<T>& matrix,
    const std::vector<T>& vec
) {
    if (vec.size() != matrix.width()) {
        throw std::invalid_argument("Vector size must match matrix width");
    }

    // Copy vector to device
    T* d_vec;
    P3_CUDA_CHECK(cudaMalloc(&d_vec, vec.size() * sizeof(T)));
    P3_CUDA_CHECK(cudaMemcpy(d_vec, vec.data(), vec.size() * sizeof(T),
        cudaMemcpyHostToDevice));

    // Allocate result on device
    T* d_result;
    P3_CUDA_CHECK(cudaMalloc(&d_result, matrix.height() * sizeof(T)));

    // Launch kernel
    P3_CONSTEXPR size_t BLOCK_SIZE = 256;
    size_t num_blocks = (matrix.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matrix_vector_mul_kernel<<<num_blocks, BLOCK_SIZE>>>(
        matrix.d_data, d_vec, d_result, matrix.height(), matrix.width());
    P3_CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    std::vector<T> result(matrix.height());
    P3_CUDA_CHECK(cudaMemcpy(result.data(), d_result, matrix.height() * sizeof(T),
        cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_vec);
    cudaFree(d_result);

    return result;
}

template<typename T, typename S>
std::vector<S> columnwise_dot_product_cuda(
    const CudaMatrix<T>& matrix,
    const std::vector<S>& vec
) {
    if (vec.size() != matrix.height()) {
        throw std::invalid_argument("Vector size must match matrix height");
    }

    // Copy vector to device
    S* d_vec;
    P3_CUDA_CHECK(cudaMalloc(&d_vec, vec.size() * sizeof(S)));
    P3_CUDA_CHECK(cudaMemcpy(d_vec, vec.data(), vec.size() * sizeof(S),
        cudaMemcpyHostToDevice));

    // Allocate result on device
    S* d_result;
    P3_CUDA_CHECK(cudaMalloc(&d_result, matrix.width() * sizeof(S)));

    // Launch kernel
    P3_CONSTEXPR size_t BLOCK_SIZE = 256;
    size_t num_blocks = (matrix.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    columnwise_dot_product_kernel<<<num_blocks, BLOCK_SIZE>>>(
        matrix.d_data, d_vec, d_result, matrix.height(), matrix.width());
    P3_CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    std::vector<S> result(matrix.width());
    P3_CUDA_CHECK(cudaMemcpy(result.data(), d_result, matrix.width() * sizeof(S),
        cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_vec);
    cudaFree(d_result);

    return result;
}

template<typename T>
CudaMatrix<T> matrix_multiply_cuda(
    const CudaMatrix<T>& a,
    const CudaMatrix<T>& b
) {
    if (a.width() != b.height()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    size_t m = a.height();
    size_t n = b.width();
    size_t k = a.width();

    CudaMatrix<T> result(m, n);

    // Launch kernel with 2D grid
    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);

    matrix_multiply_kernel<<<grid_size, block_size>>>(
        a.d_data, b.d_data, result.d_data, m, n, k);
    P3_CUDA_CHECK(cudaDeviceSynchronize());

    return result;
}

// Explicit instantiations for standalone functions
template std::vector<int> matrix_vector_mul_cuda<int>(const CudaMatrix<int>&, const std::vector<int>&);
template std::vector<float> matrix_vector_mul_cuda<float>(const CudaMatrix<float>&, const std::vector<float>&);
template std::vector<double> matrix_vector_mul_cuda<double>(const CudaMatrix<double>&, const std::vector<double>&);

template std::vector<int> columnwise_dot_product_cuda<int, int>(const CudaMatrix<int>&, const std::vector<int>&);
template std::vector<float> columnwise_dot_product_cuda<float, float>(const CudaMatrix<float>&, const std::vector<float>&);
template std::vector<double> columnwise_dot_product_cuda<double, double>(const CudaMatrix<double>&, const std::vector<double>&);

template CudaMatrix<int> matrix_multiply_cuda<int>(const CudaMatrix<int>&, const CudaMatrix<int>&);
template CudaMatrix<float> matrix_multiply_cuda<float>(const CudaMatrix<float>&, const CudaMatrix<float>&);
template CudaMatrix<double> matrix_multiply_cuda<double>(const CudaMatrix<double>&, const CudaMatrix<double>&);

} // namespace p3_matrix

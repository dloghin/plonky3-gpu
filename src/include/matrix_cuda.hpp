#pragma once

#include "dense_matrix.hpp"
#include "cuda_compat.hpp"
#include <vector>
#if P3_CUDA_ENABLED
#include <cuda_runtime.h>
#endif
#include <stdexcept>

namespace p3_matrix {

/**
 * @brief CUDA matrix operations wrapper
 *
 * This class provides GPU-accelerated matrix operations using CUDA.
 * It manages device memory and provides high-level operations.
 */
template<typename T>
class CudaMatrix {
public:
    T* d_data;      ///< Device pointer to matrix data
    size_t width_;  ///< Number of columns
    size_t height_; ///< Number of rows
    size_t size_;   ///< Total number of elements

    /**
     * @brief Construct empty CUDA matrix
     */
    CudaMatrix() : d_data(nullptr), width_(0), height_(0), size_(0) {}

    /**
     * @brief Construct CUDA matrix from host matrix
     * @param host_matrix Host matrix to copy to device
     */
    explicit CudaMatrix(const RowMajorMatrix<T>& host_matrix)
        : width_(host_matrix.width()), height_(host_matrix.height()),
          size_(host_matrix.size()) {
        if (size_ > 0) {
            P3_CUDA_CHECK(cudaMalloc(&d_data, size_ * sizeof(T)));
            P3_CUDA_CHECK(cudaMemcpy(d_data, host_matrix.values.data(),
                size_ * sizeof(T), cudaMemcpyHostToDevice));
        } else {
            d_data = nullptr;
        }
    }

    /**
     * @brief Construct CUDA matrix with specified dimensions
     * @param rows Number of rows
     * @param cols Number of columns
     * @param default_val Value to initialize with (on device)
     */
    CudaMatrix(size_t rows, size_t cols, const T& default_val = T())
        : width_(cols), height_(rows), size_(rows * cols) {
        if (size_ > 0) {
            P3_CUDA_CHECK(cudaMalloc(&d_data, size_ * sizeof(T)));
            // Initialize on device
            std::vector<T> host_data(size_, default_val);
            P3_CUDA_CHECK(cudaMemcpy(d_data, host_data.data(),
                size_ * sizeof(T), cudaMemcpyHostToDevice));
        } else {
            d_data = nullptr;
        }
    }

    /**
     * @brief Copy constructor (disabled - use explicit copy)
     */
    CudaMatrix(const CudaMatrix&) = delete;

    /**
     * @brief Move constructor
     */
    CudaMatrix(CudaMatrix&& other) noexcept
        : d_data(other.d_data), width_(other.width_), height_(other.height_),
          size_(other.size_) {
        other.d_data = nullptr;
        other.width_ = 0;
        other.height_ = 0;
        other.size_ = 0;
    }

    /**
     * @brief Destructor - frees device memory
     */
    ~CudaMatrix() {
        if (d_data != nullptr) {
            cudaFree(d_data);
        }
    }

    /**
     * @brief Copy assignment (disabled)
     */
    CudaMatrix& operator=(const CudaMatrix&) = delete;

    /**
     * @brief Move assignment
     */
    CudaMatrix& operator=(CudaMatrix&& other) noexcept {
        if (this != &other) {
            if (d_data != nullptr) {
                cudaFree(d_data);
            }
            d_data = other.d_data;
            width_ = other.width_;
            height_ = other.height_;
            size_ = other.size_;
            other.d_data = nullptr;
            other.width_ = 0;
            other.height_ = 0;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get matrix width
     */
    size_t width() const { return width_; }

    /**
     * @brief Get matrix height
     */
    size_t height() const { return height_; }

    /**
     * @brief Get total number of elements
     */
    size_t size() const { return size_; }

    /**
     * @brief Copy data from host matrix to device
     * @param host_matrix Host matrix to copy from
     */
    void copy_from_host(const RowMajorMatrix<T>& host_matrix) {
        if (host_matrix.width() != width_ || host_matrix.height() != height_) {
            // Reallocate if dimensions changed
            if (d_data != nullptr) {
                cudaFree(d_data);
            }
            width_ = host_matrix.width();
            height_ = host_matrix.height();
            size_ = host_matrix.size();
            if (size_ > 0) {
                P3_CUDA_CHECK(cudaMalloc(&d_data, size_ * sizeof(T)));
            } else {
                d_data = nullptr;
            }
        }
        if (size_ > 0) {
            P3_CUDA_CHECK(cudaMemcpy(d_data, host_matrix.values.data(),
                size_ * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    /**
     * @brief Copy data from device to host matrix
     * @param host_matrix Host matrix to copy to
     */
    void copy_to_host(RowMajorMatrix<T>& host_matrix) const {
        if (host_matrix.width() != width_ || host_matrix.height() != height_) {
            host_matrix = RowMajorMatrix<T>(height_, width_, T());
        }
        if (size_ > 0) {
            P3_CUDA_CHECK(cudaMemcpy(host_matrix.values.data(), d_data,
                size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    /**
     * @brief Transpose matrix on GPU
     * @return New transposed CUDA matrix
     */
    CudaMatrix<T> transpose() const;

    /**
     * @brief Scale entire matrix by scalar
     * @param scalar Scalar value
     */
    template<typename Scalar>
    void scale(const Scalar& scalar);

    /**
     * @brief Scale a single row
     * @param row Row index
     * @param scalar Scalar value
     */
    template<typename Scalar>
    void scale_row(size_t row, const Scalar& scalar);
};

/**
 * @brief CUDA matrix-vector multiplication
 * @param matrix CUDA matrix
 * @param vec Host vector
 * @return Result vector (on host)
 */
template<typename T>
std::vector<T> matrix_vector_mul_cuda(
    const CudaMatrix<T>& matrix,
    const std::vector<T>& vec
);

/**
 * @brief CUDA columnwise dot product (M^T * v)
 * @param matrix CUDA matrix
 * @param vec Host vector
 * @return Result vector (on host)
 */
template<typename T, typename S>
std::vector<S> columnwise_dot_product_cuda(
    const CudaMatrix<T>& matrix,
    const std::vector<S>& vec
);

/**
 * @brief CUDA matrix multiplication
 * @param a First CUDA matrix
 * @param b Second CUDA matrix
 * @return Product matrix (on GPU)
 */
template<typename T>
CudaMatrix<T> matrix_multiply_cuda(
    const CudaMatrix<T>& a,
    const CudaMatrix<T>& b
);

// Forward declarations for kernel functions (defined in matrix_cuda.cu)
template<typename T>
P3_GLOBAL void transpose_kernel(const T* input, T* output, size_t height, size_t width);

template<typename T>
P3_GLOBAL void matrix_vector_mul_kernel(
    const T* matrix, const T* vector, T* result, size_t height, size_t width);

template<typename T, typename S>
P3_GLOBAL void columnwise_dot_product_kernel(
    const T* matrix, const S* vector, S* result, size_t height, size_t width);

template<typename T, typename Scalar>
P3_GLOBAL void scale_kernel(T* matrix, Scalar scalar, size_t total_elements);

template<typename T, typename Scalar>
P3_GLOBAL void scale_row_kernel(
    T* matrix, Scalar scalar, size_t row, size_t width);

template<typename T>
P3_GLOBAL void matrix_multiply_kernel(
    const T* a, const T* b, T* c, size_t m, size_t n, size_t k);

} // namespace p3_matrix

# C++ Matrix Library

A C++ implementation of the Plonky3 matrix library providing efficient matrix operations for zero-knowledge proof systems.

## Features

- **Dense Row-Major Matrices**: Efficient storage and access patterns
- **Matrix Operations**: Transpose, multiplication, scaling, dot products
- **Utility Functions**: Bit-reversal, matrix-vector products, columnwise operations
- **Header-Only**: Easy integration into projects
- **Well-Tested**: Comprehensive test suite with Google Test
- **Benchmarked**: Performance benchmarks using Google Benchmark

## Quick Start

### Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Running Tests

```bash
cd build
./tests/matrix_tests
```

### Running Benchmarks

```bash
# Transpose benchmark
./benches/transpose_benchmark

# Columnwise dot product benchmark
./benches/columnwise_dot_product
```

### Running Example

```bash
./examples/matrix_demo
```

## Usage

### Basic Matrix Creation

```cpp
#include "dense_matrix.hpp"
using namespace p3_matrix;

// Create from vector
std::vector<int> vals = {1, 2, 3, 4, 5, 6};
RowMajorMatrix<int> mat(vals, 3);  // 2x3 matrix

// Create with default values
RowMajorMatrix<int> mat2(3, 4, 0);  // 3x4 matrix filled with zeros

// Create single row/column
auto row = RowMajorMatrix<int>::new_row({1, 2, 3, 4});
auto col = RowMajorMatrix<int>::new_col({1, 2, 3});
```

### Matrix Access

```cpp
// Safe access
int value = mat.get(0, 1);  // Throws if out of bounds

// Unchecked access (faster, but unsafe)
int value2 = mat.get_unchecked(0, 1);

// Row access
auto row_vec = mat.row(0);  // Get row as vector
const int* row_ptr = mat.row_ptr(0);  // Direct pointer access
```

### Matrix Operations

```cpp
// Transpose
auto transposed = mat.transpose();

// Transpose into pre-allocated matrix
RowMajorMatrix<int> dest(3, 2, 0);
mat.transpose_into(dest);

// Scale entire matrix
mat.scale(2);

// Scale single row
mat.scale_row(1, 3);

// Swap rows
mat.swap_rows(0, 2);
```

### Matrix-Vector Operations

```cpp
#include "util.hpp"

// Matrix-vector multiplication: M * v
std::vector<int> vec = {1, 2, 3};
auto result = matrix_vector_mul(mat, vec);

// Columnwise dot product: M^T * v
std::vector<int> scales = {2, 3};
auto col_result = columnwise_dot_product(mat, scales);

// Dot product of two vectors
auto dot = dot_product(vec1, vec2);
```

### Matrix Multiplication

```cpp
// C = A * B
auto product = matrix_multiply(a, b);
```

### Special Operations

```cpp
// Bit-reversal (for FFT/NTT)
reverse_matrix_index_bits(mat);

// Bit-reversed zero padding
auto padded = bit_reversed_zero_pad(mat, added_bits);

// Split matrix at row
auto [top, bottom] = mat.split_rows(2);

// Pad to new height
mat.pad_to_height(10, 0);
```

## API Overview

### Matrix Interface

All matrix types implement the `Matrix<T>` interface:

```cpp
template<typename T>
class Matrix {
    virtual size_t width() const = 0;
    virtual size_t height() const = 0;
    virtual T get_unchecked(size_t r, size_t c) const = 0;
    // ... more methods
};
```

### RowMajorMatrix

Main implementation storing data in row-major order:

#### Key Methods
- `width()`, `height()` - Matrix dimensions
- `get(r, c)` - Safe element access
- `get_unchecked(r, c)` - Fast unchecked access
- `set(r, c, value)` - Modify element
- `row(r)` - Get row as vector
- `row_ptr(r)` - Get direct pointer to row
- `row_mut(r)` - Get mutable pointer to row
- `transpose()` - Return transposed matrix
- `transpose_into(dest)` - Transpose into destination
- `scale(scalar)` - Scale all elements
- `scale_row(r, scalar)` - Scale single row
- `swap_rows(i, j)` - Swap two rows
- `split_rows(r)` - Split at row r
- `pad_to_height(h, fill)` - Pad to new height
- `copy_from(source)` - Copy from another matrix

### Utility Functions

From `util.hpp`:

- `reverse_bits_len(n, bits)` - Bit reversal
- `log2_strict(n)` - Log base 2 (power of 2 only)
- `reverse_matrix_index_bits(mat)` - In-place bit-reversal of rows
- `bit_reversed_zero_pad(mat, bits)` - Pad with interleaved zeros
- `dot_product(a, b)` - Vector dot product
- `matrix_vector_mul(mat, vec)` - M * v
- `columnwise_dot_product(mat, vec)` - M^T * v
- `matrix_multiply(a, b)` - A * B

## Performance

The library includes comprehensive benchmarks mirroring the Rust implementation:

### Transpose Benchmark

Tests various matrix sizes from 4x4 to very large matrices (2^23 x 2^8).

```bash
./benches/transpose_benchmark
```

### Columnwise Dot Product Benchmark

Tests the columnwise dot product operation on large matrices.

```bash
./benches/columnwise_dot_product
```

## Requirements

- **C++17** or later
- **CMake 3.15+**
- **Compiler**: GCC 7+, Clang 5+, MSVC 2019+

### Optional Dependencies
- **Google Test** (auto-fetched for tests)
- **Google Benchmark** (auto-fetched for benchmarks)

## Project Structure

```
cpp_matrix/
├── include/
│   ├── matrix.hpp           # Base matrix interface
│   ├── dense_matrix.hpp     # Dense matrix implementation
│   └── util.hpp             # Utility functions
├── tests/
│   ├── test_matrix.cpp
│   ├── test_dense_matrix.cpp
│   └── test_util.cpp
├── benches/
│   ├── transpose_benchmark.cpp
│   └── columnwise_dot_product.cpp
├── examples/
│   └── matrix_demo.cpp
├── CMakeLists.txt
└── README.md
```

## Comparison with Rust

This C++ implementation closely mirrors the Rust `p3-matrix` crate:

| Rust | C++ | Notes |
|------|-----|-------|
| `Matrix` trait | `Matrix<T>` class | Virtual base class |
| `RowMajorMatrix` | `RowMajorMatrix<T>` | Dense storage |
| `transpose()` | `transpose()` | Returns new matrix |
| `transpose_into()` | `transpose_into()` | In-place variant |
| `columnwise_dot_product()` | `columnwise_dot_product()` | M^T * v |
| `reverse_matrix_index_bits()` | `reverse_matrix_index_bits()` | Bit reversal |

### Key Differences

1. **Memory Management**: C++ uses `std::vector` instead of Rust's `Vec`
2. **Error Handling**: C++ uses exceptions instead of `Result`/`Option`
3. **Generics**: C++ templates instead of Rust generics
4. **Packed Fields**: Not yet implemented (Rust has SIMD support)
5. **Parallelism**: Not yet parallelized (Rust uses rayon)

## Benchmarks

### Sample Results

On a 32-core system @ 5.7 GHz:

```
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
BM_Transpose_8x8                 1.23 us         1.23 us       568234
BM_Transpose_12x12              18.4 us         18.4 us        38012
BM_Transpose_20x8               52.3 ms         52.3 ms           13
BM_ColumnwiseDotProduct_Small    145 us          145 us         4821
BM_ColumnwiseDotProduct_Medium  4.82 ms         4.82 ms          145
```

## Integration

### As a Header-Only Library

Simply include the headers and add the include path to your project:

```cmake
target_include_directories(your_target PRIVATE /path/to/cpp_matrix/include)
```

### With CMake

```cmake
add_subdirectory(cpp_matrix)
target_link_libraries(your_target PRIVATE matrix)
```

## Testing

Run all tests:

```bash
cd build
ctest --output-on-failure
```

Or run the test binary directly:

```bash
./tests/matrix_tests
```

## Future Work

- [ ] Parallel operations (OpenMP/TBB)
- [ ] SIMD/packed field support
- [ ] Additional matrix views (strided, truncated, etc.)
- [ ] Optimized transpose using cache-oblivious algorithms
- [ ] Extension field support for ZK operations

## License

Same as Plonky3 project

## Acknowledgments

Based on the Rust implementation from [Plonky3](https://github.com/Plonky3/Plonky3).


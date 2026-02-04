# C++ Field Library

A C++ implementation of prime field arithmetic for use in zero-knowledge proof systems. This library provides implementations of Baby Bear, Goldilocks, and Mersenne-31 fields with support for both **CPU (C++)** and **GPU (CUDA)** execution.

## Fields Implemented

### Baby Bear
- **Prime**: `p = 2^31 - 2^27 + 1 = 0x78000001 = 2013265921`
- **Characteristics**: 31-bit prime with high 2-adicity (27)
- **S-box**: x^5 is injective

### Goldilocks
- **Prime**: `p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001`
- **Characteristics**: 64-bit prime with 2-adicity of 32
- **S-box**: x^7 is injective

### Mersenne-31
- **Prime**: `p = 2^31 - 1 = 0x7FFFFFFF = 2147483647`
- **Characteristics**: Mersenne prime with efficient reduction
- **S-box**: x^5 is injective

## Features

- **Header-only library**: Easy integration
- **CRTP-based design**: Type-safe with zero runtime overhead
- **Efficient arithmetic**: Optimized field operations
- **Template-based exponentiation**: Compile-time optimization
- **Support for Poseidon2**: Compatible with cpp_poseidon2 library
- **CPU/GPU support**: Same code runs on both CPU and NVIDIA CUDA GPUs

## CUDA/GPU Support

The library supports both CPU and GPU execution using NVIDIA CUDA. The same field classes can be used in both CPU code and CUDA kernels without code duplication.

### Key Features

- `__host__ __device__` qualified methods for seamless CPU/GPU execution
- Portable 128-bit arithmetic for Goldilocks field on GPU (CUDA doesn't support `__uint128_t`)
- CUDA intrinsics for efficient 64x64->128 bit multiplication (`__umul64hi`)
- No code duplication - single implementation works on both platforms

### Building with CUDA

```bash
# Using the build script (auto-detects CUDA)
./build_cuda.sh

# Or manually with CMake
mkdir build_cuda && cd build_cuda
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_EXAMPLES=ON \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --parallel
```

### CUDA Examples

The library includes comprehensive GPU examples for each field:

```bash
# Run after building
./bin/goldilocks_cuda_example
./bin/baby_bear_cuda_example
./bin/mersenne31_cuda_example
```

These examples demonstrate:
- Element-wise field addition, subtraction, multiplication
- Squaring and power operations (S-box)
- Modular inverse computation
- Polynomial evaluation
- Dot product with parallel reduction

### Writing CUDA Kernels

```cuda
#include "goldilocks.hpp"
#include <cuda_runtime.h>

using namespace p3_field;

__global__ void field_kernel(const uint64_t* a, const uint64_t* b,
                             uint64_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Goldilocks fa(a[idx]);
        Goldilocks fb(b[idx]);
        Goldilocks fc = fa * fb + fa.square();
        result[idx] = fc.value();
    }
}
```

### GPU-Compatible Factory Methods

For GPU compatibility, use the static factory methods instead of static constants:

```cpp
// CPU-only (works with static constants)
Goldilocks a = Goldilocks::ZERO;  // OK on CPU
Goldilocks b = Goldilocks::ONE;   // OK on CPU

// GPU-compatible (works everywhere)
Goldilocks a = Goldilocks::zero_val();  // Works on CPU and GPU
Goldilocks b = Goldilocks::one_val();   // Works on CPU and GPU
```

## Usage (CPU)

```cpp
#include "baby_bear.hpp"
#include "goldilocks.hpp"
#include "mersenne31.hpp"

using namespace p3_field;

// Baby Bear arithmetic
BabyBear a(100);
BabyBear b(200);
BabyBear c = a + b;
BabyBear d = a * b;
BabyBear e = a.inverse();

// Goldilocks arithmetic
Goldilocks x(42);
Goldilocks y = x.square();
Goldilocks z = x.exp_u64(10);

// Mersenne-31 arithmetic
Mersenne31 m(7);
Mersenne31 n = m.injective_exp_n<5>();
```

## Building

### Building the Library and Examples (CPU only)

```bash
mkdir build && cd build
cmake ..
cmake --build .
./bin/field_demo
```

### Building with CUDA Support

```bash
# Quick method using the build script
./build_cuda.sh

# Or manually
mkdir build_cuda && cd build_cuda
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --parallel
```

### Building and Running Benchmarks

The project includes comprehensive benchmarks using Google Benchmark:

```bash
# Quick method using the build script
cd benches
./build_and_run.sh

# Or manually
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
cmake --build . --target bench_field
./benches/bench_field
```

See [benches/README.md](benches/README.md) for detailed benchmark documentation and usage examples.

## API Overview

### Common Operations

All fields support:
- **Arithmetic**: `+`, `-`, `*`, `+=`, `-=`, `*=`
- **Comparison**: `==`, `!=`
- **Special**: `double_val()`, `square()`, `cube()`
- **Exponentiation**: `exp_u64(n)`, `exp_const_u64<N>()`
- **Inverse**: `inverse()`, `inv()`
- **Power maps**: `injective_exp_n<D>()`

### Constants

Each field provides:
- `ZERO` - Additive identity (CPU only, use `zero_val()` for GPU)
- `ONE` - Multiplicative identity (CPU only, use `one_val()` for GPU)
- `TWO` - 1 + 1 (CPU only, use `two_val()` for GPU)
- `NEG_ONE` - -1 (p - 1) (CPU only, use `neg_one_val()` for GPU)
- `PRIME` - Field modulus

### GPU-Compatible Static Methods

For code that needs to run on both CPU and GPU:
- `zero_val()` - Returns zero element
- `one_val()` - Returns one element
- `two_val()` - Returns two element
- `neg_one_val()` - Returns -1 element

## Integration with Poseidon2

This library is designed to work seamlessly with `cpp_poseidon2`:

```cpp
#include "baby_bear.hpp"
#include "poseidon2.hpp"

using namespace p3_field;

// Create Poseidon2 with Baby Bear
auto poseidon = poseidon2::create_poseidon2<BabyBear, BabyBear, 16, 5>(
    external_constants,
    internal_constants
);

std::array<BabyBear, 16> state = ...;
poseidon->permute_mut(state);
```

## Performance Considerations

### CPU
- **Baby Bear**: 31-bit arithmetic using uint32_t
- **Goldilocks**: 64-bit arithmetic using uint64_t and __uint128_t
- **Mersenne-31**: Fast reduction using Mersenne prime properties

### GPU (CUDA)
- **Baby Bear**: Efficient 32-bit operations
- **Goldilocks**: Uses CUDA `__umul64hi` intrinsic for 128-bit multiplication
- **Mersenne-31**: Fast 31-bit reduction leveraging Mersenne properties

## Requirements

### CPU Only
- C++17 or later
- CMake 3.15+
- Compiler with __uint128_t support (GCC/Clang)

### With CUDA Support
- C++17 or later
- CMake 3.18+
- NVIDIA CUDA Toolkit 11.0+ (recommended 12.x)
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)

## File Structure

```
cpp_field/
├── include/
│   ├── cuda_compat.hpp    # CUDA compatibility macros
│   ├── field.hpp          # Base field template
│   ├── baby_bear.hpp      # Baby Bear field
│   ├── goldilocks.hpp     # Goldilocks field
│   └── mersenne31.hpp     # Mersenne-31 field
├── examples/
│   ├── field_demo.cpp               # CPU demo
│   ├── goldilocks_cuda_example.cu   # Goldilocks GPU demo
│   ├── baby_bear_cuda_example.cu    # Baby Bear GPU demo
│   └── mersenne31_cuda_example.cu   # Mersenne-31 GPU demo
├── benches/
│   └── bench_field.cpp    # CPU benchmarks
├── CMakeLists.txt
├── build_cuda.sh          # Build script with CUDA
└── README.md
```

## License

Same as Plonky3 project

## Acknowledgments

Based on the Rust implementation from [Plonky3](https://github.com/Plonky3/Plonky3).

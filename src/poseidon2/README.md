# C++ Poseidon2 Implementation

A C++ implementation of the Poseidon2 cryptographic permutation, ported from the Rust implementation in Plonky3.

## Overview

This is a header-only C++ library that implements the Poseidon2 cryptographic permutation designed for zero-knowledge applications. Poseidon2 is an optimized version of the original Poseidon hash function with improved performance characteristics.

## References

This implementation is based upon the following resources:
- [Horizen Labs Poseidon2 Implementation](https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2.rs)
- [Poseidon2 Paper](https://eprint.iacr.org/2023/323.pdf)
- [Original Poseidon Paper](https://eprint.iacr.org/2019/458.pdf)

## Features

- **Header-only library**: Easy integration into existing projects
- **Generic implementation**: Works with any field type that satisfies the required traits
- **Optimized matrix operations**: Efficient 4×4 MDS matrix implementations
- **128-bit security**: Pre-computed round numbers for common configurations
- **Supported widths**: 2, 3, 4, 8, 12, 16, 20, 24

## Components

### Core Headers

- **poseidon2.hpp**: Main header file with the `Poseidon2` class and factory functions
- **external.hpp**: External layer implementations (initial and terminal rounds)
- **internal.hpp**: Internal layer implementations (partial rounds)
- **generic.hpp**: Generic implementations for any field
- **round_numbers.hpp**: Security parameter computations

### Key Classes

1. **`Poseidon2<F, A, WIDTH, D>`**: The main permutation class
   - `F`: Base field type
   - `A`: Algebra type (typically F or a packed field)
   - `WIDTH`: State width (must be supported)
   - `D`: S-box exponent

2. **`ExternalLayer`**: Trait for external round implementations
3. **`InternalLayer`**: Trait for internal round implementations
4. **`MDSMat4`**: Optimized 4×4 MDS matrix (7 additions + 2 doubles)
5. **`HLMDSMat4`**: Horizon Labs 4×4 MDS matrix (10 additions + 4 doubles)

## Usage

### Basic Example

```cpp
#include "poseidon2.hpp"

// Define your field type
class MyField {
public:
    static MyField zero();
    MyField operator+(const MyField& other) const;
    MyField operator*(const MyField& other) const;
    MyField& operator+=(const MyField& other);
    MyField& operator*=(const MyField& other);
    MyField double_val() const;
    template<uint64_t D>
    MyField injective_exp_n() const;
};

// Create constants (you need to provide these)
poseidon2::ExternalLayerConstants<MyField, 16> external_constants = ...;
std::vector<MyField> internal_constants = ...;

// Create Poseidon2 instance
auto poseidon = poseidon2::create_poseidon2<MyField, MyField, 16, 7>(
    external_constants,
    internal_constants
);

// Use the permutation
std::array<MyField, 16> state = ...;
poseidon->permute_mut(state);
```

### With 128-bit Security

```cpp
// Automatically determine round numbers for 128-bit security
constexpr uint64_t FIELD_ORDER = (1ULL << 31) - 1; // Example: Mersenne-31

auto poseidon = poseidon2::create_poseidon2_128<MyField, MyField, 16, 7>(
    FIELD_ORDER,
    external_constants,
    internal_constants
);
```

## Building

### Requirements

- C++17 or later
- CMake 3.15 or later

### Build Instructions

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Build Options

- `BUILD_EXAMPLES=ON/OFF`: Build example programs (default: ON)
- `BUILD_TESTS=ON/OFF`: Build unit tests (default: ON)
- `ENABLE_WARNINGS=ON/OFF`: Enable compiler warnings (default: ON)

### Installation

```bash
cmake --install .
```

## Field Requirements

To use this library with your field type, it must provide:

1. **Basic arithmetic**: `+`, `*`, `+=`, `*=`
2. **Zero element**: `static F zero()`
3. **Doubling**: `F double_val() const`
4. **Power map**: `template<uint64_t D> F injective_exp_n() const`

## Round Numbers

The library includes pre-computed optimal round numbers for 128-bit security:

### 31-bit Primes (e.g., Baby Bear, Mersenne-31)
- Width 16: (8, 13-20) depending on D
- Width 24: (8, 21-23) depending on D

### 64-bit Primes (e.g., Goldilocks)
- Width 8: (8, 17-41) depending on D
- Width 12: (8, 18-42) depending on D
- Width 16: (8, 18-42) depending on D

## Security Considerations

This implementation follows the security analysis from the original Poseidon paper. The round numbers are chosen to achieve 128-bit security against:
- Statistical attacks
- Interpolation attacks
- Gröbner basis attacks
- Algebraic attacks

## Performance Notes

This is a generic implementation prioritizing correctness and readability. For production use cases requiring maximum performance:

1. Consider field-specific optimizations
2. Use packed field operations where available
3. Implement SIMD instructions for matrix operations
4. Pre-compute and store round constants in optimized formats

## License

This implementation follows the same license as the original Plonky3 project.

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- All headers are properly documented
- Examples compile and run correctly

## Acknowledgments

This C++ port is based on the Rust implementation from the Plonky3 project. Special thanks to the original authors and the Horizen Labs team for their reference implementation.


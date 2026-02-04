# C++ Poseidon Implementation

A C++ implementation of the Poseidon cryptographic permutation, ported from the Rust implementation in Plonky3.

## Overview

This is a header-only C++ library that implements the Poseidon permutation designed for zero-knowledge applications. Poseidon is a family of hash functions and cryptographic permutations specifically designed for use in proof systems like SNARKs and STARKs.

## References

This implementation is based upon:
- [Plonky3 Poseidon Implementation](https://github.com/Plonky3/Plonky3/tree/main/poseidon)
- [Poseidon Paper](https://eprint.iacr.org/2019/458.pdf)
- [Starkware Poseidon](https://docs.starkware.co/starkex/crypto/pedersen-hash-function.html)

## Key Differences: Poseidon vs Poseidon2

This library implements the original **Poseidon** permutation, which differs from **Poseidon2**:

### Poseidon (this library)
- Original design with full and partial rounds
- Uses generic MDS matrix multiplication after each round
- Full S-box layer in full rounds (all elements)
- Partial S-box layer in partial rounds (first element only)
- Round structure: Full → Partial → Full
- Well-studied security properties

### Poseidon2 (cpp_poseidon2)
- Optimized version with improved performance
- Uses specialized 4×4 MDS matrices for better efficiency
- External rounds (beginning and end)
- Internal rounds (middle, partial)
- Different round constant strategy
- Better performance characteristics

**When to use which:**
- Use **Poseidon** for maximum compatibility and well-established security analysis
- Use **Poseidon2** for better performance in new implementations

## Features

- **Header-only library**: Easy integration into existing projects
- **Generic implementation**: Works with any field type
- **Flexible configuration**: Customizable round numbers and widths
- **Multiple field support**: BabyBear, Goldilocks, Mersenne-31
- **Circulant MDS matrices**: Efficient matrix operations
- **Supported widths**: 8, 12, 16, 24, 32

## Components

### Core Headers

- **poseidon.hpp**: Main Poseidon class and factory functions
- **mds_matrix.hpp**: MDS matrix implementations for various fields and widths

### Key Classes

1. **`Poseidon<F, A, Mds, WIDTH, ALPHA>`**: The main permutation class
   - `F`: Base field type
   - `A`: Algebra type (typically F or a packed field)
   - `Mds`: MDS matrix type
   - `WIDTH`: State width
   - `ALPHA`: S-box exponent

2. **`MdsPermutation<T, WIDTH>`**: Base class for MDS matrices

3. **`CirculantMdsMatrix<F, WIDTH>`**: Generic circulant matrix implementation

4. **Field-specific MDS matrices**:
   - `MdsMatrixBabyBear8`, `MdsMatrixBabyBear12`, `MdsMatrixBabyBear16`, `MdsMatrixBabyBear24`
   - `MdsMatrixGoldilocks8`, `MdsMatrixGoldilocks12`, `MdsMatrixGoldilocks16`
   - `MdsMatrixMersenne3116`, `MdsMatrixMersenne3132`

## Usage

### Basic Example

```cpp
#include "poseidon.hpp"
#include "mds_matrix.hpp"
#include "baby_bear.hpp"

using namespace p3_field;
using namespace p3_poseidon;

// Configuration
constexpr size_t WIDTH = 16;
constexpr uint64_t ALPHA = 7;  // For BabyBear
const size_t half_num_full_rounds = 4;
const size_t num_partial_rounds = 22;

// Generate round constants (you should use proper constant generation)
std::vector<BabyBear> constants = generate_your_constants();

// Create MDS matrix
auto mds = std::make_shared<MdsMatrixBabyBear16<BabyBear>>();

// Create Poseidon instance
auto poseidon = create_poseidon<BabyBear, BabyBear,
                                MdsPermutation<BabyBear, WIDTH>,
                                WIDTH, ALPHA>(
    half_num_full_rounds,
    num_partial_rounds,
    constants,
    mds
);

// Use the permutation
std::array<BabyBear, WIDTH> state = { /* your state */ };
poseidon->permute_mut(state);
```

### Field-Specific Examples

#### BabyBear (ALPHA = 7)

```cpp
auto mds = std::make_shared<MdsMatrixBabyBear16<BabyBear>>();
auto poseidon = create_poseidon<BabyBear, BabyBear,
                                MdsPermutation<BabyBear, 16>,
                                16, 7>(...);
```

#### Goldilocks (ALPHA = 7)

```cpp
auto mds = std::make_shared<MdsMatrixGoldilocks12<Goldilocks>>();
auto poseidon = create_poseidon<Goldilocks, Goldilocks,
                                MdsPermutation<Goldilocks, 12>,
                                12, 7>(...);
```

#### Mersenne31 (ALPHA = 5)

```cpp
auto mds = std::make_shared<MdsMatrixMersenne3116<Mersenne31>>();
auto poseidon = create_poseidon<Mersenne31, Mersenne31,
                                MdsPermutation<Mersenne31, 16>,
                                16, 5>(...);
```

## Building

### Requirements

- C++17 or later
- CMake 3.15 or later
- Compiler with template support (GCC/Clang/MSVC)

### Build Instructions

```bash
cd cpp/cpp_poseidon
mkdir build && cd build
cmake ..
cmake --build .
```

### Run Examples

```bash
./bin/baby_bear_poseidon
./bin/goldilocks_poseidon
./bin/mersenne31_poseidon
```

### Build Options

- `BUILD_EXAMPLES=ON/OFF`: Build example programs (default: ON)
- `ENABLE_WARNINGS=ON/OFF`: Enable compiler warnings (default: ON)

## Field Requirements

To use this library with your field type, it must provide:

1. **Basic arithmetic**: `+`, `*`, `+=`, `*=`
2. **Zero element**: `static F zero()`
3. **Field conversion**: `static F from_canonical_u64(uint64_t)`
4. **Power map**: `template<uint64_t D> F injective_exp_n() const`
5. **Field prime**: `static constexpr uint64_t PRIME`

## Round Numbers

The security of Poseidon depends on choosing appropriate round numbers. The implementation uses:

- **Full rounds**: 2 × `half_num_full_rounds` (typically 8 total)
- **Partial rounds**: `num_partial_rounds` (typically 22-42 depending on field and width)

Typical configurations:

| Field | Width | ALPHA | Full Rounds | Partial Rounds |
|-------|-------|-------|-------------|----------------|
| BabyBear | 16 | 7 | 8 | 13-20 |
| BabyBear | 24 | 7 | 8 | 21-23 |
| Goldilocks | 8 | 7 | 8 | 17-41 |
| Goldilocks | 12 | 7 | 8 | 18-42 |
| Goldilocks | 16 | 7 | 8 | 18-42 |
| Mersenne31 | 16 | 5 | 8 | 13-20 |
| Mersenne31 | 32 | 5 | 8 | 21-23 |

**Note**: These are example values. For production use, perform proper security analysis based on the Poseidon paper.

## Round Constant Generation

Round constants should be generated using a secure method. Common approaches:

1. **From nothing-up-my-sleeve numbers**: Use digits of π, e, or field-specific constants
2. **From a hash function**: Hash a seed value with a counter
3. **From a PRNG**: Use a cryptographically secure PRNG with a fixed seed

Example (for testing only):

```cpp
std::mt19937_64 rng(seed);
std::vector<F> constants;
for (size_t i = 0; i < num_constants; ++i) {
    uint64_t val = rng() % F::PRIME;
    constants.push_back(F::from_canonical_u64(val));
}
```

**Warning**: The above is NOT cryptographically secure. For production, use proper constant generation as specified in the Poseidon paper.

## Architecture

### Round Structure

```
Input State
    |
    v
[Full Rounds × half_num_full_rounds]
    | (Add constants → S-box all → MDS)
    v
[Partial Rounds × num_partial_rounds]
    | (Add constants → S-box first → MDS)
    v
[Full Rounds × half_num_full_rounds]
    | (Add constants → S-box all → MDS)
    v
Output State
```

### S-box Layer

- **Full rounds**: Apply x^ALPHA to all WIDTH elements
- **Partial rounds**: Apply x^ALPHA to first element only

The exponent ALPHA must be coprime to (field_order - 1) to ensure injectivity.

### MDS Matrix

The MDS (Maximum Distance Separable) matrix provides diffusion. This implementation uses circulant matrices defined by their first row:

- Each row is a cyclic shift of the previous row
- Matrix-vector multiplication in O(WIDTH²) time
- For larger widths, FFT-based implementations are more efficient

## Security Considerations

This implementation follows the security analysis from the Poseidon paper. Key properties:

1. **Statistical security**: Provided by sufficient rounds
2. **Algebraic security**: Resistance to interpolation attacks
3. **Gröbner basis security**: Complexity of solving the system
4. **Differential/linear security**: MDS matrix properties

**Important**: Always verify that your configuration provides adequate security for your use case.

## Performance Notes

This is a generic implementation prioritizing correctness and clarity. For production:

1. Consider field-specific optimizations
2. Use FFT-based MDS matrix multiplication for large widths
3. Implement packed field operations where available
4. Pre-compute and cache round constants

## Comparison with Poseidon2

| Feature | Poseidon (this lib) | Poseidon2 |
|---------|-------------------|-----------|
| MDS Matrix | Generic circulant | Optimized 4×4 |
| Performance | Good | Better |
| Round Structure | Full-Partial-Full | External-Internal-External |
| Security Analysis | Well-established | Newer, also well-studied |
| Use Case | Maximum compatibility | New implementations |

## License

This implementation follows the same license as the original Plonky3 project (MIT/Apache-2.0).

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- Headers are properly documented
- Examples compile and run correctly
- Add tests for new features

## Acknowledgments

This C++ port is based on the Rust implementation from the [Plonky3 project](https://github.com/Plonky3/Plonky3). Special thanks to the original authors and the ZK proof community.

## See Also

- [cpp_field](../cpp_field/README.md) - Field implementations
- [cpp_poseidon2](../cpp_poseidon2/README.md) - Poseidon2 implementation
- [Plonky3 Documentation](https://github.com/Plonky3/Plonky3)











# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A C++/CUDA re-implementation of [Plonky3](https://github.com/Plonky3/Plonky3) (a Rust ZK proof toolkit) targeting NVIDIA GPUs. The `src/` directory contains the C++/CUDA implementations; the `plonky3/` subdirectory is the original Rust workspace used as the reference for algorithm correctness.

## Build Commands

The project is a single CMake project rooted at `src/CMakeLists.txt`. Requires CMake 3.18+ and C++17.

### CPU-only build
```bash
cd src
scripts/build_cpp.sh          # pass "clean" to wipe build dir first
# Or manually:
mkdir build_cpp && cd build_cpp
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build . --parallel
```

### CUDA build
```bash
cd src
scripts/build_cuda.sh          # auto-detects CUDA via CUDA_HOME, nvcc, or common paths
# Or manually:
mkdir build_cuda && cd build_cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --parallel
```

### CMake options
| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_CUDA` | `ON` (in CMakeLists.txt) | Enable CUDA GPU support |
| `BUILD_TESTS` | `ON` | Build Google Test suite |
| `BUILD_EXAMPLES` | `ON` | Build example programs |
| `BUILD_BENCHMARKS` | `ON` | Build Google Benchmark suite |
| `BUILD_CUDA_EXAMPLES` | `ON` | Build CUDA example programs |
| `ENABLE_WARNINGS` | `ON` | Enable compiler warnings |

### Running tests

Tests use Google Test (auto-fetched via FetchContent v1.15.2) and support ctest.

```bash
# Run all tests
cd src/build_cpp   # or build_cuda
ctest --output-on-failure

# Run a single test by name (regex match)
ctest --output-on-failure -R test_baby_bear

# Run tests via script
cd src
scripts/build_and_run_tests_cpp.sh   # CPU
scripts/build_and_run_tests_cuda.sh  # CUDA
```

### Running benchmarks

Benchmarks use Google Benchmark (auto-fetched via FetchContent v1.9.1). Benchmark targets are compiled with `-O3 -march=native`.

```bash
cd src
scripts/build_and_run_benches_cpp.sh
```

### Rust reference implementation
```bash
cd plonky3
cargo build
cargo test
cargo test -p p3-poseidon2   # run a single crate's tests
```

## Architecture

### Directory Structure

All C++ code lives under `src/` in a flat, category-based layout:

- **`src/include/`** — All header files (header-only library). A single `plonky3` INTERFACE CMake target provides access to all headers.
  - `p3_util/` subdirectory for utility headers (`util.hpp`, `linear_map.hpp`)
  - All other headers are flat in `include/`

- **`src/src/`** — CUDA library source files
  - `matrix_cuda.cu`: CUDA matrix operations (builds `matrix_cuda` static library)

- **`src/tests/`** — Google Test files (naming: `test_<domain>.cpp` or `<domain>_test.cpp`)

- **`src/examples/`** — Example programs (CPU `.cpp` and CUDA `.cu`)

- **`src/benches/`** — Benchmark programs using Google Benchmark

- **`src/scripts/`** — Build scripts

- **`src/cmake/`** — CMake config templates

### Domain Organization

Headers in `src/include/` map to these domains (each with its own namespace):

| Domain | Namespace | Key headers |
|--------|-----------|-------------|
| Field arithmetic | `p3_field` | `field.hpp` (CRTP base), `baby_bear.hpp`, `goldilocks.hpp`, `mersenne31.hpp`, `koala_bear.hpp`, `bn254.hpp`, `extension_field.hpp` |
| CUDA compatibility | — | `cuda_compat.hpp` (`P3_HOST_DEVICE`, `P3_CUDA_ENABLED`, portable `uint128_t`) |
| Poseidon2 hash | `poseidon2` | `poseidon2.hpp`, `poseidon2_cuda.hpp`, `poseidon2_bn254.hpp`, `poseidon2_koalabear.hpp`, `external.hpp`, `internal.hpp`, `generic.hpp`, `round_numbers.hpp` |
| Poseidon hash | `p3_poseidon` | `poseidon.hpp`, `mds_matrix.hpp` |
| Matrix | `p3_matrix` | `matrix.hpp`, `dense_matrix.hpp`, `matrix_cuda.hpp`, `matrix_views.hpp` (row/col slices, stacking), `matrix_extension.hpp` (zero-copy flatten/unflatten views for extension fields) |
| DFT/NTT | `p3_dft` | `radix2_dit.hpp`, `radix2_dit_parallel.hpp`, `radix2_bowers.hpp`, `ntt_cuda.hpp`, `butterflies.hpp`, `mersenne31_dft.hpp`, `naive_dft.hpp`, `traits.hpp` |
| Interpolation | — | `interpolation.hpp` (barycentric Lagrange on multiplicative cosets) |
| FRI proof system | — | `fri_prover.hpp`, `fri_verifier.hpp`, `two_adic_fri_pcs.hpp`, `fri_fold_cuda.hpp`, `fri_folding.hpp`, `fri_params.hpp`, `fri_proof.hpp`, `fri_merkle_tree_mmcs.hpp`, `fri_extension_mmcs.hpp` |
| Merkle tree | `p3_merkle_tree` | `merkle_tree.hpp`, `merkle_tree_cuda.hpp`, `merkle_tree_mmcs.hpp`, `merkle_cap.hpp` |
| Commit/PCS | `p3_merkle` | `mmcs.hpp`, `extension_mmcs.hpp`, `pcs.hpp`, `domain.hpp` |
| Challenger | — | `duplex_challenger.hpp`, `hash_challenger.hpp`, `multi_field_challenger.hpp`, `serializing_challenger.hpp`, `challenger_traits.hpp` |
| Symmetric crypto | — | `hash.hpp`, `padding_free_sponge.hpp`, `truncated_permutation.hpp`, `compression_from_hasher.hpp`, `multi_field_sponge.hpp`, `serializing_hasher.hpp`, `keccak.hpp`, `sha256.hpp`, `monolith.hpp` |
| AIR framework | — | `air.hpp` (Air/AirBuilder interfaces, FilteredAirBuilder), `symbolic_expression.hpp` (symbolic AST for constraints), `check_constraints.hpp` (debug constraint checker), `virtual_column.hpp` (derived trace columns), `constraint_folder.hpp` (prover/verifier folders) |
| UniSTARK | — | `stark_config.hpp` (binds field/PCS/challenger), `stark_prover.hpp`, `stark_verifier.hpp`, `stark_proof.hpp` — test in `tests/test_uni_stark.cpp` |
| Utilities | `p3_util`, `p3_matrix` | `p3_util/util.hpp`, `p3_util/linear_map.hpp`, `util.hpp` (matrix utils, re-exports `p3_util`) |

### Key Design Patterns

- **CRTP** throughout field arithmetic: `PrimeField<Derived>` provides zero-overhead polymorphism, avoiding virtual functions (which are problematic on GPUs).
- **`__host__ __device__`** on all field methods via `P3_HOST_DEVICE` macro, enabling the same `.hpp` to compile for both CPU (`.cpp`) and GPU (`.cu`) translation units.
- **No linking required**: the `plonky3` CMake target is `INTERFACE` (header-only) — just link against it.
- **CUDA architecture targets**: defaults to `70;75;80;86;89;90` (Volta+). Override with `-DCMAKE_CUDA_ARCHITECTURES=...`.
- On GPU, use `zero_val()`/`one_val()`/`two_val()`/`neg_one_val()`/`int_val(n)` factory methods instead of the static constants `ZERO`/`ONE`/etc. (which are CPU-only due to CUDA limitations with static device constants).
- **Portable 128-bit arithmetic**: CUDA lacks `__uint128_t`, so `p3_field::cuda_util` provides a custom `uint128_t` struct using `__umul64hi()` intrinsic. Goldilocks field depends on this.

### Plonky3 Rust Reference

The `plonky3/` subdirectory is a full Rust workspace (the upstream Plonky3 crates). It is the source of truth for algorithm correctness. Key crates to reference: `p3-field`, `p3-poseidon2`, `p3-matrix`, `p3-baby-bear`, `p3-goldilocks`, `p3-mersenne-31`.

### Development Workflow

The `ai_tasks/` directory contains numbered task specifications (01–35) that define the porting plan from Rust to C++/CUDA. Each task file includes dependencies, Rust reference files, C++ target files, and acceptance criteria. Completed tasks also have `*_summary.md` files. Tasks progress from CPU-only foundations (phases 1–6) to CUDA-accelerated versions (tasks 13–15: NTT, Merkle tree, FRI fold), then additional fields and primitives (16+: KoalaBear, BN254, Keccak, SHA256, Monolith, etc.), and advanced protocol components (26+: AIR framework, UniSTARK, lookup arguments).

### Adding New Tests

Tests use the `add_plonky3_test(target_name source_files...)` CMake function defined in `src/tests/CMakeLists.txt`. This auto-links `plonky3` and `GTest::gtest_main`. For CUDA tests (`.cu` files):
- Add inside the `if(ENABLE_CUDA)` block
- Link `CUDA::cudart` if using CUDA runtime APIs
- Link `matrix_cuda` if using CUDA matrix operations (e.g., `matrix_cuda_tests`)
- Add `--expt-relaxed-constexpr` compile option if the test uses constexpr in device code

Test target names map to ctest names via `gtest_discover_tests()`. Use `ctest -R <target_name>` to run a specific test target (e.g., `ctest -R bn254_tests`).

### Code Style

No `.clang-format` file exists. No linter is configured. CUDA `.cu` files that need constexpr in device code require the `--expt-relaxed-constexpr` nvcc flag (already set for relevant targets in CMake).

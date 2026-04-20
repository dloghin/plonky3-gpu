# AGENTS.md

## Build & Test Commands

### CPU Build
- **Build**: `cd src && scripts/build_cpp.sh`
- **Test (ctest)**: `cd src/build_cpp && ctest --output-on-failure`
- **Test (script)**: `cd src && scripts/build_and_run_tests_cpp.sh`
- **Benchmark**: `cd src && scripts/build_and_run_benches_cpp.sh`

### CUDA Build
- **Build**: `cd src && scripts/build_cuda.sh`
- **Test (ctest)**: `cd src/build_cuda && ctest --output-on-failure`
- **Test (script)**: `cd src && scripts/build_and_run_tests_cuda.sh`

### Rust Reference (Source of Truth)
- **Workspace**: `cd plonky3 && cargo build && cargo test`
- **Single crate**: `cd plonky3 && cargo test -p <crate>`

## Key Implementation Details

- **CUDA + CPU portability**: Use `P3_HOST_DEVICE` for methods shared across host/device.
- **GPU constants**: Use `zero_val()`, `one_val()`, `two_val()`, `neg_one_val()`, and `int_val(n)` instead of static constants like `ZERO`/`ONE`.
- **128-bit arithmetic on CUDA**: Use `p3_field::cuda_util::uint128_t` (`__uint128_t` is not available in device code).
- **Field polymorphism**: Use CRTP (`PrimeField<Derived>`) instead of virtual dispatch.
- **Header-only library**: `plonky3` CMake target is `INTERFACE`; no compiled core library to link.

## Architecture

- **`src/include/`**: Header-only C++ library (field, hash, matrix, FRI, AIR, etc.).
- **`src/src/`**: CUDA implementation files (for example, `matrix_cuda.cu`).
- **`src/tests/`**: Google Test suites.
- **`src/examples/`**: CPU and CUDA examples.
- **`src/benches/`**: Google Benchmark programs.
- **`plonky3/`**: Upstream Rust reference implementation (algorithmic source of truth).
- **`ai_tasks/`**: Numbered task specs and progress summaries for the porting plan.

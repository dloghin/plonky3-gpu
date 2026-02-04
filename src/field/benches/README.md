# BabyBear Field Benchmarks

This directory contains performance benchmarks for the BabyBear prime field implementation using [Google Benchmark](https://github.com/google/benchmark).

## Building

The benchmarks are automatically built when you build the project with the `BUILD_BENCHMARKS` option enabled (default is ON):

```bash
cd src/field
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target bench_field
```

## Running

After building, you can run the benchmarks:

```bash
./benches/bench_field
```

Or use the convenient make target:

```bash
make run_benchmarks
```

### Command Line Options

Google Benchmark provides many useful command-line options:

```bash
# Run specific benchmarks matching a regex pattern
./benches/bench_field --benchmark_filter="Add.*"

# Output results in different formats (console, json, csv)
./benches/bench_field --benchmark_format=json --benchmark_out=results.json

# Control the minimum benchmark time
./benches/bench_field --benchmark_min_time=5.0

# Get help with all options
./benches/bench_field --help
```

## Benchmark Categories

The benchmark suite includes the following categories:

### Basic Operations
- **Inversion** (`BM_BabyBear_Inv`): Field element inversion
- **Iter Sum**: Summing arrays of different sizes (4, 8, 12 elements)
- **Dot Product**: Dot products of varying sizes (2, 3, 4, 5, 6, 7, 8, 9, 16, 64 elements)

### Latency Tests (Dependent Operations)
These measure the latency of a single operation by creating chains of dependent operations:
- **Add Latency** (`BM_BabyBear_AddLatency_10000`): 10,000 dependent additions
- **Sub Latency** (`BM_BabyBear_SubLatency_10000`): 10,000 dependent subtractions
- **Mul Latency** (`BM_BabyBear_MulLatency_10000`): 10,000 dependent multiplications

### Throughput Tests (Independent Operations)
These measure maximum throughput by performing 10 independent operation chains in parallel:
- **Add Throughput** (`BM_BabyBear_AddThroughput_1000`): 1,000 rounds of 10 parallel additions
- **Sub Throughput** (`BM_BabyBear_SubThroughput_1000`): 1,000 rounds of 10 parallel subtractions
- **Mul Throughput** (`BM_BabyBear_MulThroughput_1000`): 1,000 rounds of 10 parallel multiplications

### Exponentiation
- **7th Root** (`BM_BabyBear_7thRoot`): Computing x^1725656503
- **Constant Exponentiation**: Optimized constant-time exponentiation for powers 3, 5, and 7

## Interpreting Results

Example output:

```
---------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
BM_BabyBear_Inv                   145 ns          145 ns      4823449
BM_BabyBear_AddLatency_10000     8234 ns         8233 ns        85023
BM_BabyBear_AddThroughput_1000   2456 ns         2456 ns       285123
```

- **Time/CPU**: Average time per iteration
- **Iterations**: Number of times the benchmark was run to get stable results

### Latency vs Throughput

The latency benchmarks measure the time for a single operation in a dependency chain, while throughput benchmarks measure how many operations can be completed per unit time when multiple independent operations run in parallel.

For modern CPUs with instruction-level parallelism, throughput is often much higher than what latency alone would suggest.

## Comparison with Rust Benchmarks

This C++ benchmark suite mirrors the Rust `bench_field.rs` implementation, allowing for direct performance comparisons between the Rust and C++ implementations of the BabyBear field.

## Requirements

- CMake 3.15 or higher
- C++17 compatible compiler
- Google Benchmark (automatically fetched if not found)

## Notes

- Benchmarks are compiled with optimizations (`-O3 -march=native` on GCC/Clang)
- Random number generation uses a fixed seed for reproducibility
- Each benchmark uses `DoNotOptimize` to prevent compiler optimizations from eliminating the code


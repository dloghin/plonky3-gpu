#include <benchmark/benchmark.h>
#include "baby_bear.hpp"
#include <random>
#include <vector>

using namespace p3_field;

// Random number generator for creating test data
static std::mt19937_64 rng(12345);

// Helper function to generate random BabyBear field elements
BabyBear random_field_element() {
    std::uniform_int_distribution<uint32_t> dist(0, BabyBear::PRIME - 1);
    return BabyBear(dist(rng));
}

// ============================================================================
// Inversion Benchmark
// ============================================================================
static void BM_BabyBear_Inv(benchmark::State& state) {
    BabyBear x = random_field_element();
    for (auto _ : state) {
        benchmark::DoNotOptimize(x.inv());
    }
}
BENCHMARK(BM_BabyBear_Inv);

// ============================================================================
// Iter Sum Benchmarks (sum of array elements)
// ============================================================================
template<size_t N, size_t REPS>
static void BM_BabyBear_IterSum(benchmark::State& state) {
    std::vector<BabyBear> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = random_field_element();
    }

    for (auto _ : state) {
        for (size_t rep = 0; rep < REPS; ++rep) {
            BabyBear sum = BabyBear::ZERO;
            for (size_t i = 0; i < N; ++i) {
                sum += data[i];
            }
            benchmark::DoNotOptimize(sum);
        }
    }
}

static void BM_BabyBear_IterSum_4(benchmark::State& state) {
    BM_BabyBear_IterSum<4, 1000>(state);
}
BENCHMARK(BM_BabyBear_IterSum_4);

static void BM_BabyBear_IterSum_8(benchmark::State& state) {
    BM_BabyBear_IterSum<8, 1000>(state);
}
BENCHMARK(BM_BabyBear_IterSum_8);

static void BM_BabyBear_IterSum_12(benchmark::State& state) {
    BM_BabyBear_IterSum<12, 1000>(state);
}
BENCHMARK(BM_BabyBear_IterSum_12);

// ============================================================================
// Dot Product Benchmarks
// ============================================================================
template<size_t N>
static void BM_BabyBear_DotProduct(benchmark::State& state) {
    std::vector<BabyBear> a(N);
    std::vector<BabyBear> b(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = random_field_element();
        b[i] = random_field_element();
    }

    for (auto _ : state) {
        BabyBear result = BabyBear::ZERO;
        for (size_t i = 0; i < N; ++i) {
            result += a[i] * b[i];
        }
        benchmark::DoNotOptimize(result);
    }
}

static void BM_BabyBear_Dot_2(benchmark::State& state) {
    BM_BabyBear_DotProduct<2>(state);
}
BENCHMARK(BM_BabyBear_Dot_2);

static void BM_BabyBear_Dot_3(benchmark::State& state) {
    BM_BabyBear_DotProduct<3>(state);
}
BENCHMARK(BM_BabyBear_Dot_3);

static void BM_BabyBear_Dot_4(benchmark::State& state) {
    BM_BabyBear_DotProduct<4>(state);
}
BENCHMARK(BM_BabyBear_Dot_4);

static void BM_BabyBear_Dot_5(benchmark::State& state) {
    BM_BabyBear_DotProduct<5>(state);
}
BENCHMARK(BM_BabyBear_Dot_5);

static void BM_BabyBear_Dot_6(benchmark::State& state) {
    BM_BabyBear_DotProduct<6>(state);
}
BENCHMARK(BM_BabyBear_Dot_6);

static void BM_BabyBear_Dot_7(benchmark::State& state) {
    BM_BabyBear_DotProduct<7>(state);
}
BENCHMARK(BM_BabyBear_Dot_7);

static void BM_BabyBear_Dot_8(benchmark::State& state) {
    BM_BabyBear_DotProduct<8>(state);
}
BENCHMARK(BM_BabyBear_Dot_8);

static void BM_BabyBear_Dot_9(benchmark::State& state) {
    BM_BabyBear_DotProduct<9>(state);
}
BENCHMARK(BM_BabyBear_Dot_9);

static void BM_BabyBear_Dot_16(benchmark::State& state) {
    BM_BabyBear_DotProduct<16>(state);
}
BENCHMARK(BM_BabyBear_Dot_16);

static void BM_BabyBear_Dot_64(benchmark::State& state) {
    BM_BabyBear_DotProduct<64>(state);
}
BENCHMARK(BM_BabyBear_Dot_64);

// ============================================================================
// Addition Latency Benchmark (chain of dependent additions)
// ============================================================================
template<size_t REPS>
static void BM_BabyBear_AddLatency(benchmark::State& state) {
    BabyBear x = random_field_element();
    BabyBear y = random_field_element();

    for (auto _ : state) {
        BabyBear result = x;
        for (size_t i = 0; i < REPS; ++i) {
            result = result + y;
        }
        benchmark::DoNotOptimize(result);
    }
}

static void BM_BabyBear_AddLatency_10000(benchmark::State& state) {
    BM_BabyBear_AddLatency<10000>(state);
}
BENCHMARK(BM_BabyBear_AddLatency_10000);

// ============================================================================
// Addition Throughput Benchmark (10 independent addition chains)
// ============================================================================
template<size_t REPS>
static void BM_BabyBear_AddThroughput(benchmark::State& state) {
    std::vector<BabyBear> xs(10);
    std::vector<BabyBear> ys(10);

    for (size_t i = 0; i < 10; ++i) {
        xs[i] = random_field_element();
        ys[i] = random_field_element();
    }

    for (auto _ : state) {
        for (size_t rep = 0; rep < REPS; ++rep) {
            xs[0] = xs[0] + ys[0];
            xs[1] = xs[1] + ys[1];
            xs[2] = xs[2] + ys[2];
            xs[3] = xs[3] + ys[3];
            xs[4] = xs[4] + ys[4];
            xs[5] = xs[5] + ys[5];
            xs[6] = xs[6] + ys[6];
            xs[7] = xs[7] + ys[7];
            xs[8] = xs[8] + ys[8];
            xs[9] = xs[9] + ys[9];
        }
        benchmark::DoNotOptimize(xs.data());
    }
}

static void BM_BabyBear_AddThroughput_1000(benchmark::State& state) {
    BM_BabyBear_AddThroughput<1000>(state);
}
BENCHMARK(BM_BabyBear_AddThroughput_1000);

// ============================================================================
// Subtraction Latency Benchmark
// ============================================================================
template<size_t REPS>
static void BM_BabyBear_SubLatency(benchmark::State& state) {
    BabyBear x = random_field_element();
    BabyBear y = random_field_element();

    for (auto _ : state) {
        BabyBear result = x;
        for (size_t i = 0; i < REPS; ++i) {
            result = result - y;
        }
        benchmark::DoNotOptimize(result);
    }
}

static void BM_BabyBear_SubLatency_10000(benchmark::State& state) {
    BM_BabyBear_SubLatency<10000>(state);
}
BENCHMARK(BM_BabyBear_SubLatency_10000);

// ============================================================================
// Subtraction Throughput Benchmark
// ============================================================================
template<size_t REPS>
static void BM_BabyBear_SubThroughput(benchmark::State& state) {
    std::vector<BabyBear> xs(10);
    std::vector<BabyBear> ys(10);

    for (size_t i = 0; i < 10; ++i) {
        xs[i] = random_field_element();
        ys[i] = random_field_element();
    }

    for (auto _ : state) {
        for (size_t rep = 0; rep < REPS; ++rep) {
            xs[0] = xs[0] - ys[0];
            xs[1] = xs[1] - ys[1];
            xs[2] = xs[2] - ys[2];
            xs[3] = xs[3] - ys[3];
            xs[4] = xs[4] - ys[4];
            xs[5] = xs[5] - ys[5];
            xs[6] = xs[6] - ys[6];
            xs[7] = xs[7] - ys[7];
            xs[8] = xs[8] - ys[8];
            xs[9] = xs[9] - ys[9];
        }
        benchmark::DoNotOptimize(xs.data());
    }
}

static void BM_BabyBear_SubThroughput_1000(benchmark::State& state) {
    BM_BabyBear_SubThroughput<1000>(state);
}
BENCHMARK(BM_BabyBear_SubThroughput_1000);

// ============================================================================
// Multiplication Latency Benchmark
// ============================================================================
template<size_t REPS>
static void BM_BabyBear_MulLatency(benchmark::State& state) {
    BabyBear x = random_field_element();
    BabyBear y = random_field_element();

    for (auto _ : state) {
        BabyBear result = x;
        for (size_t i = 0; i < REPS; ++i) {
            result = result * y;
        }
        benchmark::DoNotOptimize(result);
    }
}

static void BM_BabyBear_MulLatency_10000(benchmark::State& state) {
    BM_BabyBear_MulLatency<10000>(state);
}
BENCHMARK(BM_BabyBear_MulLatency_10000);

// ============================================================================
// Multiplication Throughput Benchmark
// ============================================================================
template<size_t REPS>
static void BM_BabyBear_MulThroughput(benchmark::State& state) {
    std::vector<BabyBear> xs(10);
    std::vector<BabyBear> ys(10);

    for (size_t i = 0; i < 10; ++i) {
        xs[i] = random_field_element();
        ys[i] = random_field_element();
    }

    for (auto _ : state) {
        for (size_t rep = 0; rep < REPS; ++rep) {
            xs[0] = xs[0] * ys[0];
            xs[1] = xs[1] * ys[1];
            xs[2] = xs[2] * ys[2];
            xs[3] = xs[3] * ys[3];
            xs[4] = xs[4] * ys[4];
            xs[5] = xs[5] * ys[5];
            xs[6] = xs[6] * ys[6];
            xs[7] = xs[7] * ys[7];
            xs[8] = xs[8] * ys[8];
            xs[9] = xs[9] * ys[9];
        }
        benchmark::DoNotOptimize(xs.data());
    }
}

static void BM_BabyBear_MulThroughput_1000(benchmark::State& state) {
    BM_BabyBear_MulThroughput<1000>(state);
}
BENCHMARK(BM_BabyBear_MulThroughput_1000);

// ============================================================================
// 7th Root Benchmark (from original Rust code)
// ============================================================================
static void BM_BabyBear_7thRoot(benchmark::State& state) {
    BabyBear x = random_field_element();

    for (auto _ : state) {
        state.PauseTiming();
        x = random_field_element();
        state.ResumeTiming();

        BabyBear result = x.exp_u64(1725656503);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_BabyBear_7thRoot);

// ============================================================================
// Constant Exponentiation Benchmarks
// ============================================================================
template<uint64_t EXP, size_t REPS>
static void BM_BabyBear_ExpConst(benchmark::State& state) {
    BabyBear x = random_field_element();

    for (auto _ : state) {
        for (size_t rep = 0; rep < REPS; ++rep) {
            BabyBear result = x.exp_const_u64<EXP>();
            benchmark::DoNotOptimize(result);
        }
    }
}

static void BM_BabyBear_ExpConst3(benchmark::State& state) {
    BM_BabyBear_ExpConst<3, 1000>(state);
}
BENCHMARK(BM_BabyBear_ExpConst3);

static void BM_BabyBear_ExpConst5(benchmark::State& state) {
    BM_BabyBear_ExpConst<5, 1000>(state);
}
BENCHMARK(BM_BabyBear_ExpConst5);

static void BM_BabyBear_ExpConst7(benchmark::State& state) {
    BM_BabyBear_ExpConst<7, 1000>(state);
}
BENCHMARK(BM_BabyBear_ExpConst7);

// Run the benchmarks
BENCHMARK_MAIN();


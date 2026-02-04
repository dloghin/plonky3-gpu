# Benchmarks using Google Benchmark
cmake_minimum_required(VERSION 3.15)

# Find or fetch Google Benchmark
find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found, fetching from GitHub...")
    include(FetchContent)
    
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1  # Use a stable release
    )
    
    # Configure benchmark options
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable gtest tests in benchmark" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Download dependencies" FORCE)
    
    # Force release build for benchmark library
    set(CMAKE_BUILD_TYPE_SAVED ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE Release)
    
    FetchContent_MakeAvailable(googlebenchmark)
    
    # Restore original build type
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_SAVED})
endif()

# BabyBear field benchmarks
add_executable(bench_field bench_field.cpp)
target_link_libraries(bench_field PRIVATE field benchmark::benchmark)
target_include_directories(bench_field PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Set compiler optimizations for benchmarks
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_field PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(bench_field PRIVATE /O2)
endif()

# Add benchmark as a custom target for easy running
add_custom_target(run_benchmarks
    COMMAND bench_field
    DEPENDS bench_field
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running BabyBear field benchmarks..."
)


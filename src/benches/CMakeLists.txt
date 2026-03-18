cmake_minimum_required(VERSION 3.15)

find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found, fetching from GitHub...")
    include(FetchContent)

    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
    )

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable gtest tests in benchmark" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Download dependencies" FORCE)

    set(CMAKE_BUILD_TYPE_SAVED ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE Release)
    FetchContent_MakeAvailable(googlebenchmark)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_SAVED})
endif()

add_executable(bench_field bench_field.cpp)
target_link_libraries(bench_field PRIVATE plonky3 benchmark::benchmark)

add_executable(transpose_benchmark transpose_benchmark.cpp)
target_link_libraries(transpose_benchmark PRIVATE plonky3 benchmark::benchmark)

add_executable(columnwise_dot_product columnwise_dot_product.cpp)
target_link_libraries(columnwise_dot_product PRIVATE plonky3 benchmark::benchmark)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_field PRIVATE -O3 -march=native)
    target_compile_options(transpose_benchmark PRIVATE -O3 -march=native)
    target_compile_options(columnwise_dot_product PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(bench_field PRIVATE /O2)
    target_compile_options(transpose_benchmark PRIVATE /O2)
    target_compile_options(columnwise_dot_product PRIVATE /O2)
endif()

add_custom_target(run_benchmarks
    COMMAND bench_field
    COMMAND transpose_benchmark
    COMMAND columnwise_dot_product
    DEPENDS bench_field transpose_benchmark columnwise_dot_product
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks..."
)

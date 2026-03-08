cmake_minimum_required(VERSION 3.15)

# Find or fetch Google Benchmark
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

    FetchContent_MakeAvailable(googlebenchmark)
endif()

# Transpose benchmark
add_executable(transpose_benchmark transpose_benchmark.cpp)
target_link_libraries(transpose_benchmark PRIVATE matrix benchmark::benchmark)
target_include_directories(transpose_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Columnwise dot product benchmark
add_executable(columnwise_dot_product columnwise_dot_product.cpp)
target_link_libraries(columnwise_dot_product PRIVATE matrix benchmark::benchmark)
target_include_directories(columnwise_dot_product PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Set compiler optimizations for benchmarks
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(transpose_benchmark PRIVATE -O3 -march=native)
    target_compile_options(columnwise_dot_product PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(transpose_benchmark PRIVATE /O2)
    target_compile_options(columnwise_dot_product PRIVATE /O2)
endif()

# Add custom targets for easy running
add_custom_target(run_transpose_benchmark
    COMMAND transpose_benchmark
    DEPENDS transpose_benchmark
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running transpose benchmark..."
)

add_custom_target(run_columnwise_benchmark
    COMMAND columnwise_dot_product
    DEPENDS columnwise_dot_product
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running columnwise dot product benchmark..."
)

add_custom_target(run_all_benchmarks
    DEPENDS run_transpose_benchmark run_columnwise_benchmark
    COMMENT "Running all matrix benchmarks..."
)


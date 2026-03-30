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

add_executable(bench_fold_even_odd bench_fold_even_odd.cpp)
target_link_libraries(bench_fold_even_odd PRIVATE plonky3 benchmark::benchmark)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_field PRIVATE -O3 -march=native)
    target_compile_options(transpose_benchmark PRIVATE -O3 -march=native)
    target_compile_options(columnwise_dot_product PRIVATE -O3 -march=native)
    target_compile_options(bench_fold_even_odd PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(bench_field PRIVATE /O2)
    target_compile_options(transpose_benchmark PRIVATE /O2)
    target_compile_options(columnwise_dot_product PRIVATE /O2)
    target_compile_options(bench_fold_even_odd PRIVATE /O2)
endif()

if(ENABLE_CUDA)
    add_executable(bench_fold_even_odd_cuda bench_fold_even_odd_cuda.cu)
    target_link_libraries(bench_fold_even_odd_cuda PRIVATE plonky3 benchmark::benchmark)
    if(TARGET matrix_cuda)
        target_link_libraries(bench_fold_even_odd_cuda PRIVATE matrix_cuda)
    endif()
    target_compile_options(bench_fold_even_odd_cuda
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    set_target_properties(bench_fold_even_odd_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

set(_bench_commands
    COMMAND bench_field
    COMMAND transpose_benchmark
    COMMAND columnwise_dot_product
    COMMAND bench_fold_even_odd
)
set(_bench_deps bench_field transpose_benchmark columnwise_dot_product bench_fold_even_odd)
if(ENABLE_CUDA)
    list(APPEND _bench_commands COMMAND bench_fold_even_odd_cuda)
    list(APPEND _bench_deps bench_fold_even_odd_cuda)
endif()

add_custom_target(run_benchmarks
    ${_bench_commands}
    DEPENDS ${_bench_deps}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks..."
)

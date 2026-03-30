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

# --- CPU benchmarks (name;source pairs) ---
set(_cpu_benchmarks
    bench_field:bench_field.cpp
    transpose_benchmark:transpose_benchmark.cpp
    columnwise_dot_product:columnwise_dot_product.cpp
    bench_fold_even_odd:bench_fold_even_odd.cpp
)

set(_all_bench_targets "")
foreach(_entry IN LISTS _cpu_benchmarks)
    string(REPLACE ":" ";" _pair "${_entry}")
    list(GET _pair 0 _name)
    list(GET _pair 1 _src)
    add_executable(${_name} ${_src})
    target_link_libraries(${_name} PRIVATE plonky3 benchmark::benchmark)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${_name} PRIVATE -O3 -march=native)
    elseif(MSVC)
        target_compile_options(${_name} PRIVATE /O2)
    endif()
    list(APPEND _all_bench_targets ${_name})
endforeach()

# --- CUDA benchmarks ---
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
    list(APPEND _all_bench_targets bench_fold_even_odd_cuda)
endif()

# --- run_benchmarks target ---
set(_bench_commands "")
foreach(_t IN LISTS _all_bench_targets)
    list(APPEND _bench_commands COMMAND ${_t})
endforeach()

add_custom_target(run_benchmarks
    ${_bench_commands}
    DEPENDS ${_all_bench_targets}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks..."
)

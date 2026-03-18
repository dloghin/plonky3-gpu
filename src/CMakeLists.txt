cmake_minimum_required(VERSION 3.18)

set(P3_GPU_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/project_version.cmake")
if(EXISTS "${P3_GPU_VERSION_FILE}")
    include("${P3_GPU_VERSION_FILE}")
else()
    set(P3_GPU_PROJECT_VERSION "1.0.0")
endif()

project(plonky3_gpu VERSION ${P3_GPU_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(BUILD_CUDA_EXAMPLES "Build CUDA examples" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)

if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic)
    endif()
endif()

# Header-only library — single target for the whole project
add_library(plonky3 INTERFACE)
target_include_directories(plonky3 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(plonky3 INTERFACE cxx_std_17)

# CUDA support (optional)
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
        enable_language(CUDA)

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
        endif()

        add_library(matrix_cuda STATIC src/matrix_cuda.cu)
        target_include_directories(matrix_cuda PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        set_target_properties(matrix_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )

        message(STATUS "CUDA library enabled: matrix_cuda")
    else()
        message(STATUS "CUDA compiler not found, disabling CUDA support")
        set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
    endif()
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benches)
endif()

# Installation
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS plonky3
    EXPORT plonky3-targets
    INCLUDES DESTINATION include
)

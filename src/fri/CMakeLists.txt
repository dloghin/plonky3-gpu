cmake_minimum_required(VERSION 3.18)
project(p3_fri VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_CUDA_EXAMPLES "Build CUDA FRI fold example" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)

if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
    else()
        add_compile_options(-Wall -Wextra)
    endif()
endif()

# Header-only library target
add_library(fri INTERFACE)
target_include_directories(fri INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../field/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../matrix/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../util/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../interpolation/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../dft/include>
    $<INSTALL_INTERFACE:include>
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------
# CUDA FRI fold example (optional, only if CUDA compiler is available)
# -----------------------------------------------------------------------
if(BUILD_CUDA_EXAMPLES)
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

        set(FRI_FOLD_CUDA_INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../field/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../matrix/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../util/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../interpolation/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../dft/include
        )

        add_executable(fri_fold_cuda_example src/fri_fold_cuda.cu)
        target_include_directories(fri_fold_cuda_example
            PRIVATE ${FRI_FOLD_CUDA_INCLUDE_DIRS}
        )
        set_target_properties(fri_fold_cuda_example
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            CUDA_SEPARABLE_COMPILATION ON
        )

        message(STATUS "CUDA FRI fold example enabled")
    else()
        message(STATUS "CUDA compiler not found, skipping CUDA FRI fold example")
    endif()
endif()

# Installation
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS fri
    EXPORT fri-targets
    INCLUDES DESTINATION include
)

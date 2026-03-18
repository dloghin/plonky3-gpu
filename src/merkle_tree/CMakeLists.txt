cmake_minimum_required(VERSION 3.18)
project(p3_merkle_tree VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_CUDA_TESTS "Build CUDA Merkle tree test program" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)

if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
    else()
        add_compile_options(-Wall -Wextra)
    endif()
endif()

# Header-only CPU library target
add_library(merkle_tree INTERFACE)
target_include_directories(merkle_tree INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../field/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../matrix/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../symmetric/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../poseidon2/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include>
    $<INSTALL_INTERFACE:include>
)

# CPU tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# CUDA support: detect CUDA compiler and build the CUDA test program
if(BUILD_CUDA_TESTS)
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

        set(MERKLE_CUDA_INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../field/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../matrix/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../symmetric/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../poseidon2/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include
        )

        # CUDA Merkle tree test/example executable
        add_executable(merkle_tree_cuda_test
            src/merkle_tree_cuda.cu
        )
        target_include_directories(merkle_tree_cuda_test PRIVATE
            ${MERKLE_CUDA_INCLUDE_DIRS}
        )
        set_target_properties(merkle_tree_cuda_test
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            CUDA_SEPARABLE_COMPILATION ON
        )

        message(STATUS "CUDA Merkle tree test enabled")
    else()
        message(STATUS "CUDA compiler not found, skipping CUDA Merkle tree test")
    endif()
endif()

# Installation
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS merkle_tree
    EXPORT merkle_tree-targets
    INCLUDES DESTINATION include
)

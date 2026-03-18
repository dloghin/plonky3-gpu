cmake_minimum_required(VERSION 3.18)
project(p3_dft VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_CUDA_EXAMPLES "Build CUDA NTT example" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)

if(ENABLE_WARNINGS)
    if(NOT MSVC)
        add_compile_options(-Wall -Wextra)
    endif()
endif()

# Bring in dependencies (field, matrix, p3_util) -- they are siblings.
# Use guards to prevent duplicate target errors when subdirectories share deps.
if(NOT TARGET p3_util)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../util  util_build)
endif()
if(NOT TARGET field)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../field field_build)
endif()
if(NOT TARGET matrix)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../matrix matrix_build)
endif()

# Header-only library
add_library(p3_dft INTERFACE)
target_include_directories(p3_dft INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(p3_dft INTERFACE field matrix p3_util)

# CUDA compat header
target_include_directories(p3_dft INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include
)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------
# CUDA NTT example (optional, only if CUDA compiler is available)
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

        set(NTT_CUDA_INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../field/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../matrix/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../util/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include
        )

        add_executable(ntt_cuda_example src/ntt_cuda.cu)
        target_include_directories(ntt_cuda_example PRIVATE ${NTT_CUDA_INCLUDE_DIRS})
        set_target_properties(ntt_cuda_example
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            CUDA_SEPARABLE_COMPILATION ON
        )

        message(STATUS "CUDA NTT example enabled")
    else()
        message(STATUS "CUDA compiler not found, skipping CUDA NTT example")
    endif()
endif()

# Installation
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

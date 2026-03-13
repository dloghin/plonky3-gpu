cmake_minimum_required(VERSION 3.15)
project(cpp_matrix VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)
option(ENABLE_CUDA "Enable CUDA support" OFF)

# Compiler flags
if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic)
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Bring in p3_util (header-only, sibling directory)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../util util_build)

# Header-only library
add_library(matrix INTERFACE)
target_include_directories(matrix INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(matrix INTERFACE p3_util)

# CUDA support (optional)
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
        enable_language(CUDA)
        
        # Set CUDA standard
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        
        # CUDA architecture (adjust for your GPU)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
        endif()
        
        # CUDA library (no explicit cudart target; nvcc links it automatically)
        add_library(matrix_cuda STATIC
            src/matrix_cuda.cu
        )
        target_include_directories(matrix_cuda PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include
        )
        set_target_properties(matrix_cuda
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        
        message(STATUS "CUDA library enabled: matrix_cuda")
    else()
        message(WARNING "ENABLE_CUDA is ON but CUDA compiler not found")
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benches)
endif()

# Installation
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS matrix
    EXPORT matrix-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)


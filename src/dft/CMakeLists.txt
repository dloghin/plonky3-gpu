cmake_minimum_required(VERSION 3.15)
project(p3_dft VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)

if(ENABLE_WARNINGS)
    if(NOT MSVC)
        add_compile_options(-Wall -Wextra)
    endif()
endif()

# Bring in dependencies (field, matrix, p3_util) -- they are siblings
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../util  util_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../field field_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../matrix matrix_build)

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

# Installation
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

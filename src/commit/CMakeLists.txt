cmake_minimum_required(VERSION 3.18)

# Centralized project version shared across modules.
set(P3_GPU_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/project_version.cmake")
if(EXISTS "${P3_GPU_VERSION_FILE}")
    include("${P3_GPU_VERSION_FILE}")
else()
    # Fallback keeps standalone builds working even if the shared file is absent.
    set(P3_GPU_PROJECT_VERSION "1.0.0")
endif()

project(p3_commit VERSION ${P3_GPU_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)

if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
    else()
        add_compile_options(-Wall -Wextra)
    endif()
endif()

# Header-only library target
add_library(commit INTERFACE)
target_include_directories(commit INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../field/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../matrix/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../util/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include>
    $<INSTALL_INTERFACE:include>
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS commit
    EXPORT commit-targets
    INCLUDES DESTINATION include
)

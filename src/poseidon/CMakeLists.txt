cmake_minimum_required(VERSION 3.15)
project(cpp_poseidon VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)

# Header-only library
add_library(poseidon INTERFACE)
target_include_directories(poseidon INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link to field library and cuda library
target_include_directories(poseidon INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../field/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include>
)

# Compiler warnings
if(ENABLE_WARNINGS)
    if(MSVC)
        target_compile_options(poseidon INTERFACE /W4)
    else()
        target_compile_options(poseidon INTERFACE
            -Wall -Wextra -Wpedantic
            -Wno-unused-parameter
        )
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
install(TARGETS poseidon
    EXPORT poseidon-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT poseidon-targets
    FILE poseidon-targets.cmake
    NAMESPACE poseidon::
    DESTINATION lib/cmake/poseidon
)

# Generate config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/poseidon-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/poseidon-config.cmake
    INSTALL_DESTINATION lib/cmake/poseidon
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/poseidon-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/poseidon-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/poseidon-config-version.cmake
    DESTINATION lib/cmake/poseidon
)


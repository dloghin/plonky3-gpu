cmake_minimum_required(VERSION 3.14)

include(FetchContent)

# Download Google Test if not present
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Extension field test binary
add_executable(test_extension_field test_extension_field.cpp)

target_include_directories(test_extension_field PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/include
)

target_link_libraries(test_extension_field
    PRIVATE
    GTest::gtest_main
    field
)

set_target_properties(test_extension_field
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Register with CTest
include(GoogleTest)
gtest_discover_tests(test_extension_field)

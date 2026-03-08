cmake_minimum_required(VERSION 3.15)

# Find or fetch Google Test
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found, fetching from GitHub...")
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)
endif()

# Test executable
add_executable(matrix_tests
    test_matrix.cpp
    test_dense_matrix.cpp
    test_util.cpp
)

target_link_libraries(matrix_tests
    PRIVATE
    matrix
    GTest::gtest_main
)

target_include_directories(matrix_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Discover tests
include(GoogleTest)
gtest_discover_tests(matrix_tests)


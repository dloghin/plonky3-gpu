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

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Test executable
add_executable(interpolation_tests
    test_interpolation.cpp
)

target_link_libraries(interpolation_tests
    PRIVATE
    interpolation
    GTest::gtest_main
)

target_include_directories(interpolation_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../field/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/include
)

include(GoogleTest)
gtest_discover_tests(interpolation_tests)

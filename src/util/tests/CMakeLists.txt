cmake_minimum_required(VERSION 3.15)

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

add_executable(util_tests
    test_util.cpp
    test_linear_map.cpp
)

target_link_libraries(util_tests
    PRIVATE
    p3_util
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(util_tests)

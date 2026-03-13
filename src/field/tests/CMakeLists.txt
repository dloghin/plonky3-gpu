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

# Two-adic field tests
add_executable(field_two_adic_tests
    test_two_adic.cpp
)

target_link_libraries(field_two_adic_tests
    PRIVATE
    field
    GTest::gtest_main
)

target_include_directories(field_two_adic_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/../cuda/include
)

include(GoogleTest)
gtest_discover_tests(field_two_adic_tests)

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

add_executable(dft_tests
    test_dft.cpp
)

target_link_libraries(dft_tests
    PRIVATE
    p3_dft
    GTest::gtest_main
)

target_include_directories(dft_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/../field/include
    ${CMAKE_SOURCE_DIR}/../matrix/include
    ${CMAKE_SOURCE_DIR}/../util/include
    ${CMAKE_SOURCE_DIR}/../cuda/include
)

include(GoogleTest)
gtest_discover_tests(dft_tests)

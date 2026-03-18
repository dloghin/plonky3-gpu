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

set(DFT_TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/../field/include
    ${CMAKE_SOURCE_DIR}/../matrix/include
    ${CMAKE_SOURCE_DIR}/../util/include
    ${CMAKE_SOURCE_DIR}/../cuda/include
)

# -----------------------------------------------------------------------
# CPU DFT tests (Radix2Dit)
# -----------------------------------------------------------------------
add_executable(dft_tests test_dft.cpp)
target_link_libraries(dft_tests PRIVATE p3_dft GTest::gtest_main)
target_include_directories(dft_tests PRIVATE ${DFT_TEST_INCLUDE_DIRS})

include(GoogleTest)
gtest_discover_tests(dft_tests)

# -----------------------------------------------------------------------
# NttCuda tests (CPU-fallback path; also tests GPU when nvcc is used)
# -----------------------------------------------------------------------
add_executable(ntt_cuda_tests test_ntt_cuda.cpp)
target_link_libraries(ntt_cuda_tests PRIVATE p3_dft GTest::gtest_main)
target_include_directories(ntt_cuda_tests PRIVATE ${DFT_TEST_INCLUDE_DIRS})
gtest_discover_tests(ntt_cuda_tests)

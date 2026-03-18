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

include(GoogleTest)

# -- Field tests --
add_executable(field_two_adic_tests test_two_adic.cpp)
target_link_libraries(field_two_adic_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(field_two_adic_tests)

add_executable(field_extension_tests test_extension_field.cpp)
target_link_libraries(field_extension_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(field_extension_tests)

# -- Matrix tests --
add_executable(matrix_tests test_matrix.cpp)
target_link_libraries(matrix_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(matrix_tests)

add_executable(dense_matrix_tests test_dense_matrix.cpp)
target_link_libraries(dense_matrix_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(dense_matrix_tests)

add_executable(matrix_util_tests test_matrix_util.cpp)
target_link_libraries(matrix_util_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(matrix_util_tests)

# -- Util tests --
add_executable(util_tests test_util.cpp)
target_link_libraries(util_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(util_tests)

add_executable(linear_map_tests test_linear_map.cpp)
target_link_libraries(linear_map_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(linear_map_tests)

# -- DFT tests --
add_executable(dft_tests test_dft.cpp)
target_link_libraries(dft_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(dft_tests)

add_executable(ntt_cuda_tests test_ntt_cuda.cpp)
target_link_libraries(ntt_cuda_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(ntt_cuda_tests)

# -- FRI tests --
add_executable(fri_tests test_fri.cpp fri_test.cpp test_two_adic_fri_pcs.cpp)
target_link_libraries(fri_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(fri_tests)

# -- Merkle tree tests --
add_executable(merkle_tree_tests test_merkle_tree.cpp)
target_link_libraries(merkle_tree_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(merkle_tree_tests)

# -- Commit tests --
add_executable(commit_tests test_commit.cpp)
target_link_libraries(commit_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(commit_tests)

# -- Challenger tests --
add_executable(challenger_tests test_challenger.cpp)
target_link_libraries(challenger_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(challenger_tests)

# -- Symmetric tests --
add_executable(symmetric_tests test_symmetric.cpp)
target_link_libraries(symmetric_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(symmetric_tests)

# -- Interpolation tests --
add_executable(interpolation_tests test_interpolation.cpp)
target_link_libraries(interpolation_tests PRIVATE plonky3 GTest::gtest_main)
gtest_discover_tests(interpolation_tests)

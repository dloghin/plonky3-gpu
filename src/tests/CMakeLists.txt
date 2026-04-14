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

function(add_plonky3_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE plonky3 GTest::gtest_main)
    gtest_discover_tests(${name})
endfunction()

# -- Field tests --
add_plonky3_test(field_two_adic_tests test_two_adic.cpp)

add_plonky3_test(koala_bear_tests test_koala_bear.cpp)

add_plonky3_test(field_extension_tests test_extension_field.cpp)

# -- Matrix tests --
add_plonky3_test(matrix_tests test_matrix.cpp)

add_plonky3_test(dense_matrix_tests test_dense_matrix.cpp)

add_plonky3_test(matrix_util_tests test_matrix_util.cpp)

add_plonky3_test(matrix_views_tests test_matrix_views.cpp)

# -- Util tests --
add_plonky3_test(util_tests test_util.cpp)

add_plonky3_test(linear_map_tests test_linear_map.cpp)

# -- DFT tests --
add_plonky3_test(dft_tests test_dft.cpp)
add_plonky3_test(dft_variants_tests test_dft_variants.cpp)

add_plonky3_test(ntt_cuda_tests test_ntt_cuda.cpp)

# -- FRI tests --
add_plonky3_test(fri_tests test_fri.cpp fri_test.cpp test_two_adic_fri_pcs.cpp)

# -- Merkle tree tests --
add_plonky3_test(merkle_tree_tests test_merkle_tree.cpp)

# -- Commit tests --
add_plonky3_test(commit_tests test_commit.cpp)

# -- Challenger tests --
add_plonky3_test(challenger_tests test_challenger.cpp)
add_plonky3_test(challengers_tests test_challengers.cpp)

# -- Symmetric tests --
add_plonky3_test(symmetric_tests test_symmetric.cpp)
add_plonky3_test(symmetric_extended_tests test_symmetric_extended.cpp)

add_plonky3_test(keccak_tests test_keccak.cpp)
add_plonky3_test(monolith_tests test_monolith.cpp)
add_plonky3_test(sha256_tests test_sha256.cpp)

if(ENABLE_CUDA)
    add_plonky3_test(sha256_cuda_tests test_sha256_cuda.cu)
    add_plonky3_test(keccak_cuda_tests test_keccak_cuda.cu)
    add_plonky3_test(monolith_cuda_tests test_monolith_cuda.cu)
    add_plonky3_test(fri_fold_cuda_tests test_fri_fold_cuda.cu)
    add_plonky3_test(merkle_tree_cuda_tests test_merkle_tree_cuda.cu)
    add_plonky3_test(poseidon2_cuda_tests test_poseidon2_cuda.cu)
    add_plonky3_test(matrix_cuda_tests test_matrix_cuda.cu)
    target_compile_options(monolith_cuda_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_options(fri_fold_cuda_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(TARGET CUDA::cudart)
        target_link_libraries(sha256_cuda_tests PRIVATE CUDA::cudart)
        target_link_libraries(keccak_cuda_tests PRIVATE CUDA::cudart)
        target_link_libraries(monolith_cuda_tests PRIVATE CUDA::cudart)
        target_link_libraries(fri_fold_cuda_tests PRIVATE CUDA::cudart)
        target_link_libraries(merkle_tree_cuda_tests PRIVATE CUDA::cudart)
        target_link_libraries(poseidon2_cuda_tests PRIVATE CUDA::cudart)
        target_link_libraries(matrix_cuda_tests PRIVATE CUDA::cudart)
    endif()
    if(TARGET matrix_cuda)
        target_link_libraries(matrix_cuda_tests PRIVATE matrix_cuda)
    endif()
endif()

# -- Interpolation tests --
add_plonky3_test(interpolation_tests test_interpolation.cpp)

# -- BN254 field tests --
add_plonky3_test(bn254_tests test_bn254.cpp)

cmake_minimum_required(VERSION 3.15)

# -- CPU examples --

add_executable(field_demo field_demo.cpp)
target_link_libraries(field_demo PRIVATE plonky3)

add_executable(baby_bear_poseidon baby_bear_poseidon.cpp)
target_link_libraries(baby_bear_poseidon PRIVATE plonky3)

add_executable(goldilocks_poseidon goldilocks_poseidon.cpp)
target_link_libraries(goldilocks_poseidon PRIVATE plonky3)

add_executable(mersenne31_poseidon mersenne31_poseidon.cpp)
target_link_libraries(mersenne31_poseidon PRIVATE plonky3)

add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PRIVATE plonky3)

add_executable(field_example field_example.cpp)
target_link_libraries(field_example PRIVATE plonky3)

add_executable(baby_bear_example baby_bear_example.cpp)
target_link_libraries(baby_bear_example PRIVATE plonky3)

add_executable(goldilocks_example goldilocks_example.cpp)
target_link_libraries(goldilocks_example PRIVATE plonky3)

add_executable(matrix_demo matrix_demo.cpp)
target_link_libraries(matrix_demo PRIVATE plonky3)

set_target_properties(
    field_demo baby_bear_poseidon goldilocks_poseidon mersenne31_poseidon
    simple_example field_example baby_bear_example goldilocks_example
    matrix_demo
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# -- CUDA examples (optional) --
if(BUILD_CUDA_EXAMPLES AND ENABLE_CUDA AND CMAKE_CUDA_COMPILER)

    # Field CUDA examples
    add_executable(goldilocks_cuda_example goldilocks_cuda_example.cu)
    target_include_directories(goldilocks_cuda_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(goldilocks_cuda_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(baby_bear_cuda_example baby_bear_cuda_example.cu)
    target_include_directories(baby_bear_cuda_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(baby_bear_cuda_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(mersenne31_cuda_example mersenne31_cuda_example.cu)
    target_include_directories(mersenne31_cuda_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(mersenne31_cuda_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Poseidon2 CUDA examples
    add_executable(goldilocks_poseidon2_cuda goldilocks_poseidon2_cuda.cu)
    target_include_directories(goldilocks_poseidon2_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(goldilocks_poseidon2_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(baby_bear_poseidon2_cuda baby_bear_poseidon2_cuda.cu)
    target_include_directories(baby_bear_poseidon2_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(baby_bear_poseidon2_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(mersenne31_poseidon2_cuda mersenne31_poseidon2_cuda.cu)
    target_include_directories(mersenne31_poseidon2_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(mersenne31_poseidon2_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Matrix CUDA example
    add_executable(matrix_cuda_example matrix_cuda_example.cu)
    target_include_directories(matrix_cuda_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(matrix_cuda_example PRIVATE matrix_cuda)
    set_target_properties(matrix_cuda_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    # NTT CUDA example
    add_executable(ntt_cuda_example ntt_cuda_example.cu)
    target_include_directories(ntt_cuda_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(ntt_cuda_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    # FRI fold CUDA example
    add_executable(fri_fold_cuda_example fri_fold_cuda_example.cu)
    target_include_directories(fri_fold_cuda_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(fri_fold_cuda_example
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    set_target_properties(fri_fold_cuda_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Merkle tree CUDA example
    add_executable(merkle_tree_cuda_example merkle_tree_cuda_example.cu)
    target_include_directories(merkle_tree_cuda_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(merkle_tree_cuda_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    message(STATUS "CUDA examples enabled")
else()
    message(STATUS "CUDA examples disabled")
endif()

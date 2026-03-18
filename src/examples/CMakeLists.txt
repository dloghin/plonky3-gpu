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

    function(add_cuda_example name)
        add_executable(${name} ${name}.cu)
        target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
        if(ARGN)
            target_link_libraries(${name} PRIVATE ${ARGN})
        endif()
        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            CUDA_SEPARABLE_COMPILATION ON
        )
    endfunction()

    # Field CUDA examples
    add_cuda_example(goldilocks_cuda_example)

    add_cuda_example(baby_bear_cuda_example)

    add_cuda_example(mersenne31_cuda_example)

    # Poseidon2 CUDA examples
    add_cuda_example(goldilocks_poseidon2_cuda)

    add_cuda_example(baby_bear_poseidon2_cuda)

    add_cuda_example(mersenne31_poseidon2_cuda)

    # Matrix CUDA example
    add_cuda_example(matrix_cuda_example matrix_cuda)

    # NTT CUDA example
    add_cuda_example(ntt_cuda_example)

    # FRI fold CUDA example
    add_cuda_example(fri_fold_cuda_example)
    target_compile_options(fri_fold_cuda_example
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # Merkle tree CUDA example
    add_cuda_example(merkle_tree_cuda_example)

    message(STATUS "CUDA examples enabled")
else()
    message(STATUS "CUDA examples disabled")
endif()

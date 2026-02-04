# Add example executables

# CPU examples
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PRIVATE poseidon2)

add_executable(field_example field_example.cpp)
target_link_libraries(field_example PRIVATE poseidon2)

add_executable(baby_bear_example baby_bear_example.cpp)
target_link_libraries(baby_bear_example PRIVATE poseidon2)

add_executable(goldilocks_example goldilocks_example.cpp)
target_link_libraries(goldilocks_example PRIVATE poseidon2)

# Set output directory for CPU examples
set_target_properties(simple_example field_example baby_bear_example goldilocks_example
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# CUDA examples (optional, only if CUDA is available)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    enable_language(CUDA)

    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CUDA architecture (adjust for your GPU)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    endif()

    # Include directories for CUDA files
    set(CUDA_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../field/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include
    )

    # Goldilocks Poseidon2 CUDA example
    add_executable(goldilocks_poseidon2_cuda goldilocks_poseidon2_cuda.cu)
    target_include_directories(goldilocks_poseidon2_cuda PRIVATE ${CUDA_INCLUDE_DIRS})
    set_target_properties(goldilocks_poseidon2_cuda
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Baby Bear Poseidon2 CUDA example
    add_executable(baby_bear_poseidon2_cuda baby_bear_poseidon2_cuda.cu)
    target_include_directories(baby_bear_poseidon2_cuda PRIVATE ${CUDA_INCLUDE_DIRS})
    set_target_properties(baby_bear_poseidon2_cuda
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Mersenne31 Poseidon2 CUDA example
    add_executable(mersenne31_poseidon2_cuda mersenne31_poseidon2_cuda.cu)
    target_include_directories(mersenne31_poseidon2_cuda PRIVATE ${CUDA_INCLUDE_DIRS})
    set_target_properties(mersenne31_poseidon2_cuda
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    message(STATUS "CUDA Poseidon2 examples enabled")
else()
    message(STATUS "CUDA compiler not found, skipping CUDA Poseidon2 examples")
endif()

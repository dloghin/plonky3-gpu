# Add example executables

# CPU-only example
add_executable(field_demo field_demo.cpp)
target_link_libraries(field_demo PRIVATE field)

# Set output directory
set_target_properties(field_demo
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
    # Default to common architectures, can be overridden
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    endif()

    # Goldilocks CUDA example
    add_executable(goldilocks_cuda_example goldilocks_cuda_example.cu)
    target_include_directories(goldilocks_cuda_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(goldilocks_cuda_example
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Baby Bear CUDA example
    add_executable(baby_bear_cuda_example baby_bear_cuda_example.cu)
    target_include_directories(baby_bear_cuda_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(baby_bear_cuda_example
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Mersenne31 CUDA example
    add_executable(mersenne31_cuda_example mersenne31_cuda_example.cu)
    target_include_directories(mersenne31_cuda_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(mersenne31_cuda_example
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
    )

    message(STATUS "CUDA examples enabled: goldilocks_cuda_example, baby_bear_cuda_example, mersenne31_cuda_example")
else()
    message(STATUS "CUDA compiler not found, skipping CUDA examples")
endif()

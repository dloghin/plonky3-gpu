cmake_minimum_required(VERSION 3.15)

# CPU-only example
add_executable(matrix_demo matrix_demo.cpp)
target_link_libraries(matrix_demo PRIVATE matrix)
target_include_directories(matrix_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Set output directory
set_target_properties(matrix_demo
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# CUDA examples (optional, only if CUDA is available)
if(ENABLE_CUDA)
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
        
        # Matrix CUDA example
        add_executable(matrix_cuda_example matrix_cuda_example.cu)
        target_include_directories(matrix_cuda_example PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/include)
        # Link against matrix_cuda; nvcc will pull in cudart automatically
        target_link_libraries(matrix_cuda_example PRIVATE
            matrix_cuda)
        set_target_properties(matrix_cuda_example
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            CUDA_SEPARABLE_COMPILATION ON
        )
        
        message(STATUS "CUDA example enabled: matrix_cuda_example")
    else()
        message(STATUS "CUDA compiler not found, skipping CUDA examples")
    endif()
endif()

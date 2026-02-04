add_executable(baby_bear_poseidon baby_bear_poseidon.cpp)
target_link_libraries(baby_bear_poseidon PRIVATE poseidon)
set_target_properties(baby_bear_poseidon PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(goldilocks_poseidon goldilocks_poseidon.cpp)
target_link_libraries(goldilocks_poseidon PRIVATE poseidon)
set_target_properties(goldilocks_poseidon PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(mersenne31_poseidon mersenne31_poseidon.cpp)
target_link_libraries(mersenne31_poseidon PRIVATE poseidon)
set_target_properties(mersenne31_poseidon PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)











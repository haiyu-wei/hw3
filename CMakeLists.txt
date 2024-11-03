cmake_minimum_required(VERSION 3.29)
project(HPML_HW3 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(HPML_HW3 main.cu)

set_target_properties(HPML_HW3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

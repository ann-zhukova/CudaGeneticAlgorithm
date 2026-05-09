cmake_minimum_required(VERSION 3.27)

if (NOT DEFINED CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _env_cuda_path)
    if (EXISTS "${_env_cuda_path}/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "${_env_cuda_path}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
    else()
        set(_possible_cuda_paths
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe"
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe"
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe"
        )

        foreach(_path IN LISTS _possible_cuda_paths)
            if (EXISTS "${_path}")
                set(CMAKE_CUDA_COMPILER "${_path}" CACHE FILEPATH "CUDA compiler" FORCE)
                break()
            endif()
        endforeach()
    endif()
endif()

if (CMAKE_CUDA_COMPILER)
    message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")
endif()

project(CudaGeneticAlgorithm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(CMAKE_CUDA_SEPARATE_COMPILATION ON)
add_executable(CudaGeneticAlgorithm
        GA_CPU.h
        Polynomial.h
        main.cpp
        GA_GPU.cu
        GA_GPU.cuh)

target_include_directories(CudaGeneticAlgorithm PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(CudaGeneticAlgorithm PRIVATE CUDA::cudart)

cmake_minimum_required(VERSION 3.10)
project(vit_mnist_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Option to force CPU even if CUDA is available
option(USE_CPU_ONLY "Force CPU only compilation" OFF)

if(NOT USE_CPU_ONLY)
    find_package(CUDAToolkit QUIET)
    include(CheckLanguage)
    check_language(CUDA)
endif()

if(CMAKE_CUDA_COMPILER AND NOT USE_CPU_ONLY)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    
    add_library(vit_cuda STATIC include/cuda_kernels.cu)
    target_compile_definitions(vit_cuda PUBLIC USE_CUDA)
    set_target_properties(vit_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_executable(vit_train src/train.cpp)
    add_executable(vit_eval  src/eval.cpp)
    
    target_link_libraries(vit_train PUBLIC vit_cuda CUDA::cudart)
    target_link_libraries(vit_eval  PUBLIC vit_cuda CUDA::cudart)
    
    target_compile_definitions(vit_train PUBLIC USE_CUDA)
    target_compile_definitions(vit_eval  PUBLIC USE_CUDA)

    message(STATUS "CUDA encontrado: se compilará con aceleración GPU (USE_CUDA=1)")
else()
    add_executable(vit_train src/train.cpp)
    add_executable(vit_eval  src/eval.cpp)

    # OpenMP es opcional: acelera CPU matmul
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(vit_train PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(vit_eval  PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "Compilando para CPU con OpenMP (paralelización de matmul)")
    else()
        message(STATUS "Compilando para CPU en modo secuencial")
    endif()
endif()

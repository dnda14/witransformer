cmake_minimum_required(VERSION 3.10)
project(vit_mnist_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# OpenMP es opcional: si está disponible, acelera el matmul (la operación
# dominante). El proyecto compila y funciona igual sin él, solo más lento.
find_package(OpenMP)

add_executable(vit_train src/train.cpp)
add_executable(vit_eval  src/eval.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(vit_train PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(vit_eval  PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP encontrado: se compilará con paralelización de matmul")
else()
    message(STATUS "OpenMP no encontrado: se compilará en modo secuencial")
endif()

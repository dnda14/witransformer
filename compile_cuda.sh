#!/bin/bash
# Script para compilar el proyecto usando nvcc (CUDA)

# Configuración
NVCC=/usr/local/cuda-13.3/bin/nvcc
CXXFLAGS="-O3 -std=c++17"
INCLUDES="-Iinclude"
MACROS="-DUSE_CUDA"
ARCH="-arch=native"
SRC_CUDA="include/cuda_kernels.cu"
SRC_TRAIN="src/train.cpp"
SRC_EVAL="src/eval.cpp"

echo "Compilando CUDA kernels y train..."
$NVCC $CXXFLAGS $ARCH $MACROS $INCLUDES $SRC_CUDA $SRC_TRAIN -o vit_train_cuda

if [ $? -eq 0 ]; then
    echo "¡vit_train_cuda compilado con éxito!"
else
    echo "Error compilando train"
    exit 1
fi

echo "Compilando CUDA kernels y eval..."
$NVCC $CXXFLAGS $ARCH $MACROS $INCLUDES $SRC_CUDA $SRC_EVAL -o vit_eval_cuda

if [ $? -eq 0 ]; then
    echo "¡vit_eval_cuda compilado con éxito!"
else
    echo "Error compilando eval"
    exit 1
fi

echo "Listo. Puedes ejecutar ./vit_train_cuda y ./vit_eval_cuda"

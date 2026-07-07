#include "tensor.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>

Tensor::Tensor(int dx,int dy,int dz):dx(dx),dy(dy),dz(dz){
    valores.resize(dx * dy * dz, 0.0f);
}



Tensor Tensor::multiplicar(Tensor tensor2){
    // A: (Batch, M, K) -> (dx, dy, dz)
    // B: (Batch, K, N) -> (tensor2.dx, tensor2.dy, tensor2.dz)
    // Resultado C: (Batch, M, N) -> (dx, dy, tensor2.dz)
    
    if (dx != tensor2.dx || dz != tensor2.dy) {
        throw std::invalid_argument("Las dimensiones no coinciden para Batched MatMul.");
    }
    
    Tensor tensor3(dx, dy, tensor2.dz); // (Batch, M, N)

    for (int b = 0; b < dx; b++) {
        for (int i = 0; i < dy; i++) {
            for (int j = 0; j < tensor2.dz; j++) {
                float suma = 0.0f;
                for (int k = 0; k < dz; k++) {
                    suma += valores[pos(b, i, k)] * tensor2.valores[tensor2.pos(b, k, j)];
                }
                tensor3.valores[tensor3.pos(b, i, j)] = suma;
            }
        }
    }
    return tensor3;
}

Tensor Tensor::transponer(){
    // Útil para hacer Q * K^T en Attention.
    
    Tensor tensor3(dx, dz, dy);

    for (int b = 0; b < dx; b++) {
        for (int i = 0; i < dy; i++) {
            for (int j = 0; j < dz; j++) {
                tensor3.valores[tensor3.pos(b, j, i)] = valores[pos(b, i, j)];
            }
        }
    }
    return tensor3;
}

Tensor Tensor::sumar(Tensor tensor2){
    if (dx != tensor2.dx || dy != tensor2.dy || dz != tensor2.dz) {
            throw std::invalid_argument("Las dimensiones no coinciden para sumar.");
        }
        
        Tensor tensor3(dx, dy, dz);
        for (size_t i = 0; i < valores.size(); i++) {
            tensor3.valores[i] = valores[i] + tensor2.valores[i];
        }
        return tensor3;
}
Tensor Tensor::devidir_escalar(float e){
    if (e == 0.0f) {
            throw std::invalid_argument("El escalar a dividir no debe ser cero.");
        }
        Tensor tensor3(dx, dy, dz);
        for (size_t i = 0; i < valores.size(); i++) {
            tensor3.valores[i] = valores[i] / e;
        }
        return tensor3;
    }

Tensor Tensor::multiplicar_escalar(float e){
    Tensor tensor3(dx, dy, dz);
    for (size_t i = 0; i < valores.size(); i++) {
        tensor3.valores[i] = valores[i] * e;
    }
    return tensor3;
}

Tensor Tensor::multiplicar_elemento(Tensor tensor2){
    if (dx != tensor2.dx || dy != tensor2.dy || dz != tensor2.dz) {
        throw std::invalid_argument("Las dimensiones no coinciden para la multiplicación elemento a elemento.");
    }
    
    Tensor tensor3(dx, dy, dz);
    for (size_t i = 0; i < valores.size(); i++) {
        tensor3.valores[i] = valores[i] * tensor2.valores[i];
    }
    return tensor3;
}


Tensor Tensor::aplicar_funcion(std::function<float(float)> func){
    Tensor result(dx, dy, dz);
    for (size_t i = 0; i < valores.size(); i++) {
        result.valores[i] = func(valores[i]);
    }
    return result;
}

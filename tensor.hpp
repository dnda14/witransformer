#pragma once
#include <vector>
#include <functional>
using namespace std;

class Tensor {
    private:
    vector<float> valores;
    int dx;
    int dy;
    int dz;

    int pos(int x, int y, int z) const {
        return x * (dy * dz) + y * dz + z;
    }

    public:
    Tensor(int dx,int dy,int dz);
    Tensor multiplicar(Tensor);
    Tensor multiplicar_elemento(Tensor);
    Tensor multiplicar_escalar(float);
    Tensor transponer();
    Tensor sumar(Tensor);
    Tensor devidir_escalar(float);  
    Tensor aplicar_funcion(std::function<float(float)> func);

    // Métodos para acceder a dimensiones y datos desde funciones externas
    int get_dx() const { return dx; }
    int get_dy() const { return dy; }
    int get_dz() const { return dz; }

    float& operator()(int x, int y, int z) {
        return valores[pos(x, y, z)];
    }
    float operator()(int x, int y, int z) const {
        return valores[pos(x, y, z)];
    }

};
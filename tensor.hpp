#pragma once
#include <vector>
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
    Tensor transponer();
    Tensor sumar(Tensor);
    Tensor devidir_escalar(float); 

};
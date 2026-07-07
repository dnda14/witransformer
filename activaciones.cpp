#include "activaciones.hpp"
#include <cmath>
#include <algorithm>

Tensor softmax(const Tensor& t) {
    int dx = t.get_dx();
    int dy = t.get_dy();
    int dz = t.get_dz();
    
    Tensor result(dx, dy, dz);

    for (int b = 0; b < dx; b++) {
        for (int i = 0; i < dy; i++) {
            // 1. Encontrar el valor máximo de la fila (para estabilidad numérica)
            float max_val = t(b, i, 0);
            for (int j = 1; j < dz; j++) {
                max_val = std::max(max_val, t(b, i, j));
            }

            // 2. Calcular suma de exponenciales
            float suma_exp = 0.0f;
            for (int j = 0; j < dz; j++) {
                float e = std::exp(t(b, i, j) - max_val);
                result(b, i, j) = e;
                suma_exp += e;
            }

            // 3. Normalizar dividiendo por la suma
            for (int j = 0; j < dz; j++) {
                result(b, i, j) /= suma_exp;
            }
        }
    }
    return result;
}

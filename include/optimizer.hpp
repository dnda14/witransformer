// optimizer.hpp — Adam implementado a mano (momentos m, v + corrección de sesgo).
#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>

namespace vit {

struct Adam {
    std::vector<Tensor> params;
    float lr, beta1, beta2, eps;
    long t = 0;
    std::vector<std::vector<float>> m, v; // un vector por parámetro

    Adam(std::vector<Tensor> params_, float lr_ = 1e-3f, float beta1_ = 0.9f,
         float beta2_ = 0.999f, float eps_ = 1e-8f)
        : params(std::move(params_)), lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_) {
        for (auto& p : params) {
            m.emplace_back(p->data.size(), 0.0f);
            v.emplace_back(p->data.size(), 0.0f);
        }
    }

    void zero_grad() {
        for (auto& p : params) p->zero_grad();
    }

    // grad_scale permite dividir por el tamaño de mini-batch antes del paso.
    void step(float grad_scale = 1.0f) {
        ++t;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));
        for (size_t pi = 0; pi < params.size(); ++pi) {
            auto& p = params[pi];
            for (size_t i = 0; i < p->data.size(); ++i) {
                float g = p->grad[i] * grad_scale;
                m[pi][i] = beta1 * m[pi][i] + (1.0f - beta1) * g;
                v[pi][i] = beta2 * v[pi][i] + (1.0f - beta2) * g * g;
                float mhat = m[pi][i] / bc1;
                float vhat = v[pi][i] / bc2;
                p->data[i] -= lr * mhat / (std::sqrt(vhat) + eps);
            }
        }
    }
};

} // namespace vit

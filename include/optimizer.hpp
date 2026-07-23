// optimizer.hpp — Adam implementado a mano (momentos m, v + corrección de sesgo).
//
// Cuando USE_CUDA está definido, los momentos m y v se almacenan en GPU
// y el paso de actualización se ejecuta directamente en la GPU con un kernel,
// evitando transferir gradientes a CPU.
#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#endif

namespace vit {

struct Adam {
    std::vector<Tensor> params;
    float lr, beta1, beta2, eps;
    long t = 0;

#ifdef USE_CUDA
    std::vector<float*> d_m, d_v; // momentos en GPU (un buffer por parámetro)
#endif
    std::vector<std::vector<float>> m, v; // momentos en CPU (un vector por parámetro)

    Adam(std::vector<Tensor> params_, float lr_ = 1e-3f, float beta1_ = 0.9f,
         float beta2_ = 0.999f, float eps_ = 1e-8f)
        : params(std::move(params_)), lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_) {
        for (auto& p : params) {
            size_t sz = p->data.size();
            m.emplace_back(sz, 0.0f);
            v.emplace_back(sz, 0.0f);
#ifdef USE_CUDA
            float* dm = nullptr; cudaMalloc(&dm, sz * sizeof(float)); cudaMemset(dm, 0, sz * sizeof(float));
            float* dv = nullptr; cudaMalloc(&dv, sz * sizeof(float)); cudaMemset(dv, 0, sz * sizeof(float));
            d_m.push_back(dm);
            d_v.push_back(dv);
#endif
        }
    }

#ifdef USE_CUDA
    ~Adam() {
        for (auto* p : d_m) if (p) cudaFree(p);
        for (auto* p : d_v) if (p) cudaFree(p);
    }
#endif

    void zero_grad() {
        for (auto& p : params) p->zero_grad();
    }

    // grad_scale permite dividir por el tamaño de mini-batch antes del paso.
    void step(float grad_scale = 1.0f) {
        ++t;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));
#ifdef USE_CUDA
        for (size_t pi = 0; pi < params.size(); ++pi) {
            auto& p = params[pi];
            int sz = static_cast<int>(p->data.size());
            cuda::adam_step(p->d_data, p->d_grad, d_m[pi], d_v[pi],
                            lr, beta1, beta2, eps, bc1, bc2, grad_scale, sz);
        }
#else
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
#endif
    }
};

} // namespace vit

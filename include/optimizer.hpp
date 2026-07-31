/**
 * @file optimizer.hpp
 * @brief Implementación del optimizador AdamW (Adam con Weight Decay desacoplado).
 *
 * El optimizador AdamW actualiza los pesos de la red neuronal usando los
 * gradientes calculados por backpropagation. Combina:
 * - **Momentum (m):** Promedio móvil exponencial de los gradientes pasados.
 *   Le da "inercia" a la optimización para acelerar la convergencia.
 * - **Varianza adaptativa (v):** Promedio móvil de los gradientes al cuadrado.
 *   Escala el learning rate por parámetro, frenando cambios bruscos.
 * - **Bias correction:** Compensa el sesgo inicial de m y v (que empiezan en 0).
 * - **Weight decay desacoplado:** Penalización L2 aplicada directamente al peso
 *   (no al gradiente), lo que distingue AdamW de Adam+L2.
 *
 * Fórmula de actualización:
 *   m_t = β₁·m_{t-1} + (1-β₁)·g_t
 *   v_t = β₂·v_{t-1} + (1-β₂)·g_t²
 *   m̂_t = m_t / (1 - β₁ᵗ)
 *   v̂_t = v_t / (1 - β₂ᵗ)
 *   θ_t = θ_{t-1} - η·m̂_t/(√v̂_t + ε) - η·λ·θ_{t-1}
 *
 * Cuando `USE_CUDA` está definido, los momentos m y v se almacenan en GPU
 * y el paso de actualización se ejecuta directamente en la GPU con un kernel,
 * evitando transferir gradientes a CPU.
 */
#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#endif

namespace vit {

/**
 * @brief Optimizador AdamW: Adam con weight decay desacoplado.
 *
 * Mantiene un estado interno (momentos m y v) para cada parámetro del modelo.
 * El flujo típico de uso es:
 * 1. opt.zero_grad()  — limpiar gradientes del paso anterior.
 * 2. forward + backward — calcular nuevos gradientes.
 * 3. opt.step()       — actualizar pesos usando los gradientes.
 */
struct Adam {
    std::vector<Tensor> params; ///< Referencia a todos los parámetros del modelo.
    float lr;           ///< Learning rate (tasa de aprendizaje, η).
    float beta1;        ///< Coeficiente de decay del primer momento (default: 0.9).
    float beta2;        ///< Coeficiente de decay del segundo momento (default: 0.999).
    float eps;          ///< Épsilon para estabilidad numérica (default: 1e-8).
    float weight_decay; ///< Coeficiente de weight decay desacoplado, λ (default: 0.0).
    long t = 0;         ///< Contador de pasos (para bias correction).

#ifdef USE_CUDA
    std::vector<float*> d_m, d_v; ///< Momentos m y v en GPU (un buffer por parámetro).
#endif
    std::vector<std::vector<float>> m, v; ///< Momentos m y v en CPU (un vector por parámetro).

    /**
     * @brief Construye el optimizador AdamW.
     *
     * Inicializa los momentos m y v a cero para cada parámetro.
     * Si se usa CUDA, también aloja los momentos en GPU.
     *
     * @param params_      Vector de tensores entrenables (obtenidos de model.parameters()).
     * @param lr_          Learning rate (default: 1e-3).
     * @param beta1_       Coeficiente β₁ del primer momento (default: 0.9).
     * @param beta2_       Coeficiente β₂ del segundo momento (default: 0.999).
     * @param eps_         Épsilon de estabilidad (default: 1e-8).
     * @param weight_decay_ Coeficiente de weight decay λ (default: 0.0, sin penalización).
     */
    Adam(std::vector<Tensor> params_, float lr_ = 1e-3f, float beta1_ = 0.9f,
         float beta2_ = 0.999f, float eps_ = 1e-8f, float weight_decay_ = 0.0f)
        : params(std::move(params_)), lr(lr_), beta1(beta1_), beta2(beta2_),
          eps(eps_), weight_decay(weight_decay_) {
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
    /** @brief Destructor: libera los buffers de momentos en GPU. */
    ~Adam() {
        for (auto* p : d_m) if (p) cudaFree(p);
        for (auto* p : d_v) if (p) cudaFree(p);
    }
#endif

    /**
     * @brief Pone a cero los gradientes de todos los parámetros.
     *
     * Debe llamarse antes de cada paso de forward+backward para evitar
     * acumular gradientes de iteraciones anteriores.
     */
    void zero_grad() {
        for (auto& p : params) p->zero_grad();
    }

    /**
     * @brief Ejecuta un paso de optimización AdamW.
     *
     * Actualiza todos los parámetros del modelo usando los gradientes acumulados.
     * Los gradientes se multiplican por grad_scale antes de usarse, lo que
     * permite promediar gradientes cuando se usa mini-batch.
     *
     * @param grad_scale Factor de escala para los gradientes (típicamente 1/batch_size).
     *                   Default: 1.0 (sin escala).
     */
    void step(float grad_scale = 1.0f) {
        ++t;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t)); // bias correction para m
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t)); // bias correction para v
#ifdef USE_CUDA
        for (size_t pi = 0; pi < params.size(); ++pi) {
            auto& p = params[pi];
            int sz = static_cast<int>(p->data.size());
            cuda::adam_step(p->d_data, p->d_grad, d_m[pi], d_v[pi],
                            lr, beta1, beta2, eps, bc1, bc2, grad_scale,
                            weight_decay, sz);
        }
#else
        for (size_t pi = 0; pi < params.size(); ++pi) {
            auto& p = params[pi];
            for (size_t i = 0; i < p->data.size(); ++i) {
                float g = p->grad[i] * grad_scale;
                m[pi][i] = beta1 * m[pi][i] + (1.0f - beta1) * g;       // actualizar primer momento
                v[pi][i] = beta2 * v[pi][i] + (1.0f - beta2) * g * g;   // actualizar segundo momento
                float mhat = m[pi][i] / bc1;   // bias-corrected m
                float vhat = v[pi][i] / bc2;   // bias-corrected v
                p->data[i] -= lr * mhat / (std::sqrt(vhat) + eps);       // paso Adam
                // AdamW: weight decay desacoplado (penalización L2 directa sobre el peso)
                if (weight_decay > 0.0f)
                    p->data[i] -= lr * weight_decay * p->data[i];
            }
        }
#endif
    }
};

} // namespace vit

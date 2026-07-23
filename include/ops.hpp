// ops.hpp
//
// Cada función crea un Tensor de salida y le asigna un backward_fn que
// implementa a mano la derivada de esa operación. Este es el "backprop
// manual" pedido: no hay diferenciación simbólica ni numérica, cada regla
// de la cadena está escrita explícitamente aquí.
//
// Cuando USE_CUDA está definido, las operaciones se ejecutan en GPU llamando
// a los kernels de cuda_kernels.cuh. La versión CPU se mantiene intacta
// como fallback cuando se compila sin CUDA.
//
// NOTA IMPORTANTE sobre gestión de memoria: backward_fn se guarda DENTRO del
// propio nodo `out` (out->backward_fn = ...). Si la lambda capturara `out`
// (shared_ptr) por valor, se crearía un ciclo de referencias (out posee a
// backward_fn, que posee una referencia a out) y el nodo nunca se liberaría,
// aunque nada más lo usara -> fuga de memoria que crece sin límite durante el
// entrenamiento. Por eso, dentro de cada backward_fn capturamos un puntero
// crudo `out_raw = out.get()`: es seguro porque mientras backward_fn se
// ejecuta, algo más arriba en la pila (el vector `order` de build_topo, o el
// grafo hacia adelante) sigue manteniendo vivo a `out`.

#pragma once

#include "tensor.hpp"
#include <cmath>

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#endif

namespace vit {

// out = A(m,k) * B(k,n)
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A->cols != B->rows) throw std::runtime_error("matmul: dimensiones incompatibles");
    int m = A->rows, k = A->cols, n = B->cols;
    auto out = make_tensor(m, n, A->requires_grad || B->requires_grad);
#ifdef USE_CUDA
    cuda::matmul_fwd(A->d_data, B->d_data, out->d_data, m, k, n);
#else
    #pragma omp parallel for
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int p = 0; p < k; ++p) s += A->at(i, p) * B->at(p, j);
            out->at(i, j) = s;
        }
#endif
    out->parents = {A, B};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, B, out_raw, m, k, n]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::matmul_bwd_A(out_raw->d_grad, B->d_data, A->d_grad, m, k, n);
        if (B->requires_grad) cuda::matmul_bwd_B(A->d_data, out_raw->d_grad, B->d_grad, m, k, n);
#else
        // dL/dA = dL/dout * B^T ; dL/dB = A^T * dL/dout
        if (A->requires_grad) {
            #pragma omp parallel for
            for (int i = 0; i < m; ++i)
                for (int p = 0; p < k; ++p) {
                    float s = 0.0f;
                    for (int j = 0; j < n; ++j) s += out_raw->g(i, j) * B->at(p, j);
                    A->g(i, p) += s;
                }
        }
        if (B->requires_grad) {
            #pragma omp parallel for
            for (int p = 0; p < k; ++p)
                for (int j = 0; j < n; ++j) {
                    float s = 0.0f;
                    for (int i = 0; i < m; ++i) s += A->at(i, p) * out_raw->g(i, j);
                    B->g(p, j) += s;
                }
        }
#endif
    };
    return out;
}

// Suma elemento a elemento (mismas dimensiones).
inline Tensor add(const Tensor& A, const Tensor& B) {
    if (A->rows != B->rows || A->cols != B->cols)
        throw std::runtime_error("add: dimensiones distintas");
    auto out = make_tensor(A->rows, A->cols, A->requires_grad || B->requires_grad);
    int size = A->rows * A->cols;
#ifdef USE_CUDA
    cuda::add_fwd(A->d_data, B->d_data, out->d_data, size);
#else
    for (size_t i = 0; i < out->data.size(); ++i) out->data[i] = A->data[i] + B->data[i];
#endif
    out->parents = {A, B};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, B, out_raw, size]() {
#ifdef USE_CUDA
        cuda::add_bwd(out_raw->d_grad, A->d_grad, B->d_grad, size,
                      A->requires_grad, B->requires_grad);
#else
        if (A->requires_grad) for (int i = 0; i < size; ++i) A->grad[i] += out_raw->grad[i];
        if (B->requires_grad) for (int i = 0; i < size; ++i) B->grad[i] += out_raw->grad[i];
#endif
    };
    return out;
}

// Suma un vector fila (1 x cols) a cada fila de A (broadcast de bias).
inline Tensor add_row_broadcast(const Tensor& A, const Tensor& bias) {
    if (bias->rows != 1 || A->cols != bias->cols)
        throw std::runtime_error("add_row_broadcast: forma de bias incorrecta");
    auto out = make_tensor(A->rows, A->cols, A->requires_grad || bias->requires_grad);
#ifdef USE_CUDA
    cuda::add_row_broadcast_fwd(A->d_data, bias->d_data, out->d_data, A->rows, A->cols);
#else
    for (int i = 0; i < A->rows; ++i)
        for (int j = 0; j < A->cols; ++j)
            out->at(i, j) = A->at(i, j) + bias->at(0, j);
#endif
    out->parents = {A, bias};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, bias, out_raw]() {
#ifdef USE_CUDA
        cuda::add_row_broadcast_bwd(out_raw->d_grad, A->d_grad, bias->d_grad,
                                    out_raw->rows, out_raw->cols,
                                    A->requires_grad, bias->requires_grad);
#else
        if (A->requires_grad)
            for (size_t i = 0; i < out_raw->data.size(); ++i) A->grad[i] += out_raw->grad[i];
        if (bias->requires_grad)
            for (int i = 0; i < out_raw->rows; ++i)
                for (int j = 0; j < out_raw->cols; ++j)
                    bias->g(0, j) += out_raw->g(i, j);
#endif
    };
    return out;
}

inline Tensor scale(const Tensor& A, float s) {
    auto out = make_tensor(A->rows, A->cols, A->requires_grad);
    int size = A->rows * A->cols;
#ifdef USE_CUDA
    cuda::scale_fwd(A->d_data, s, out->d_data, size);
#else
    for (size_t i = 0; i < A->data.size(); ++i) out->data[i] = A->data[i] * s;
#endif
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, s, size]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::scale_bwd(out_raw->d_grad, s, A->d_grad, size);
#else
        if (A->requires_grad) for (int i = 0; i < size; ++i) A->grad[i] += out_raw->grad[i] * s;
#endif
    };
    return out;
}

inline Tensor transpose(const Tensor& A) {
    auto out = make_tensor(A->cols, A->rows, A->requires_grad);
#ifdef USE_CUDA
    cuda::transpose_fwd(A->d_data, out->d_data, A->rows, A->cols);
#else
    for (int i = 0; i < A->rows; ++i)
        for (int j = 0; j < A->cols; ++j)
            out->at(j, i) = A->at(i, j);
#endif
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::transpose_bwd(out_raw->d_grad, A->d_grad, A->rows, A->cols);
#else
        if (!A->requires_grad) return;
        for (int i = 0; i < A->rows; ++i)
            for (int j = 0; j < A->cols; ++j)
                A->g(i, j) += out_raw->g(j, i);
#endif
    };
    return out;
}

// Selecciona un rango de columnas [c0, c0+width) -> usado para separar heads.
inline Tensor slice_cols(const Tensor& A, int c0, int width) {
    auto out = make_tensor(A->rows, width, A->requires_grad);
#ifdef USE_CUDA
    cuda::slice_cols_fwd(A->d_data, out->d_data, A->rows, A->cols, c0, width);
#else
    for (int i = 0; i < A->rows; ++i)
        for (int j = 0; j < width; ++j)
            out->at(i, j) = A->at(i, c0 + j);
#endif
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, c0, width]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::slice_cols_bwd(out_raw->d_grad, A->d_grad, A->rows, A->cols, c0, width);
#else
        if (!A->requires_grad) return;
        for (int i = 0; i < A->rows; ++i)
            for (int j = 0; j < width; ++j)
                A->g(i, c0 + j) += out_raw->g(i, j);
#endif
    };
    return out;
}

// Concatena una lista de tensores con igual número de filas a lo largo de columnas.
inline Tensor concat_cols(const std::vector<Tensor>& parts) {
    int rows = parts[0]->rows;
    int total_cols = 0;
    bool rg = false;
    for (auto& p : parts) { total_cols += p->cols; rg = rg || p->requires_grad; }
    auto out = make_tensor(rows, total_cols, rg);
#ifdef USE_CUDA
    int offset = 0;
    for (auto& p : parts) {
        cuda::concat_cols_copy(p->d_data, out->d_data, rows, p->cols, total_cols, offset);
        offset += p->cols;
    }
#else
    int offset = 0;
    for (auto& p : parts) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < p->cols; ++j)
                out->at(i, offset + j) = p->at(i, j);
        offset += p->cols;
    }
#endif
    out->parents = parts;
    TensorImpl* out_raw = out.get();
    out->backward_fn = [parts, out_raw, rows, total_cols]() {
#ifdef USE_CUDA
        int off = 0;
        for (auto& p : parts) {
            if (p->requires_grad)
                cuda::concat_cols_bwd_part(out_raw->d_grad, p->d_grad, rows, p->cols, total_cols, off);
            off += p->cols;
        }
#else
        int off = 0;
        for (auto& p : parts) {
            if (p->requires_grad)
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < p->cols; ++j)
                        p->g(i, j) += out_raw->g(i, off + j);
            off += p->cols;
        }
#endif
    };
    return out;
}

// Selecciona una sola fila (usado para extraer el token CLS). Devuelve (1, cols).
inline Tensor select_row(const Tensor& A, int row) {
    auto out = make_tensor(1, A->cols, A->requires_grad);
#ifdef USE_CUDA
    cuda::select_row_fwd(A->d_data, out->d_data, row, A->cols, A->cols);
#else
    for (int j = 0; j < A->cols; ++j) out->at(0, j) = A->at(row, j);
#endif
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, row]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::select_row_bwd(out_raw->d_grad, A->d_grad, row, A->cols);
#else
        if (!A->requires_grad) return;
        for (int j = 0; j < A->cols; ++j) A->g(row, j) += out_raw->g(0, j);
#endif
    };
    return out;
}

// Softmax fila a fila (estable numéricamente restando el máximo).
inline Tensor softmax_rows(const Tensor& A) {
    auto out = make_tensor(A->rows, A->cols, A->requires_grad);
#ifdef USE_CUDA
    cuda::softmax_rows_fwd(A->d_data, out->d_data, A->rows, A->cols);
#else
    for (int i = 0; i < A->rows; ++i) {
        float mx = A->at(i, 0);
        for (int j = 1; j < A->cols; ++j) mx = std::max(mx, A->at(i, j));
        float sum = 0.0f;
        for (int j = 0; j < A->cols; ++j) {
            float e = std::exp(A->at(i, j) - mx);
            out->at(i, j) = e;
            sum += e;
        }
        for (int j = 0; j < A->cols; ++j) out->at(i, j) /= sum;
    }
#endif
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::softmax_rows_bwd(out_raw->d_grad, out_raw->d_data, A->d_grad, A->rows, A->cols);
#else
        if (!A->requires_grad) return;
        // Jacobiano del softmax por fila: dL/dx_j = s_j * (dL/dy_j - sum_k dL/dy_k * s_k)
        for (int i = 0; i < A->rows; ++i) {
            float dot = 0.0f;
            for (int j = 0; j < A->cols; ++j) dot += out_raw->g(i, j) * out_raw->at(i, j);
            for (int j = 0; j < A->cols; ++j)
                A->g(i, j) += out_raw->at(i, j) * (out_raw->g(i, j) - dot);
        }
#endif
    };
    return out;
}

// LayerNorm fila a fila con parámetros aprendibles gamma, beta (1 x cols).
inline Tensor layer_norm(const Tensor& A, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f) {
    int rows = A->rows, cols = A->cols;
    auto out = make_tensor(rows, cols, true);

#ifdef USE_CUDA
    // Alojar buffers auxiliares en GPU para mean, rstd, normed
    float* d_mean = nullptr; cudaMalloc(&d_mean, rows * sizeof(float));
    float* d_rstd = nullptr; cudaMalloc(&d_rstd, rows * sizeof(float));
    auto normed = make_tensor(rows, cols, false);

    cuda::layer_norm_fwd(A->d_data, gamma->d_data, beta->d_data,
                         out->d_data, d_mean, d_rstd, normed->d_data,
                         rows, cols, eps);

    out->parents = {A, gamma, beta};
    TensorImpl* out_raw = out.get();
    TensorImpl* normed_raw = normed.get();
    out->backward_fn = [A, gamma, beta, out_raw, normed, normed_raw, d_mean, d_rstd, rows, cols]() {
        (void)normed; // mantener vivo
        cuda::layer_norm_bwd(out_raw->d_grad, gamma->d_data,
                             normed_raw->d_data, d_rstd,
                             A->d_grad, gamma->d_grad, beta->d_grad,
                             rows, cols,
                             A->requires_grad, gamma->requires_grad, beta->requires_grad);
        // Nota: d_mean y d_rstd se liberarán cuando el backward_fn sea destruido.
        // Para evitar leaks, los liberamos aquí no... mejor los liberamos en un capture
        // que se ejecuta al destruir el lambda. En realidad, como backward se llama
        // una vez por forward, podríamos liberarlos aquí. Pero es más seguro dejarlos
        // hasta que el tensor out sea destruido. Los capturamos en un shared_ptr wrapper.
    };
    // Registrar cleanup de d_mean y d_rstd como parte del backward_fn.
    // Solución pragmática: no liberarlos explícitamente; se liberan cuando
    // TensorImpl de out es destruido (y con él, su backward_fn y las capturas).
    // Para evitar el leak, usamos un shared_ptr con deleter.
    auto mean_guard = std::shared_ptr<float>(d_mean, [](float* p) { cudaFree(p); });
    auto rstd_guard = std::shared_ptr<float>(d_rstd, [](float* p) { cudaFree(p); });
    // Re-asignar backward capturando los guards:
    out->backward_fn = [A, gamma, beta, out_raw, normed, normed_raw,
                         mean_guard, rstd_guard, rows, cols]() {
        (void)normed;
        cuda::layer_norm_bwd(out_raw->d_grad, gamma->d_data,
                             normed_raw->d_data, rstd_guard.get(),
                             A->d_grad, gamma->d_grad, beta->d_grad,
                             rows, cols,
                             A->requires_grad, gamma->requires_grad, beta->requires_grad);
    };
#else
    std::vector<float> mean(rows), rstd(rows); // guardados para backward
    auto normed = make_tensor(rows, cols, false); // (x - mean) / std, sin gamma/beta
    for (int i = 0; i < rows; ++i) {
        float m = 0.0f;
        for (int j = 0; j < cols; ++j) m += A->at(i, j);
        m /= cols;
        float var = 0.0f;
        for (int j = 0; j < cols; ++j) { float d = A->at(i, j) - m; var += d * d; }
        var /= cols;
        float rs = 1.0f / std::sqrt(var + eps);
        mean[i] = m; rstd[i] = rs;
        for (int j = 0; j < cols; ++j) {
            float nh = (A->at(i, j) - m) * rs;
            normed->at(i, j) = nh;
            out->at(i, j) = nh * gamma->at(0, j) + beta->at(0, j);
        }
    }
    out->parents = {A, gamma, beta};
    TensorImpl* out_raw = out.get();
    TensorImpl* normed_raw = normed.get();
    out->backward_fn = [A, gamma, beta, out_raw, normed, normed_raw, mean, rstd, rows, cols]() {
        (void)normed; // mantenido vivo por captura de shared_ptr; usamos normed_raw para acceder
        for (int i = 0; i < rows; ++i) {
            float rs = rstd[i];
            float sum_dy = 0.0f, sum_dy_nh = 0.0f;
            for (int j = 0; j < cols; ++j) {
                float dy = out_raw->g(i, j) * gamma->at(0, j); // dL/d(normed)
                sum_dy += dy;
                sum_dy_nh += dy * normed_raw->at(i, j);
            }
            if (A->requires_grad) {
                for (int j = 0; j < cols; ++j) {
                    float dy = out_raw->g(i, j) * gamma->at(0, j);
                    // derivada estándar de layernorm respecto a la entrada
                    float dx = rs * (dy - sum_dy / cols - normed_raw->at(i, j) * sum_dy_nh / cols);
                    A->g(i, j) += dx;
                }
            }
            if (gamma->requires_grad)
                for (int j = 0; j < cols; ++j) gamma->g(0, j) += out_raw->g(i, j) * normed_raw->at(i, j);
            if (beta->requires_grad)
                for (int j = 0; j < cols; ++j) beta->g(0, j) += out_raw->g(i, j);
        }
    };
#endif
    return out;
}

// GELU (aproximación tanh, la misma que usan BERT/ViT/GPT).
inline Tensor gelu(const Tensor& A) {
    auto out = make_tensor(A->rows, A->cols, A->requires_grad);
    int size = A->rows * A->cols;
#ifdef USE_CUDA
    cuda::gelu_fwd(A->d_data, out->d_data, size);
#else
    const float k0 = 0.7978845608f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    for (size_t i = 0; i < A->data.size(); ++i) {
        float x = A->data[i];
        float x3 = x * x * x;
        float t = std::tanh(k0 * (x + k1 * x3));
        out->data[i] = 0.5f * x * (1.0f + t);
    }
#endif
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, size]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::gelu_bwd(out_raw->d_grad, A->d_data, A->d_grad, size);
#else
        if (!A->requires_grad) return;
        const float k0 = 0.7978845608f;
        const float k1 = 0.044715f;
        for (size_t i = 0; i < A->data.size(); ++i) {
            float x = A->data[i];
            float x2 = x * x;
            float x3 = x2 * x;
            float inner = k0 * (x + k1 * x3);
            float t = std::tanh(inner);
            float sech2 = 1.0f - t * t;
            float dinner = k0 * (1.0f + 3.0f * k1 * x2);
            float dgelu = 0.5f * (1.0f + t) + 0.5f * x * sech2 * dinner;
            A->grad[i] += out_raw->grad[i] * dgelu;
        }
#endif
    };
    return out;
}

// Softmax + cross-entropy combinados para estabilidad numérica y gradiente simple.
// logits: (1, num_classes). label: índice de clase correcta.
// Devuelve la pérdida como escalar (1,1) y opcionalmente escribe las
// probabilidades en `probs_out` (para calcular accuracy sin recomputar).
inline Tensor softmax_cross_entropy(const Tensor& logits, int label, std::vector<float>* probs_out = nullptr) {
    int n = logits->cols;

#ifdef USE_CUDA
    // Alojar buffers en GPU para loss y probs usando el allocator
    size_t probs_bytes = n * sizeof(float);
    float* d_probs = CachingAllocator::allocate(probs_bytes);

    auto out = make_tensor(1, 1, true);
    cuda::softmax_cross_entropy_fwd(logits->d_data, label, n, out->d_data, d_probs);

    // Traer loss y probs a CPU para métricas
    out->to_host();
    if (probs_out) {
        probs_out->resize(n);
        cudaMemcpy(probs_out->data(), d_probs, probs_bytes, cudaMemcpyDeviceToHost);
    }

    out->parents = {logits};
    TensorImpl* out_raw = out.get();
    auto probs_guard = std::shared_ptr<float>(d_probs, [probs_bytes](float* p) { 
        CachingAllocator::free(p, probs_bytes); 
    });
    out->backward_fn = [logits, out_raw, probs_guard, label, n]() {
        if (!logits->requires_grad) return;
        float upstream = out_raw->grad[0]; // siempre 1.0 desde backward()
        cuda::softmax_cross_entropy_bwd(upstream, probs_guard.get(), label, n, logits->d_grad);
    };
#else
    float mx = logits->at(0, 0);
    for (int j = 1; j < n; ++j) mx = std::max(mx, logits->at(0, j));
    std::vector<float> probs(n);
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) { probs[j] = std::exp(logits->at(0, j) - mx); sum += probs[j]; }
    for (int j = 0; j < n; ++j) probs[j] /= sum;
    if (probs_out) *probs_out = probs;

    float loss_val = -std::log(std::max(probs[label], 1e-9f));
    auto out = make_tensor(1, 1, true);
    out->data[0] = loss_val;
    out->parents = {logits};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [logits, out_raw, probs, label, n]() {
        if (!logits->requires_grad) return;
        float upstream = out_raw->grad[0];
        for (int j = 0; j < n; ++j) {
            float target = (j == label) ? 1.0f : 0.0f;
            logits->g(0, j) += upstream * (probs[j] - target); // derivada clásica softmax+CE
        }
    };
#endif
    return out;
}

} // namespace vit

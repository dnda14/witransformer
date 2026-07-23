// ops.hpp
//
// Cada función crea un Tensor de salida y le asigna un backward_fn que
// implementa a mano la derivada de esa operación. Este es el "backprop
// manual" pedido: no hay diferenciación simbólica ni numérica, cada regla
// de la cadena está escrita explícitamente aquí.
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

namespace vit {

// out = A(m,k) * B(k,n)
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A->cols != B->rows) throw std::runtime_error("matmul: dimensiones incompatibles");
    int m = A->rows, k = A->cols, n = B->cols;
    auto out = make_tensor(m, n, A->requires_grad || B->requires_grad);
    #pragma omp parallel for
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int p = 0; p < k; ++p) s += A->at(i, p) * B->at(p, j);
            out->at(i, j) = s;
        }
    out->parents = {A, B};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, B, out_raw, m, k, n]() {
        // dL/dA = dL/dout * B^T ; dL/dB = A^T * dL/dout
        // Cada iteración escribe celdas exclusivas de A->grad / B->grad, por
        // lo que paralelizar el bucle externo es seguro (sin condiciones de carrera).
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
    };
    return out;
}

// Suma elemento a elemento (mismas dimensiones).
inline Tensor add(const Tensor& A, const Tensor& B) {
    if (A->rows != B->rows || A->cols != B->cols)
        throw std::runtime_error("add: dimensiones distintas");
    auto out = make_tensor(A->rows, A->cols, A->requires_grad || B->requires_grad);
    for (size_t i = 0; i < out->data.size(); ++i) out->data[i] = A->data[i] + B->data[i];
    out->parents = {A, B};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, B, out_raw]() {
        if (A->requires_grad) for (size_t i = 0; i < out_raw->data.size(); ++i) A->grad[i] += out_raw->grad[i];
        if (B->requires_grad) for (size_t i = 0; i < out_raw->data.size(); ++i) B->grad[i] += out_raw->grad[i];
    };
    return out;
}

// Suma un vector fila (1 x cols) a cada fila de A (broadcast de bias).
inline Tensor add_row_broadcast(const Tensor& A, const Tensor& bias) {
    if (bias->rows != 1 || A->cols != bias->cols)
        throw std::runtime_error("add_row_broadcast: forma de bias incorrecta");
    auto out = make_tensor(A->rows, A->cols, A->requires_grad || bias->requires_grad);
    for (int i = 0; i < A->rows; ++i)
        for (int j = 0; j < A->cols; ++j)
            out->at(i, j) = A->at(i, j) + bias->at(0, j);
    out->parents = {A, bias};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, bias, out_raw]() {
        if (A->requires_grad)
            for (size_t i = 0; i < out_raw->data.size(); ++i) A->grad[i] += out_raw->grad[i];
        if (bias->requires_grad)
            for (int i = 0; i < out_raw->rows; ++i)
                for (int j = 0; j < out_raw->cols; ++j)
                    bias->g(0, j) += out_raw->g(i, j);
    };
    return out;
}

inline Tensor scale(const Tensor& A, float s) {
    auto out = make_tensor(A->rows, A->cols, A->requires_grad);
    for (size_t i = 0; i < A->data.size(); ++i) out->data[i] = A->data[i] * s;
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, s]() {
        if (A->requires_grad) for (size_t i = 0; i < out_raw->data.size(); ++i) A->grad[i] += out_raw->grad[i] * s;
    };
    return out;
}

inline Tensor transpose(const Tensor& A) {
    auto out = make_tensor(A->cols, A->rows, A->requires_grad);
    for (int i = 0; i < A->rows; ++i)
        for (int j = 0; j < A->cols; ++j)
            out->at(j, i) = A->at(i, j);
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw]() {
        if (!A->requires_grad) return;
        for (int i = 0; i < A->rows; ++i)
            for (int j = 0; j < A->cols; ++j)
                A->g(i, j) += out_raw->g(j, i);
    };
    return out;
}

// Selecciona un rango de columnas [c0, c0+width) -> usado para separar heads.
inline Tensor slice_cols(const Tensor& A, int c0, int width) {
    auto out = make_tensor(A->rows, width, A->requires_grad);
    for (int i = 0; i < A->rows; ++i)
        for (int j = 0; j < width; ++j)
            out->at(i, j) = A->at(i, c0 + j);
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, c0, width]() {
        if (!A->requires_grad) return;
        for (int i = 0; i < A->rows; ++i)
            for (int j = 0; j < width; ++j)
                A->g(i, c0 + j) += out_raw->g(i, j);
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
    int offset = 0;
    for (auto& p : parts) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < p->cols; ++j)
                out->at(i, offset + j) = p->at(i, j);
        offset += p->cols;
    }
    out->parents = parts;
    TensorImpl* out_raw = out.get();
    out->backward_fn = [parts, out_raw, rows]() {
        int off = 0;
        for (auto& p : parts) {
            if (p->requires_grad)
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < p->cols; ++j)
                        p->g(i, j) += out_raw->g(i, off + j);
            off += p->cols;
        }
    };
    return out;
}

// Selecciona una sola fila (usado para extraer el token CLS). Devuelve (1, cols).
inline Tensor select_row(const Tensor& A, int row) {
    auto out = make_tensor(1, A->cols, A->requires_grad);
    for (int j = 0; j < A->cols; ++j) out->at(0, j) = A->at(row, j);
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, row]() {
        if (!A->requires_grad) return;
        for (int j = 0; j < A->cols; ++j) A->g(row, j) += out_raw->g(0, j);
    };
    return out;
}

// Softmax fila a fila (estable numéricamente restando el máximo).
inline Tensor softmax_rows(const Tensor& A) {
    auto out = make_tensor(A->rows, A->cols, A->requires_grad);
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
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw]() {
        if (!A->requires_grad) return;
        // Jacobiano del softmax por fila: dL/dx_j = s_j * (dL/dy_j - sum_k dL/dy_k * s_k)
        for (int i = 0; i < A->rows; ++i) {
            float dot = 0.0f;
            for (int j = 0; j < A->cols; ++j) dot += out_raw->g(i, j) * out_raw->at(i, j);
            for (int j = 0; j < A->cols; ++j)
                A->g(i, j) += out_raw->at(i, j) * (out_raw->g(i, j) - dot);
        }
    };
    return out;
}

// LayerNorm fila a fila con parámetros aprendibles gamma, beta (1 x cols).
inline Tensor layer_norm(const Tensor& A, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f) {
    int rows = A->rows, cols = A->cols;
    auto out = make_tensor(rows, cols, true);
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
    return out;
}

// GELU (aproximación tanh, la misma que usan BERT/ViT/GPT).
inline Tensor gelu(const Tensor& A) {
    auto out = make_tensor(A->rows, A->cols, A->requires_grad);
    const float k0 = 0.7978845608f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    for (size_t i = 0; i < A->data.size(); ++i) {
        float x = A->data[i];
        float x3 = x * x * x;
        float t = std::tanh(k0 * (x + k1 * x3));
        out->data[i] = 0.5f * x * (1.0f + t);
    }
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, k0, k1]() {
        if (!A->requires_grad) return;
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
    };
    return out;
}

// Softmax + cross-entropy combinados para estabilidad numérica y gradiente simple.
// logits: (1, num_classes). label: índice de clase correcta.
// Devuelve la pérdida como escalar (1,1) y opcionalmente escribe las
// probabilidades en `probs_out` (para calcular accuracy sin recomputar).
inline Tensor softmax_cross_entropy(const Tensor& logits, int label, std::vector<float>* probs_out = nullptr) {
    int n = logits->cols;
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
    return out;
}

} // namespace vit

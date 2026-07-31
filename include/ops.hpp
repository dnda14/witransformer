/**
 * @file ops.hpp
 * @brief Operaciones matemáticas diferenciables para Tensores.
 *
 * Contiene las operaciones fundamentales del motor de cálculo tensorial:
 * suma, multiplicación de matrices, softmax, LayerNorm, GELU y cross-entropy.
 *
 * **Backpropagation manual:** Cada función crea un Tensor de salida y le asigna
 * un `backward_fn` que implementa a mano la derivada de esa operación. Este es
 * el "backprop manual" — no hay diferenciación simbólica ni numérica, cada regla
 * de la cadena está escrita explícitamente aquí.
 *
 * **Soporte dual CPU/GPU:** Cuando `USE_CUDA` está definido, las operaciones
 * se ejecutan en GPU llamando a los kernels de cuda_kernels.cuh. La versión
 * CPU se mantiene intacta como fallback.
 *
 * @note **Gestión de memoria:** Dentro de cada backward_fn capturamos un puntero
 * crudo `out_raw = out.get()` en vez de capturar `out` (shared_ptr) por valor.
 * Si capturáramos `out`, se crearía un ciclo de referencias (out posee a
 * backward_fn, que posee una referencia a out) y el nodo nunca se liberaría
 * → fuga de memoria. El puntero crudo es seguro porque mientras backward_fn
 * se ejecuta, el grafo hacia adelante sigue manteniendo vivo a `out`.
 */

#pragma once

#include "tensor.hpp"
#include <cmath>

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#endif

namespace vit {

/**
 * @brief Multiplicación de matrices: out = A × B.
 *
 * Calcula el producto matricial estándar. En CPU usa OpenMP para paralelizar
 * filas; en GPU usa tiling con shared memory.
 *
 * **Backward (regla de la cadena para matmul):**
 * - dL/dA = dL/dout × Bᵀ
 * - dL/dB = Aᵀ × dL/dout
 *
 * @param A Tensor de entrada con forma (m, k).
 * @param B Tensor de entrada con forma (k, n).
 * @return  Tensor resultado con forma (m, n), con backward_fn registrado.
 * @throws std::runtime_error Si A.cols != B.rows.
 */
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
        // dL/dA += dL/dout × Bᵀ
        if (A->requires_grad) {
            #pragma omp parallel for
            for (int i = 0; i < m; ++i)
                for (int p = 0; p < k; ++p) {
                    float s = 0.0f;
                    for (int j = 0; j < n; ++j) s += out_raw->g(i, j) * B->at(p, j);
                    A->g(i, p) += s;
                }
        }
        // dL/dB += Aᵀ × dL/dout
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

/**
 * @brief Suma elemento a elemento: out = A + B.
 *
 * Ambos tensores deben tener exactamente las mismas dimensiones.
 *
 * **Backward:** La derivada de la suma es la identidad.
 * - dL/dA += dL/dout
 * - dL/dB += dL/dout
 *
 * @param A Tensor de entrada con forma (rows, cols).
 * @param B Tensor de entrada con forma (rows, cols).
 * @return  Tensor resultado con forma (rows, cols).
 * @throws std::runtime_error Si las dimensiones no coinciden.
 */
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

/**
 * @brief Suma con broadcast por fila: out[i,:] = A[i,:] + bias[0,:].
 *
 * Suma un vector fila (1 × cols) a cada fila de A. Se usa típicamente
 * para sumar el sesgo (bias) después de una capa lineal.
 *
 * **Backward:**
 * - dL/dA += dL/dout  (copia directa)
 * - dL/dbias += Σᵢ dL/dout[i,:]  (suma sobre todas las filas)
 *
 * @param A    Tensor de entrada con forma (rows, cols).
 * @param bias Tensor fila con forma (1, cols).
 * @return     Tensor resultado con forma (rows, cols).
 * @throws std::runtime_error Si bias no es (1, cols) o cols no coinciden.
 */
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

/**
 * @brief Escala un tensor por un escalar: out = A × s.
 *
 * **Backward:** dL/dA += dL/dout × s
 *
 * @param A Tensor de entrada con forma (rows, cols).
 * @param s Factor escalar de multiplicación.
 * @return  Tensor resultado con forma (rows, cols).
 */
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

/**
 * @brief Transpone una matriz: out = Aᵀ. Si A es (m, n), out es (n, m).
 *
 * **Backward:** dL/dA[i,j] += dL/dout[j,i]  (transponer el gradiente)
 *
 * @param A Tensor de entrada con forma (rows, cols).
 * @return  Tensor resultado con forma (cols, rows).
 */
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

/**
 * @brief Extrae un rango de columnas: out = A[:, c0 : c0+width].
 *
 * Se usa en Multi-Head Attention para separar las cabezas (heads) a partir
 * de las proyecciones Q, K, V concatenadas.
 *
 * **Backward:** dL/dA[:, c0 : c0+width] += dL/dout
 *
 * @param A     Tensor de entrada con forma (rows, total_cols).
 * @param c0    Índice de la primera columna a extraer.
 * @param width Número de columnas a extraer.
 * @return      Tensor resultado con forma (rows, width).
 */
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

/**
 * @brief Concatena tensores horizontalmente (por columnas).
 *
 * Todos los tensores deben tener el mismo número de filas. Se usa en
 * Multi-Head Attention para reunir las salidas de todas las cabezas.
 *
 * **Backward:** Cada parte recibe el gradiente correspondiente a sus columnas.
 * Es la operación inversa de slice_cols.
 *
 * @param parts Vector de tensores a concatenar, todos con forma (rows, cols_i).
 * @return      Tensor resultado con forma (rows, Σ cols_i).
 */
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

/**
 * @brief Selecciona una sola fila de un tensor: out = A[row, :].
 *
 * Se usa para extraer el token CLS del ViT, que contiene la representación
 * global de la imagen para clasificación.
 *
 * **Backward:** dL/dA[row, :] += dL/dout[0, :]
 *
 * @param A   Tensor de entrada con forma (rows, cols).
 * @param row Índice de la fila a extraer (0-indexed).
 * @return    Tensor resultado con forma (1, cols).
 */
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

/**
 * @brief Softmax estable fila por fila.
 *
 * Para cada fila i, calcula:
 *   out[i,j] = exp(A[i,j] - max_j A[i,j]) / Σ_k exp(A[i,k] - max_k A[i,k])
 *
 * Resta el máximo por fila antes de exp() para evitar overflow numérico.
 *
 * **Backward (Jacobiano del softmax por fila):**
 *   dL/dA[i,j] += s[i,j] × (dL/dout[i,j] − Σ_k dL/dout[i,k] × s[i,k])
 * donde s = softmax(A) es la salida del forward.
 *
 * @param A Tensor de entrada con forma (rows, cols).
 * @return  Tensor resultado con forma (rows, cols), cada fila sumando 1.0.
 */
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

/**
 * @brief Layer Normalization con parámetros aprendibles gamma y beta.
 *
 * Para cada fila i, normaliza a media 0 y varianza 1, luego escala y desplaza:
 *   normed[i,j] = (A[i,j] - mean_i) / √(var_i + ε)
 *   out[i,j]    = gamma[j] × normed[i,j] + beta[j]
 *
 * **Backward:** Se calculan las derivadas respecto a A, gamma y beta
 * usando la fórmula estándar de LayerNorm (ver Apéndice de Ba et al., 2016).
 * - dL/dgamma[j] += Σᵢ dL/dout[i,j] × normed[i,j]
 * - dL/dbeta[j]  += Σᵢ dL/dout[i,j]
 * - dL/dA[i,j]    se calcula con la regla de la cadena completa incluyendo mean y var.
 *
 * @param A     Tensor de entrada con forma (rows, cols).
 * @param gamma Tensor de escala con forma (1, cols), inicializado a 1.
 * @param beta  Tensor de desplazamiento con forma (1, cols), inicializado a 0.
 * @param eps   Épsilon para estabilidad numérica (evita división por cero). Default: 1e-5.
 * @return      Tensor normalizado con forma (rows, cols).
 */
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
        (void)normed; // mantener vivo el shared_ptr
        cuda::layer_norm_bwd(out_raw->d_grad, gamma->d_data,
                             normed_raw->d_data, d_rstd,
                             A->d_grad, gamma->d_grad, beta->d_grad,
                             rows, cols,
                             A->requires_grad, gamma->requires_grad, beta->requires_grad);
    };
    // Registrar cleanup de d_mean y d_rstd usando shared_ptr con deleter personalizado.
    // Se liberan automáticamente cuando el backward_fn (y sus capturas) se destruyen.
    auto mean_guard = std::shared_ptr<float>(d_mean, [](float* p) { cudaFree(p); });
    auto rstd_guard = std::shared_ptr<float>(d_rstd, [](float* p) { cudaFree(p); });
    // Re-asignar backward capturando los guards para que mantengan vivos los buffers:
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
        // Calcular media de la fila
        float m = 0.0f;
        for (int j = 0; j < cols; ++j) m += A->at(i, j);
        m /= cols;
        // Calcular varianza de la fila
        float var = 0.0f;
        for (int j = 0; j < cols; ++j) { float d = A->at(i, j) - m; var += d * d; }
        var /= cols;
        float rs = 1.0f / std::sqrt(var + eps);
        mean[i] = m; rstd[i] = rs;
        // Normalizar y aplicar gamma/beta
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
            // Acumular sumas parciales para la derivada de LayerNorm
            float sum_dy = 0.0f, sum_dy_nh = 0.0f;
            for (int j = 0; j < cols; ++j) {
                float dy = out_raw->g(i, j) * gamma->at(0, j); // dL/d(normed)
                sum_dy += dy;
                sum_dy_nh += dy * normed_raw->at(i, j);
            }
            // dL/dA: derivada estándar de LayerNorm respecto a la entrada
            if (A->requires_grad) {
                for (int j = 0; j < cols; ++j) {
                    float dy = out_raw->g(i, j) * gamma->at(0, j);
                    float dx = rs * (dy - sum_dy / cols - normed_raw->at(i, j) * sum_dy_nh / cols);
                    A->g(i, j) += dx;
                }
            }
            // dL/dgamma y dL/dbeta
            if (gamma->requires_grad)
                for (int j = 0; j < cols; ++j) gamma->g(0, j) += out_raw->g(i, j) * normed_raw->at(i, j);
            if (beta->requires_grad)
                for (int j = 0; j < cols; ++j) beta->g(0, j) += out_raw->g(i, j);
        }
    };
#endif
    return out;
}

/**
 * @brief Función de activación GELU (Gaussian Error Linear Unit).
 *
 * Usa la aproximación por tanh, idéntica a la de BERT/ViT/GPT:
 *   GELU(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
 *
 * **Backward:**
 *   dGELU/dx = 0.5 × (1 + t) + 0.5 × x × sech²(inner) × d(inner)/dx
 *   donde t = tanh(inner), inner = √(2/π) × (x + 0.044715 × x³)
 *
 * @param A Tensor de entrada con forma (rows, cols).
 * @return  Tensor resultado con forma (rows, cols), misma dimensión.
 */
inline Tensor gelu(const Tensor& A) {
    auto out = make_tensor(A->rows, A->cols, A->requires_grad);
    int size = A->rows * A->cols;
#ifdef USE_CUDA
    cuda::gelu_fwd(A->d_data, out->d_data, size);
#else
    const float k0 = 0.7978845608f; // √(2/π)
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

/**
 * @brief Softmax + Cross-Entropy combinados en una sola operación.
 *
 * Combinar ambas operaciones en una sola función tiene dos ventajas:
 * 1. **Estabilidad numérica:** evita calcular log(softmax) por separado.
 * 2. **Gradiente simple:** la derivada combinada es simplemente (probs - target),
 *    evitando el Jacobiano completo del softmax.
 *
 * Fórmulas:
 * - Forward:  L = −log(probs[label]),  donde probs = softmax(logits)
 * - Backward: dL/dlogits[j] = probs[j] − 1{j == label}
 *
 * @param logits    Tensor con las puntuaciones crudas, forma (1, num_classes).
 * @param label     Índice de la clase correcta (0 a num_classes-1).
 * @param probs_out [out] Puntero opcional donde se escriben las probabilidades
 *                  resultantes del softmax (útil para calcular accuracy sin recomputar).
 * @return          Tensor escalar (1×1) con el valor de la pérdida cross-entropy.
 */
inline Tensor softmax_cross_entropy(const Tensor& logits, int label, std::vector<float>* probs_out = nullptr) {
    int n = logits->cols;

#ifdef USE_CUDA
    // Alojar buffer en GPU para las probabilidades usando el CachingAllocator
    size_t probs_bytes = n * sizeof(float);
    float* d_probs = CachingAllocator::allocate(probs_bytes);

    auto out = make_tensor(1, 1, true);
    cuda::softmax_cross_entropy_fwd(logits->d_data, label, n, out->d_data, d_probs);

    // Traer loss y probs a CPU para métricas de entrenamiento
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
    // Softmax estable: restar el máximo para evitar overflow en exp()
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
        // Derivada combinada softmax+CE: dL/dlogits[j] = probs[j] - target[j]
        for (int j = 0; j < n; ++j) {
            float target = (j == label) ? 1.0f : 0.0f;
            logits->g(0, j) += upstream * (probs[j] - target);
        }
    };
#endif
    return out;
}

} // namespace vit

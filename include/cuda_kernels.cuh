/**
 * @file cuda_kernels.cuh
 * @brief Declaraciones de las funciones wrapper que lanzan los kernels CUDA.
 *
 * Cada función recibe punteros device (float*) y dimensiones, calcula la
 * configuración de bloques/hilos y lanza el kernel correspondiente definido
 * en cuda_kernels.cu.
 *
 * Se usa desde ops.hpp cuando `USE_CUDA` está definido. Las implementaciones
 * están en cuda_kernels.cu.
 *
 * @note Convenciones:
 * - Matrices en row-major: A[i*cols + j].
 * - Los kernels backward **acumulan** gradientes (+=), no los sobreescriben.
 * - Los punteros (A, B, out, dA, dB, etc.) deben apuntar a memoria de GPU.
 */

#pragma once

#ifdef USE_CUDA

#include <cstddef>

namespace vit { namespace cuda {

// ======================== Forward ========================

/**
 * @brief Multiplicación de matrices: out(m,n) = A(m,k) × B(k,n).
 * @param A   Puntero device a la matriz A (m×k).
 * @param B   Puntero device a la matriz B (k×n).
 * @param out Puntero device a la salida (m×n).
 * @param m   Filas de A / filas de out.
 * @param k   Columnas de A / filas de B.
 * @param n   Columnas de B / columnas de out.
 */
void matmul_fwd(const float* A, const float* B, float* out,
                int m, int k, int n);

/**
 * @brief Suma elemento a elemento: out = A + B.
 * @param A    Puntero device al primer operando.
 * @param B    Puntero device al segundo operando.
 * @param out  Puntero device a la salida.
 * @param size Número total de elementos.
 */
void add_fwd(const float* A, const float* B, float* out, int size);

/**
 * @brief Suma con broadcast por fila: out[i,:] = A[i,:] + bias[0,:].
 * @param A    Puntero device a la matriz (rows × cols).
 * @param bias Puntero device al vector fila (1 × cols).
 * @param out  Puntero device a la salida (rows × cols).
 * @param rows Número de filas.
 * @param cols Número de columnas.
 */
void add_row_broadcast_fwd(const float* A, const float* bias, float* out,
                           int rows, int cols);

/**
 * @brief Escala por un escalar: out = A × scalar.
 * @param A      Puntero device a la entrada.
 * @param scalar Factor de escala.
 * @param out    Puntero device a la salida.
 * @param size   Número total de elementos.
 */
void scale_fwd(const float* A, float scalar, float* out, int size);

/**
 * @brief Transpone una matriz: out(cols, rows) = A(rows, cols)ᵀ.
 * @param A    Puntero device a la entrada (rows × cols).
 * @param out  Puntero device a la salida (cols × rows).
 * @param rows Filas de A.
 * @param cols Columnas de A.
 */
void transpose_fwd(const float* A, float* out, int rows, int cols);

/**
 * @brief Extrae columnas: out = A[:, c0 : c0+width].
 * @param A          Puntero device a la matriz fuente (rows × total_cols).
 * @param out        Puntero device a la salida (rows × width).
 * @param rows       Número de filas.
 * @param total_cols Número total de columnas de A.
 * @param c0         Índice de la primera columna a extraer.
 * @param width      Número de columnas a extraer.
 */
void slice_cols_fwd(const float* A, float* out, int rows, int total_cols,
                    int c0, int width);

/**
 * @brief Selecciona una fila: out(1, cols) = A(row, :).
 * @param A          Puntero device a la matriz (rows × total_cols).
 * @param out        Puntero device a la salida (1 × cols).
 * @param row        Índice de la fila a extraer.
 * @param cols       Número de columnas a copiar.
 * @param total_cols Stride total de columnas de A.
 */
void select_row_fwd(const float* A, float* out, int row, int cols, int total_cols);

/**
 * @brief Softmax estable por filas.
 *
 * Para cada fila: resta el máximo, aplica exp(), y normaliza dividiendo por la suma.
 * Usa shared memory para las reducciones (max y sum).
 *
 * @param A    Puntero device a la entrada (rows × cols).
 * @param out  Puntero device a la salida (rows × cols), cada fila suma 1.0.
 * @param rows Número de filas.
 * @param cols Número de columnas (elementos por fila).
 */
void softmax_rows_fwd(const float* A, float* out, int rows, int cols);

/**
 * @brief LayerNorm forward: out = gamma × (A - mean) / √(var + ε) + beta.
 *
 * Calcula la normalización por fila y guarda valores intermedios para backward.
 *
 * @param A          Puntero device a la entrada (rows × cols).
 * @param gamma      Puntero device al factor de escala (1 × cols).
 * @param beta       Puntero device al factor de desplazamiento (1 × cols).
 * @param out        Puntero device a la salida (rows × cols).
 * @param mean_out   [out] Puntero device para guardar la media por fila (rows).
 * @param rstd_out   [out] Puntero device para guardar 1/√(var+ε) por fila (rows).
 * @param normed_out [out] Puntero device para guardar (A-mean)×rstd (rows × cols).
 * @param rows       Número de filas.
 * @param cols       Número de columnas.
 * @param eps        Épsilon de estabilidad numérica.
 */
void layer_norm_fwd(const float* A, const float* gamma, const float* beta,
                    float* out, float* mean_out, float* rstd_out,
                    float* normed_out, int rows, int cols, float eps);

/**
 * @brief GELU forward (aproximación tanh): GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
 * @param A    Puntero device a la entrada.
 * @param out  Puntero device a la salida.
 * @param size Número total de elementos.
 */
void gelu_fwd(const float* A, float* out, int size);

/**
 * @brief Softmax + Cross-Entropy combinados.
 *
 * Calcula softmax de los logits y luego la pérdida cross-entropy en un solo paso.
 *
 * @param logits      Puntero device a los logits (1 × num_classes).
 * @param label       Índice de la clase correcta.
 * @param num_classes Número de clases.
 * @param loss_out    [out] Puntero device donde se escribe la pérdida escalar.
 * @param probs_out   [out] Puntero device donde se escriben las probabilidades.
 */
void softmax_cross_entropy_fwd(const float* logits, int label, int num_classes,
                               float* loss_out, float* probs_out);

// ======================== Backward ========================

/**
 * @brief Backward de matmul respecto a A: dA(m,k) += dOut(m,n) × B(k,n)ᵀ.
 * @param dOut Puntero device al gradiente de la salida (m × n).
 * @param B    Puntero device a la matriz B del forward (k × n).
 * @param dA   Puntero device donde se acumulan los gradientes de A (m × k).
 */
void matmul_bwd_A(const float* dOut, const float* B, float* dA,
                  int m, int k, int n);

/**
 * @brief Backward de matmul respecto a B: dB(k,n) += A(m,k)ᵀ × dOut(m,n).
 * @param A    Puntero device a la matriz A del forward (m × k).
 * @param dOut Puntero device al gradiente de la salida (m × n).
 * @param dB   Puntero device donde se acumulan los gradientes de B (k × n).
 */
void matmul_bwd_B(const float* A, const float* dOut, float* dB,
                  int m, int k, int n);

/**
 * @brief Backward de suma: dA += dOut, dB += dOut (elemento a elemento).
 * @param dOut           Puntero device al gradiente de salida.
 * @param dA             Puntero device a los gradientes de A.
 * @param dB             Puntero device a los gradientes de B.
 * @param size           Número total de elementos.
 * @param A_requires_grad Si A necesita gradientes.
 * @param B_requires_grad Si B necesita gradientes.
 */
void add_bwd(const float* dOut, float* dA, float* dB, int size,
             bool A_requires_grad, bool B_requires_grad);

/**
 * @brief Backward de suma con broadcast: dA += dOut, dbias += Σ_filas(dOut).
 * @param dOut              Puntero device al gradiente de salida (rows × cols).
 * @param dA                Puntero device a los gradientes de A (rows × cols).
 * @param dbias             Puntero device a los gradientes del bias (1 × cols).
 * @param rows              Número de filas.
 * @param cols              Número de columnas.
 * @param A_requires_grad   Si A necesita gradientes.
 * @param bias_requires_grad Si bias necesita gradientes.
 */
void add_row_broadcast_bwd(const float* dOut, float* dA, float* dbias,
                           int rows, int cols,
                           bool A_requires_grad, bool bias_requires_grad);

/**
 * @brief Backward de scale: dA += dOut × scalar.
 * @param dOut   Puntero device al gradiente de salida.
 * @param scalar Factor de escala del forward.
 * @param dA     Puntero device donde se acumulan los gradientes.
 * @param size   Número total de elementos.
 */
void scale_bwd(const float* dOut, float scalar, float* dA, int size);

/**
 * @brief Backward de transpose: dA[i,j] += dOut[j,i].
 * @param dOut   Puntero device al gradiente de salida (cols_A × rows_A).
 * @param dA     Puntero device a los gradientes de A (rows_A × cols_A).
 * @param rows_A Filas de A (en el forward).
 * @param cols_A Columnas de A (en el forward).
 */
void transpose_bwd(const float* dOut, float* dA, int rows_A, int cols_A);

/**
 * @brief Backward de slice_cols: dA[:, c0:c0+width] += dOut.
 * @param dOut       Puntero device al gradiente de salida (rows × width).
 * @param dA         Puntero device a los gradientes de A (rows × total_cols).
 * @param rows       Número de filas.
 * @param total_cols Columnas totales de A.
 * @param c0         Columna inicial del slice.
 * @param width      Ancho del slice.
 */
void slice_cols_bwd(const float* dOut, float* dA, int rows, int total_cols,
                    int c0, int width);

/**
 * @brief Copia parte → destino en concat por columnas: out[:, offset:offset+width] = part.
 * @param part       Puntero device a la parte a copiar (rows × width).
 * @param out        Puntero device al destino (rows × total_cols).
 * @param rows       Número de filas.
 * @param width      Columnas de la parte.
 * @param total_cols Columnas totales del destino.
 * @param offset     Offset de columna en el destino.
 */
void concat_cols_copy(const float* part, float* out,
                      int rows, int width, int total_cols, int offset);

/**
 * @brief Backward de concat por columnas: dPart += dOut[:, offset:offset+width].
 * @param dOut       Puntero device al gradiente de salida (rows × total_cols).
 * @param dPart      Puntero device a los gradientes de la parte (rows × width).
 * @param rows       Número de filas.
 * @param width      Columnas de la parte.
 * @param total_cols Columnas totales.
 * @param offset     Offset de columna.
 */
void concat_cols_bwd_part(const float* dOut, float* dPart,
                          int rows, int width, int total_cols, int offset);

/**
 * @brief Copia filas en concat por filas: dst[start_row:, :] = src.
 * @param src       Puntero device a la fuente (rows × cols).
 * @param dst       Puntero device al destino.
 * @param rows      Número de filas a copiar.
 * @param cols      Número de columnas.
 * @param start_row Fila inicial en el destino.
 */
void concat_rows_copy(const float* src, float* dst, int rows, int cols, int start_row);

/**
 * @brief Backward de concat por filas: dSrc += dOut[start_row:, :].
 * @param dOut      Puntero device al gradiente de salida.
 * @param dSrc      Puntero device a los gradientes de la fuente.
 * @param rows      Número de filas.
 * @param cols      Número de columnas.
 * @param start_row Fila inicial en dOut.
 */
void concat_rows_bwd_part(const float* dOut, float* dSrc, int rows, int cols, int start_row);

/**
 * @brief Backward de select_row: dA[row, :] += dOut[0, :].
 * @param dOut Puntero device al gradiente de salida (1 × cols).
 * @param dA   Puntero device a los gradientes de A.
 * @param row  Índice de fila seleccionada en el forward.
 * @param cols Número de columnas.
 */
void select_row_bwd(const float* dOut, float* dA, int row, int cols);

/**
 * @brief Backward del softmax por filas.
 *
 * Usa el Jacobiano del softmax: dA[i,j] += s[i,j] × (dOut[i,j] − dot(dOut[i,:], s[i,:])).
 *
 * @param dOut        Puntero device al gradiente de salida (rows × cols).
 * @param softmax_out Puntero device a las probabilidades del forward (rows × cols).
 * @param dA          Puntero device a los gradientes de la entrada (rows × cols).
 * @param rows        Número de filas.
 * @param cols        Número de columnas.
 */
void softmax_rows_bwd(const float* dOut, const float* softmax_out,
                      float* dA, int rows, int cols);

/**
 * @brief Backward de LayerNorm.
 *
 * Propaga gradientes hacia la entrada A, gamma y beta.
 *
 * @param dOut              Puntero device al gradiente de la salida (rows × cols).
 * @param gamma             Puntero device a los pesos gamma del forward (1 × cols).
 * @param normed            Puntero device a los valores normalizados del forward (rows × cols).
 * @param rstd              Puntero device a los 1/√(var+ε) del forward (rows).
 * @param dA                Puntero device a los gradientes de A (rows × cols).
 * @param dgamma            Puntero device a los gradientes de gamma (1 × cols).
 * @param dbeta             Puntero device a los gradientes de beta (1 × cols).
 * @param rows              Número de filas.
 * @param cols              Número de columnas.
 * @param A_requires_grad   Si A necesita gradientes.
 * @param gamma_requires_grad Si gamma necesita gradientes.
 * @param beta_requires_grad  Si beta necesita gradientes.
 */
void layer_norm_bwd(const float* dOut, const float* gamma,
                    const float* normed, const float* rstd,
                    float* dA, float* dgamma, float* dbeta,
                    int rows, int cols,
                    bool A_requires_grad, bool gamma_requires_grad,
                    bool beta_requires_grad);

/**
 * @brief Backward de GELU: dA += dOut × dGELU/dx.
 * @param dOut   Puntero device al gradiente de salida.
 * @param A_data Puntero device a los datos originales de A (del forward).
 * @param dA     Puntero device a los gradientes de A.
 * @param size   Número total de elementos.
 */
void gelu_bwd(const float* dOut, const float* A_data, float* dA, int size);

/**
 * @brief Backward de Softmax + Cross-Entropy: dLogits[j] += upstream × (probs[j] − target[j]).
 * @param upstream    Gradiente upstream (escalar, típicamente 1.0).
 * @param probs       Puntero device a las probabilidades del forward.
 * @param label       Índice de la clase correcta.
 * @param num_classes Número de clases.
 * @param dLogits     Puntero device a los gradientes de los logits.
 */
void softmax_cross_entropy_bwd(float upstream, const float* probs, int label,
                               int num_classes, float* dLogits);

// ======================== Optimizador ========================

/**
 * @brief Un paso de AdamW ejecutado directamente en GPU.
 *
 * Actualiza parámetros, momentos m y v, y aplica weight decay en un solo kernel.
 *
 * @param param        Puntero device a los parámetros del modelo.
 * @param grad         Puntero device a los gradientes.
 * @param m            Puntero device al primer momento (promedio de gradientes).
 * @param v            Puntero device al segundo momento (promedio de gradientes²).
 * @param lr           Learning rate.
 * @param beta1        Coeficiente β₁.
 * @param beta2        Coeficiente β₂.
 * @param eps          Épsilon de estabilidad.
 * @param bc1          Factor de bias correction para m: (1 − β₁ᵗ).
 * @param bc2          Factor de bias correction para v: (1 − β₂ᵗ).
 * @param grad_scale   Factor de escala de gradientes (típicamente 1/batch_size).
 * @param weight_decay Coeficiente de weight decay λ.
 * @param size         Número total de parámetros.
 */
void adam_step(float* param, float* grad, float* m, float* v,
              float lr, float beta1, float beta2, float eps,
              float bc1, float bc2, float grad_scale, float weight_decay,
              int size);

/**
 * @brief Wrapper de cudaMemset para poner a cero gradientes.
 * @param ptr  Puntero device al buffer a limpiar.
 * @param size Número de floats a poner a cero.
 */
void zero_memory(float* ptr, int size);

}} // namespace vit::cuda

#endif // USE_CUDA

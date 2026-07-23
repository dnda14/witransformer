// cuda_kernels.cuh — Declaraciones de funciones wrapper que lanzan los kernels CUDA.
//
// Cada función recibe punteros device (float*) y dimensiones, calcula la
// configuración de bloques/hilos y lanza el kernel correspondiente.
// Se usa desde ops.hpp cuando USE_CUDA está definido.

#pragma once

#ifdef USE_CUDA

#include <cstddef>

namespace vit { namespace cuda {

// ======================== Forward ========================

// out(m,n) = A(m,k) * B(k,n)
void matmul_fwd(const float* A, const float* B, float* out,
                int m, int k, int n);

// out = A + B (elemento a elemento, total elementos = size)
void add_fwd(const float* A, const float* B, float* out, int size);

// out(rows, cols) = A(rows, cols) + bias(1, cols)  (broadcast por fila)
void add_row_broadcast_fwd(const float* A, const float* bias, float* out,
                           int rows, int cols);

// out = A * scalar
void scale_fwd(const float* A, float scalar, float* out, int size);

// out(cols, rows) = A(rows, cols)^T
void transpose_fwd(const float* A, float* out, int rows, int cols);

// out(rows, width) = A(rows, total_cols)[:, c0:c0+width]
void slice_cols_fwd(const float* A, float* out, int rows, int total_cols,
                    int c0, int width);

// out(1, cols) = A(row, :)
void select_row_fwd(const float* A, float* out, int row, int cols, int total_cols);

// Softmax estable por filas: out(rows, cols)
void softmax_rows_fwd(const float* A, float* out, int rows, int cols);

// LayerNorm forward: out = gamma * (A - mean) / sqrt(var + eps) + beta
// mean_out y rstd_out se guardan para backward (tamaño = rows).
// normed_out guarda (A - mean) * rstd para backward (tamaño = rows*cols).
void layer_norm_fwd(const float* A, const float* gamma, const float* beta,
                    float* out, float* mean_out, float* rstd_out,
                    float* normed_out, int rows, int cols, float eps);

// GELU forward (aproximación tanh)
void gelu_fwd(const float* A, float* out, int size);

// Softmax + cross-entropy combinados.
// Escribe la pérdida escalar en loss_out[0] y las probabilidades en probs_out.
void softmax_cross_entropy_fwd(const float* logits, int label, int num_classes,
                               float* loss_out, float* probs_out);

// ======================== Backward ========================

// dA(m,k) += dOut(m,n) * B(k,n)^T
void matmul_bwd_A(const float* dOut, const float* B, float* dA,
                  int m, int k, int n);

// dB(k,n) += A(m,k)^T * dOut(m,n)
void matmul_bwd_B(const float* A, const float* dOut, float* dB,
                  int m, int k, int n);

// dA += dOut, dB += dOut (elemento a elemento)
void add_bwd(const float* dOut, float* dA, float* dB, int size,
             bool A_requires_grad, bool B_requires_grad);

// dA(rows,cols) += dOut(rows,cols)
// dbias(1,cols) += sum_rows(dOut)
void add_row_broadcast_bwd(const float* dOut, float* dA, float* dbias,
                           int rows, int cols,
                           bool A_requires_grad, bool bias_requires_grad);

// dA += dOut * scalar
void scale_bwd(const float* dOut, float scalar, float* dA, int size);

// dA(i,j) += dOut(j,i)
void transpose_bwd(const float* dOut, float* dA, int rows_A, int cols_A);

// dA[:, c0:c0+width] += dOut
void slice_cols_bwd(const float* dOut, float* dA, int rows, int total_cols,
                    int c0, int width);

// Copia part(rows, width) -> out(rows, total_cols) en columnas [offset, offset+width)
void concat_cols_copy(const float* part, float* out,
                      int rows, int width, int total_cols, int offset);

// dPart(rows, width) += dOut(rows, total_cols)[:, offset:offset+width]
void concat_cols_bwd_part(const float* dOut, float* dPart,
                          int rows, int width, int total_cols, int offset);

void concat_rows_copy(const float* src, float* dst, int rows, int cols, int start_row);
void concat_rows_bwd_part(const float* dOut, float* dSrc, int rows, int cols, int start_row);

// dA(row, :) += dOut(0, :)
void select_row_bwd(const float* dOut, float* dA, int row, int cols);

// Backward del softmax por filas.
// softmax_out contiene las probabilidades del forward.
void softmax_rows_bwd(const float* dOut, const float* softmax_out,
                      float* dA, int rows, int cols);

// Backward de LayerNorm.
void layer_norm_bwd(const float* dOut, const float* gamma,
                    const float* normed, const float* rstd,
                    float* dA, float* dgamma, float* dbeta,
                    int rows, int cols,
                    bool A_requires_grad, bool gamma_requires_grad,
                    bool beta_requires_grad);

// Backward de GELU.
void gelu_bwd(const float* dOut, const float* A_data, float* dA, int size);

// Backward de softmax + cross-entropy.
void softmax_cross_entropy_bwd(float upstream, const float* probs, int label,
                               int num_classes, float* dLogits);

// ======================== Optimizador ========================

// Un paso de Adam directamente en GPU.
void adam_step(float* param, float* grad, float* m, float* v,
              float lr, float beta1, float beta2, float eps,
              float bc1, float bc2, float grad_scale, int size);

// cudaMemset wrapper para zero_grad
void zero_memory(float* ptr, int size);

}} // namespace vit::cuda

#endif // USE_CUDA

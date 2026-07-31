/**
 * @file cuda_kernels.cuh
 * @brief Declaraciones de las funciones wrapper que lanzan los kernels CUDA.
 */
#pragma once

#ifdef USE_CUDA
#include <cstddef>

namespace vit { namespace cuda {

// ======================== Forward ========================

void matmul_fwd(const float* A, const float* B, float* out,
                int batch, int m, int k, int n,
                bool B_is_batched);

void add_fwd(const float* A, const float* B, float* out, 
             int batch, int size_per_batch, bool B_is_batched);

void add_row_broadcast_fwd(const float* A, const float* bias, float* out,
                           int batch, int rows, int cols);

void scale_fwd(const float* A, float scalar, float* out, int size);

void transpose_fwd(const float* A, float* out, int batch, int rows, int cols);

void slice_cols_fwd(const float* A, float* out, int batch, int rows, int total_cols,
                    int c0, int width);

void select_row_fwd(const float* A, float* out, int batch, int row, int cols, int total_cols);

void softmax_rows_fwd(const float* A, float* out, int total_rows, int cols);

void layer_norm_fwd(const float* A, const float* gamma, const float* beta,
                    float* out, float* mean_out, float* rstd_out,
                    float* normed_out, int total_rows, int cols, float eps);

void gelu_fwd(const float* A, float* out, int size);

void softmax_cross_entropy_fwd(const float* logits, const int* labels, int batch, int num_classes,
                               float* loss_out, float* probs_out);

// ======================== Backward ========================

void matmul_bwd_A(const float* dOut, const float* B, float* dA,
                  int batch, int m, int k, int n, bool B_is_batched);

void matmul_bwd_B(const float* A, const float* dOut, float* dB,
                  int batch, int m, int k, int n, bool B_is_batched);

void add_bwd(const float* dOut, float* dA, float* dB, 
             int batch, int size_per_batch, bool B_is_batched,
             bool A_requires_grad, bool B_requires_grad);

void add_row_broadcast_bwd(const float* dOut, float* dA, float* dbias,
                           int batch, int rows, int cols,
                           bool A_requires_grad, bool bias_requires_grad);

void scale_bwd(const float* dOut, float scalar, float* dA, int size);

void transpose_bwd(const float* dOut, float* dA, int batch, int rows_A, int cols_A);

void slice_cols_bwd(const float* dOut, float* dA, int batch, int rows, int total_cols,
                    int c0, int width);

void concat_cols_copy(const float* part, float* out,
                      int batch, int rows, int width, int total_cols, int offset);

void concat_cols_bwd_part(const float* dOut, float* dPart,
                          int batch, int rows, int width, int total_cols, int offset);

void concat_rows_copy(const float* src, float* dst, int batch, int rows, int cols, int start_row, int dst_total_rows, bool src_batched);

void concat_rows_bwd_part(const float* dOut, float* dSrc, int batch, int rows, int cols, int start_row, int dOut_total_rows, bool src_batched);

void select_row_bwd(const float* dOut, float* dA, int batch, int row, int cols, int total_rows);

void softmax_rows_bwd(const float* dOut, const float* softmax_out,
                      float* dA, int total_rows, int cols);

void layer_norm_bwd(const float* dOut, const float* gamma,
                    const float* normed, const float* rstd,
                    float* dA, float* dgamma, float* dbeta,
                    int total_rows, int cols,
                    bool A_requires_grad, bool gamma_requires_grad,
                    bool beta_requires_grad);

void gelu_bwd(const float* dOut, const float* A_data, float* dA, int size);

void softmax_cross_entropy_bwd(float upstream, const float* probs, const int* labels,
                               int batch, int num_classes, float* dLogits);

// ======================== Optimizador ========================

void adam_step(float* param, float* grad, float* m, float* v,
               float lr, float beta1, float beta2, float eps,
               float bc1, float bc2, float grad_scale, float weight_decay,
               int size);

void zero_memory(float* ptr, int size);

}} // namespace vit::cuda

#endif // USE_CUDA

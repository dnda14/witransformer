/**
 * @file ops.hpp
 * @brief Operaciones matemáticas diferenciables para Tensores (Batch Mode).
 */

#pragma once

#include "tensor.hpp"
#include <cmath>
#include <vector>

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#endif

namespace vit {

inline Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A->cols != B->rows) throw std::runtime_error("matmul: dimensiones incompatibles");
    int batch = std::max(A->batch, B->batch);
    bool B_batched = (B->batch > 1);
    int m = A->rows, k = A->cols, n = B->cols;
    auto out = make_tensor(batch, m, n, A->requires_grad || B->requires_grad);
#ifdef USE_CUDA
    cuda::matmul_fwd(A->d_data, B->d_data, out->d_data, batch, m, k, n, B_batched);
#else
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        int b_B = B_batched ? b : 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float s = 0.0f;
                for (int p = 0; p < k; ++p) s += A->at(b, i, p) * B->at(b_B, p, j);
                out->at(b, i, j) = s;
            }
        }
    }
#endif
    out->parents = {A, B};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, B, out_raw, batch, m, k, n, B_batched]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::matmul_bwd_A(out_raw->d_grad, B->d_data, A->d_grad, batch, m, k, n, B_batched);
        if (B->requires_grad) cuda::matmul_bwd_B(A->d_data, out_raw->d_grad, B->d_grad, batch, m, k, n, B_batched);
#else
        if (A->requires_grad) {
            #pragma omp parallel for
            for (int b = 0; b < batch; ++b) {
                int b_B = B_batched ? b : 0;
                for (int i = 0; i < m; ++i) {
                    for (int p = 0; p < k; ++p) {
                        float s = 0.0f;
                        for (int j = 0; j < n; ++j) s += out_raw->g(b, i, j) * B->at(b_B, p, j);
                        A->g(b, i, p) += s;
                    }
                }
            }
        }
        if (B->requires_grad) {
            #pragma omp parallel for
            for (int b = 0; b < batch; ++b) {
                int b_B = B_batched ? b : 0;
                for (int p = 0; p < k; ++p) {
                    for (int j = 0; j < n; ++j) {
                        float s = 0.0f;
                        for (int i = 0; i < m; ++i) s += A->at(b, i, p) * out_raw->g(b, i, j);
                        #pragma omp atomic
                        B->g(b_B, p, j) += s;
                    }
                }
            }
        }
#endif
    };
    return out;
}

inline Tensor add(const Tensor& A, const Tensor& B) {
    if (A->rows != B->rows || A->cols != B->cols)
        throw std::runtime_error("add: dimensiones distintas");
    int batch = std::max(A->batch, B->batch);
    bool B_batched = (B->batch > 1);
    auto out = make_tensor(batch, A->rows, A->cols, A->requires_grad || B->requires_grad);
    int size_per_batch = A->rows * A->cols;
#ifdef USE_CUDA
    cuda::add_fwd(A->d_data, B->d_data, out->d_data, batch, size_per_batch, B_batched);
#else
    for (int b = 0; b < batch; ++b) {
        int b_B = B_batched ? b : 0;
        for (int i = 0; i < size_per_batch; ++i) 
            out->data[b * size_per_batch + i] = A->data[b * size_per_batch + i] + B->data[b_B * size_per_batch + i];
    }
#endif
    out->parents = {A, B};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, B, out_raw, batch, size_per_batch, B_batched]() {
#ifdef USE_CUDA
        cuda::add_bwd(out_raw->d_grad, A->d_grad, B->d_grad, batch, size_per_batch, B_batched,
                      A->requires_grad, B->requires_grad);
#else
        if (A->requires_grad) {
            for (int b = 0; b < batch; ++b) {
                for (int i = 0; i < size_per_batch; ++i) 
                    A->grad[b * size_per_batch + i] += out_raw->grad[b * size_per_batch + i];
            }
        }
        if (B->requires_grad) {
            for (int b = 0; b < batch; ++b) {
                int b_B = B_batched ? b : 0;
                for (int i = 0; i < size_per_batch; ++i) 
                    B->grad[b_B * size_per_batch + i] += out_raw->grad[b * size_per_batch + i];
            }
        }
#endif
    };
    return out;
}

inline Tensor add_row_broadcast(const Tensor& A, const Tensor& bias) {
    if (bias->rows != 1 || A->cols != bias->cols)
        throw std::runtime_error("add_row_broadcast: forma de bias incorrecta");
    int batch = A->batch;
    auto out = make_tensor(batch, A->rows, A->cols, A->requires_grad || bias->requires_grad);
#ifdef USE_CUDA
    cuda::add_row_broadcast_fwd(A->d_data, bias->d_data, out->d_data, batch, A->rows, A->cols);
#else
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < A->rows; ++i)
            for (int j = 0; j < A->cols; ++j)
                out->at(b, i, j) = A->at(b, i, j) + bias->at(0, 0, j);
#endif
    out->parents = {A, bias};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, bias, out_raw, batch]() {
#ifdef USE_CUDA
        cuda::add_row_broadcast_bwd(out_raw->d_grad, A->d_grad, bias->d_grad,
                                    batch, out_raw->rows, out_raw->cols,
                                    A->requires_grad, bias->requires_grad);
#else
        if (A->requires_grad)
            for (size_t i = 0; i < out_raw->data.size(); ++i) A->grad[i] += out_raw->grad[i];
        if (bias->requires_grad)
            for (int b = 0; b < batch; ++b)
                for (int i = 0; i < out_raw->rows; ++i)
                    for (int j = 0; j < out_raw->cols; ++j)
                        bias->g(0, 0, j) += out_raw->g(b, i, j);
#endif
    };
    return out;
}

inline Tensor scale(const Tensor& A, float s) {
    auto out = make_tensor(A->batch, A->rows, A->cols, A->requires_grad);
    int size = A->batch * A->rows * A->cols;
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
    int batch = A->batch;
    auto out = make_tensor(batch, A->cols, A->rows, A->requires_grad);
#ifdef USE_CUDA
    cuda::transpose_fwd(A->d_data, out->d_data, batch, A->rows, A->cols);
#else
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < A->rows; ++i)
            for (int j = 0; j < A->cols; ++j)
                out->at(b, j, i) = A->at(b, i, j);
#endif
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, batch]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::transpose_bwd(out_raw->d_grad, A->d_grad, batch, A->rows, A->cols);
#else
        if (!A->requires_grad) return;
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < A->rows; ++i)
                for (int j = 0; j < A->cols; ++j)
                    A->g(b, i, j) += out_raw->g(b, j, i);
#endif
    };
    return out;
}

inline Tensor slice_cols(const Tensor& A, int c0, int width) {
    int batch = A->batch;
    auto out = make_tensor(batch, A->rows, width, A->requires_grad);
#ifdef USE_CUDA
    cuda::slice_cols_fwd(A->d_data, out->d_data, batch, A->rows, A->cols, c0, width);
#else
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < A->rows; ++i)
            for (int j = 0; j < width; ++j)
                out->at(b, i, j) = A->at(b, i, c0 + j);
#endif
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, batch, c0, width]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::slice_cols_bwd(out_raw->d_grad, A->d_grad, batch, A->rows, A->cols, c0, width);
#else
        if (!A->requires_grad) return;
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < A->rows; ++i)
                for (int j = 0; j < width; ++j)
                    A->g(b, i, c0 + j) += out_raw->g(b, i, j);
#endif
    };
    return out;
}

inline Tensor concat_cols(const std::vector<Tensor>& parts) {
    int batch = parts[0]->batch;
    int rows = parts[0]->rows;
    int total_cols = 0;
    bool rg = false;
    for (auto& p : parts) { total_cols += p->cols; rg = rg || p->requires_grad; }
    auto out = make_tensor(batch, rows, total_cols, rg);
#ifdef USE_CUDA
    int offset = 0;
    for (auto& p : parts) {
        cuda::concat_cols_copy(p->d_data, out->d_data, batch, rows, p->cols, total_cols, offset);
        offset += p->cols;
    }
#else
    int offset = 0;
    for (auto& p : parts) {
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < p->cols; ++j)
                    out->at(b, i, offset + j) = p->at(b, i, j);
        offset += p->cols;
    }
#endif
    out->parents = parts;
    TensorImpl* out_raw = out.get();
    out->backward_fn = [parts, out_raw, batch, rows, total_cols]() {
#ifdef USE_CUDA
        int off = 0;
        for (auto& p : parts) {
            if (p->requires_grad)
                cuda::concat_cols_bwd_part(out_raw->d_grad, p->d_grad, batch, rows, p->cols, total_cols, off);
            off += p->cols;
        }
#else
        int off = 0;
        for (auto& p : parts) {
            if (p->requires_grad)
                for (int b = 0; b < batch; ++b)
                    for (int i = 0; i < rows; ++i)
                        for (int j = 0; j < p->cols; ++j)
                            p->g(b, i, j) += out_raw->g(b, i, off + j);
            off += p->cols;
        }
#endif
    };
    return out;
}

inline Tensor select_row(const Tensor& A, int row) {
    int batch = A->batch;
    auto out = make_tensor(batch, 1, A->cols, A->requires_grad);
#ifdef USE_CUDA
    cuda::select_row_fwd(A->d_data, out->d_data, batch, row, A->cols, A->rows);
#else
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < A->cols; ++j) out->at(b, 0, j) = A->at(b, row, j);
#endif
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, batch, row]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::select_row_bwd(out_raw->d_grad, A->d_grad, batch, row, A->cols, A->rows);
#else
        if (!A->requires_grad) return;
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < A->cols; ++j) A->g(b, row, j) += out_raw->g(b, 0, j);
#endif
    };
    return out;
}

inline Tensor softmax_rows(const Tensor& A) {
    int batch = A->batch;
    auto out = make_tensor(batch, A->rows, A->cols, A->requires_grad);
    int total_rows = batch * A->rows;
#ifdef USE_CUDA
    cuda::softmax_rows_fwd(A->d_data, out->d_data, total_rows, A->cols);
#else
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < A->rows; ++i) {
            float mx = A->at(b, i, 0);
            for (int j = 1; j < A->cols; ++j) mx = std::max(mx, A->at(b, i, j));
            float sum = 0.0f;
            for (int j = 0; j < A->cols; ++j) {
                float e = std::exp(A->at(b, i, j) - mx);
                out->at(b, i, j) = e;
                sum += e;
            }
            for (int j = 0; j < A->cols; ++j) out->at(b, i, j) /= sum;
        }
    }
#endif
    out->parents = {A};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [A, out_raw, batch, total_rows]() {
#ifdef USE_CUDA
        if (A->requires_grad) cuda::softmax_rows_bwd(out_raw->d_grad, out_raw->d_data, A->d_grad, total_rows, A->cols);
#else
        if (!A->requires_grad) return;
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < A->rows; ++i) {
                float dot = 0.0f;
                for (int j = 0; j < A->cols; ++j) dot += out_raw->g(b, i, j) * out_raw->at(b, i, j);
                for (int j = 0; j < A->cols; ++j)
                    A->g(b, i, j) += out_raw->at(b, i, j) * (out_raw->g(b, i, j) - dot);
            }
        }
#endif
    };
    return out;
}

inline Tensor layer_norm(const Tensor& A, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f) {
    int batch = A->batch;
    int rows = A->rows, cols = A->cols;
    int total_rows = batch * rows;
    auto out = make_tensor(batch, rows, cols, true);

#ifdef USE_CUDA
    float* d_mean = nullptr; cudaMalloc(&d_mean, total_rows * sizeof(float));
    float* d_rstd = nullptr; cudaMalloc(&d_rstd, total_rows * sizeof(float));
    auto normed = make_tensor(batch, rows, cols, false);

    cuda::layer_norm_fwd(A->d_data, gamma->d_data, beta->d_data,
                         out->d_data, d_mean, d_rstd, normed->d_data,
                         total_rows, cols, eps);

    out->parents = {A, gamma, beta};
    TensorImpl* out_raw = out.get();
    TensorImpl* normed_raw = normed.get();
    auto mean_guard = std::shared_ptr<float>(d_mean, [](float* p) { cudaFree(p); });
    auto rstd_guard = std::shared_ptr<float>(d_rstd, [](float* p) { cudaFree(p); });
    out->backward_fn = [A, gamma, beta, out_raw, normed, normed_raw,
                         mean_guard, rstd_guard, total_rows, cols]() {
        (void)normed;
        cuda::layer_norm_bwd(out_raw->d_grad, gamma->d_data,
                             normed_raw->d_data, rstd_guard.get(),
                             A->d_grad, gamma->d_grad, beta->d_grad,
                             total_rows, cols,
                             A->requires_grad, gamma->requires_grad, beta->requires_grad);
    };
#else
    std::vector<float> mean(total_rows), rstd(total_rows); 
    auto normed = make_tensor(batch, rows, cols, false); 
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < rows; ++i) {
            float m = 0.0f;
            for (int j = 0; j < cols; ++j) m += A->at(b, i, j);
            m /= cols;
            float var = 0.0f;
            for (int j = 0; j < cols; ++j) { float d = A->at(b, i, j) - m; var += d * d; }
            var /= cols;
            float rs = 1.0f / std::sqrt(var + eps);
            mean[b * rows + i] = m; rstd[b * rows + i] = rs;
            for (int j = 0; j < cols; ++j) {
                float nh = (A->at(b, i, j) - m) * rs;
                normed->at(b, i, j) = nh;
                out->at(b, i, j) = nh * gamma->at(0, 0, j) + beta->at(0, 0, j);
            }
        }
    }
    out->parents = {A, gamma, beta};
    TensorImpl* out_raw = out.get();
    TensorImpl* normed_raw = normed.get();
    out->backward_fn = [A, gamma, beta, out_raw, normed, normed_raw, mean, rstd, batch, rows, cols]() {
        (void)normed; 
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < rows; ++i) {
                float rs = rstd[b * rows + i];
                float sum_dy = 0.0f, sum_dy_nh = 0.0f;
                for (int j = 0; j < cols; ++j) {
                    float dy = out_raw->g(b, i, j) * gamma->at(0, 0, j); 
                    sum_dy += dy;
                    sum_dy_nh += dy * normed_raw->at(b, i, j);
                }
                if (A->requires_grad) {
                    for (int j = 0; j < cols; ++j) {
                        float dy = out_raw->g(b, i, j) * gamma->at(0, 0, j);
                        float dx = rs * (dy - sum_dy / cols - normed_raw->at(b, i, j) * sum_dy_nh / cols);
                        A->g(b, i, j) += dx;
                    }
                }
                if (gamma->requires_grad)
                    for (int j = 0; j < cols; ++j) gamma->g(0, 0, j) += out_raw->g(b, i, j) * normed_raw->at(b, i, j);
                if (beta->requires_grad)
                    for (int j = 0; j < cols; ++j) beta->g(0, 0, j) += out_raw->g(b, i, j);
            }
        }
    };
#endif
    return out;
}

inline Tensor gelu(const Tensor& A) {
    auto out = make_tensor(A->batch, A->rows, A->cols, A->requires_grad);
    int size = A->batch * A->rows * A->cols;
#ifdef USE_CUDA
    cuda::gelu_fwd(A->d_data, out->d_data, size);
#else
    const float k0 = 0.7978845608f; 
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

inline Tensor softmax_cross_entropy(const Tensor& logits, const std::vector<int>& labels, std::vector<float>* probs_out = nullptr) {
    int batch = logits->batch;
    int n = logits->cols;

#ifdef USE_CUDA
    size_t probs_bytes = batch * n * sizeof(float);
    float* d_probs = CachingAllocator::allocate(probs_bytes);
    int* d_labels = nullptr; cudaMalloc(&d_labels, batch * sizeof(int));
    cudaMemcpy(d_labels, labels.data(), batch * sizeof(int), cudaMemcpyHostToDevice);

    auto out = make_tensor(1, 1, 1, true); // scalar loss
    cuda::softmax_cross_entropy_fwd(logits->d_data, d_labels, batch, n, out->d_data, d_probs);

    out->to_host();
    if (probs_out) {
        probs_out->resize(batch * n);
        cudaMemcpy(probs_out->data(), d_probs, probs_bytes, cudaMemcpyDeviceToHost);
    }

    out->parents = {logits};
    TensorImpl* out_raw = out.get();
    auto probs_guard = std::shared_ptr<float>(d_probs, [probs_bytes](float* p) { 
        CachingAllocator::free(p, probs_bytes); 
    });
    auto labels_guard = std::shared_ptr<int>(d_labels, [](int* p) { cudaFree(p); });
    out->backward_fn = [logits, out_raw, probs_guard, labels_guard, batch, n]() {
        if (!logits->requires_grad) return;
        float upstream = out_raw->grad[0]; 
        cuda::softmax_cross_entropy_bwd(upstream, probs_guard.get(), labels_guard.get(), batch, n, logits->d_grad);
    };
#else
    std::vector<float> probs(batch * n);
    float total_loss = 0.0f;
    for (int b = 0; b < batch; ++b) {
        float mx = logits->at(b, 0, 0);
        for (int j = 1; j < n; ++j) mx = std::max(mx, logits->at(b, 0, j));
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) { 
            probs[b * n + j] = std::exp(logits->at(b, 0, j) - mx); 
            sum += probs[b * n + j]; 
        }
        for (int j = 0; j < n; ++j) probs[b * n + j] /= sum;
        total_loss += -std::log(std::max(probs[b * n + labels[b]], 1e-9f));
    }
    
    if (probs_out) *probs_out = probs;

    auto out = make_tensor(1, 1, 1, true);
    out->data[0] = total_loss / batch;
    out->parents = {logits};
    TensorImpl* out_raw = out.get();
    out->backward_fn = [logits, out_raw, probs, labels, batch, n]() {
        if (!logits->requires_grad) return;
        float upstream = out_raw->grad[0];
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < n; ++j) {
                float target = (j == labels[b]) ? 1.0f : 0.0f;
                logits->g(b, 0, j) += (upstream / batch) * (probs[b * n + j] - target);
            }
        }
    };
#endif
    return out;
}

} // namespace vit

/**
 * @file cuda_kernels.cu
 * @brief Implementación de todos los kernels CUDA para el Vision Transformer (Batch mode).
 */

#ifdef USE_CUDA

#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err));                                  \
    }                                                                      \
} while(0)

static constexpr int TILE_SIZE = 16;
static constexpr int BLOCK_SIZE = 256;

namespace vit { namespace cuda {

// ============================================================================
//                              MATMUL
// ============================================================================

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ out,
                              int batch, int m, int k, int n, bool B_is_batched) {
    int b = blockIdx.z;
    if (b >= batch) return;

    A += b * m * k;
    if (B_is_batched) B += b * k * n;
    out += b * m * n;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? A[row * k + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n)
        out[row * n + col] = sum;
}

void matmul_fwd(const float* A, const float* B, float* out, int batch, int m, int k, int n, bool B_is_batched) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE, batch);
    matmul_kernel<<<grid, block>>>(A, B, out, batch, m, k, n, B_is_batched);
}

__global__ void matmul_bwd_A_kernel(const float* __restrict__ dOut,
                                     const float* __restrict__ B,
                                     float* __restrict__ dA,
                                     int batch, int m, int k, int n, bool B_is_batched) {
    int b = blockIdx.z;
    if (b >= batch) return;

    dOut += b * m * n;
    if (B_is_batched) B += b * k * n;
    dA += b * m * k;

    __shared__ float sDO[TILE_SIZE][TILE_SIZE];
    __shared__ float sBT[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int doCol = t * TILE_SIZE + threadIdx.x;
        int btRow = t * TILE_SIZE + threadIdx.y;
        sDO[threadIdx.y][threadIdx.x] = (row < m && doCol < n) ? dOut[row * n + doCol] : 0.0f;
        sBT[threadIdx.y][threadIdx.x] = (btRow < n && col < k) ? B[col * n + btRow] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += sDO[threadIdx.y][i] * sBT[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < k)
        atomicAdd(&dA[row * k + col], sum);
}

void matmul_bwd_A(const float* dOut, const float* B, float* dA, int batch, int m, int k, int n, bool B_is_batched) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((k + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE, batch);
    matmul_bwd_A_kernel<<<grid, block>>>(dOut, B, dA, batch, m, k, n, B_is_batched);
}

__global__ void matmul_bwd_B_kernel(const float* __restrict__ A,
                                     const float* __restrict__ dOut,
                                     float* __restrict__ dB,
                                     int batch, int m, int k, int n, bool B_is_batched) {
    int b = blockIdx.z;
    if (b >= batch) return;

    A += b * m * k;
    dOut += b * m * n;
    if (B_is_batched) dB += b * k * n;
    // Si B_is_batched == false, dB se comparte entre todos los batches.

    __shared__ float sAT[TILE_SIZE][TILE_SIZE];
    __shared__ float sDO[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int atCol = t * TILE_SIZE + threadIdx.x;
        int doRow = t * TILE_SIZE + threadIdx.y;
        sAT[threadIdx.y][threadIdx.x] = (row < k && atCol < m) ? A[atCol * k + row] : 0.0f;
        sDO[threadIdx.y][threadIdx.x] = (doRow < m && col < n) ? dOut[doRow * n + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += sAT[threadIdx.y][i] * sDO[i][threadIdx.x];
        __syncthreads();
    }
    if (row < k && col < n)
        atomicAdd(&dB[row * n + col], sum);
}

void matmul_bwd_B(const float* A, const float* dOut, float* dB, int batch, int m, int k, int n, bool B_is_batched) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (k + TILE_SIZE - 1) / TILE_SIZE, batch);
    matmul_bwd_B_kernel<<<grid, block>>>(A, dOut, dB, batch, m, k, n, B_is_batched);
}

// ============================================================================
//                         ELEMENTWISE ADD
// ============================================================================

__global__ void add_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ out, int batch, int size_per_batch, bool B_is_batched) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size_per_batch) {
        int i = idx % size_per_batch;
        float b_val = B_is_batched ? B[idx] : B[i];
        out[idx] = A[idx] + b_val;
    }
}

void add_fwd(const float* A, const float* B, float* out, int batch, int size_per_batch, bool B_is_batched) {
    int total = batch * size_per_batch;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<<<grid, BLOCK_SIZE>>>(A, B, out, batch, size_per_batch, B_is_batched);
}

__global__ void add_bwd_kernel(const float* __restrict__ dOut, float* dA, float* dB,
                               int batch, int size_per_batch, bool B_is_batched, bool doA, bool doB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size_per_batch) {
        float g = dOut[idx];
        if (doA) atomicAdd(&dA[idx], g);
        if (doB) {
            int b_idx = B_is_batched ? idx : (idx % size_per_batch);
            atomicAdd(&dB[b_idx], g);
        }
    }
}

void add_bwd(const float* dOut, float* dA, float* dB, int batch, int size_per_batch, bool B_is_batched,
             bool A_rg, bool B_rg) {
    int total = batch * size_per_batch;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dA, dB, batch, size_per_batch, B_is_batched, A_rg, B_rg);
}

// ============================================================================
//                       ADD ROW BROADCAST (bias)
// ============================================================================

__global__ void add_row_broadcast_kernel(const float* __restrict__ A,
                                         const float* __restrict__ bias,
                                         float* __restrict__ out,
                                         int batch, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * rows * cols) {
        int j = idx % cols;
        out[idx] = A[idx] + bias[j];
    }
}

void add_row_broadcast_fwd(const float* A, const float* bias, float* out,
                           int batch, int rows, int cols) {
    int size = batch * rows * cols;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_row_broadcast_kernel<<<grid, BLOCK_SIZE>>>(A, bias, out, batch, rows, cols);
}

__global__ void add_row_broadcast_bwd_bias_kernel(const float* __restrict__ dOut,
                                                   float* __restrict__ dbias,
                                                   int batch, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float sum = 0.0f;
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < rows; ++i) {
                sum += dOut[b * rows * cols + i * cols + j];
            }
        }
        atomicAdd(&dbias[j], sum);
    }
}

void add_row_broadcast_bwd(const float* dOut, float* dA, float* dbias,
                           int batch, int rows, int cols,
                           bool A_rg, bool bias_rg) {
    int size = batch * rows * cols;
    if (A_rg) {
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dA, nullptr, batch, rows * cols, true, true, false);
    }
    if (bias_rg) {
        int grid = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_row_broadcast_bwd_bias_kernel<<<grid, BLOCK_SIZE>>>(dOut, dbias, batch, rows, cols);
    }
}

// ============================================================================
//                              SCALE
// ============================================================================

__global__ void scale_kernel(const float* __restrict__ A, float s,
                             float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = A[idx] * s;
}

void scale_fwd(const float* A, float scalar, float* out, int size) {
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_kernel<<<grid, BLOCK_SIZE>>>(A, scalar, out, size);
}

__global__ void scale_bwd_kernel(const float* __restrict__ dOut, float s,
                                 float* __restrict__ dA, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) atomicAdd(&dA[idx], dOut[idx] * s);
}

void scale_bwd(const float* dOut, float scalar, float* dA, int size) {
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, scalar, dA, size);
}

// ============================================================================
//                            TRANSPOSE
// ============================================================================

__global__ void transpose_kernel(const float* __restrict__ A, float* __restrict__ out,
                                 int batch, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size_per_batch = rows * cols;
    if (idx < batch * size_per_batch) {
        int b = idx / size_per_batch;
        int rem = idx % size_per_batch;
        int i = rem / cols;
        int j = rem % cols;
        out[b * size_per_batch + j * rows + i] = A[idx];
    }
}

void transpose_fwd(const float* A, float* out, int batch, int rows, int cols) {
    int size = batch * rows * cols;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transpose_kernel<<<grid, BLOCK_SIZE>>>(A, out, batch, rows, cols);
}

__global__ void transpose_bwd_kernel(const float* __restrict__ dOut,
                                      float* __restrict__ dA,
                                      int batch, int rows_A, int cols_A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size_per_batch = rows_A * cols_A;
    if (idx < batch * size_per_batch) {
        int b = idx / size_per_batch;
        int rem = idx % size_per_batch;
        int i = rem / cols_A;
        int j = rem % cols_A;
        atomicAdd(&dA[idx], dOut[b * size_per_batch + j * rows_A + i]);
    }
}

void transpose_bwd(const float* dOut, float* dA, int batch, int rows_A, int cols_A) {
    int size = batch * rows_A * cols_A;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transpose_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dA, batch, rows_A, cols_A);
}

// ============================================================================
//                          SLICE COLS
// ============================================================================

__global__ void slice_cols_kernel(const float* __restrict__ A, float* __restrict__ out,
                                  int batch, int rows, int total_cols, int c0, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size_per_batch = rows * width;
    if (idx < batch * size_per_batch) {
        int b = idx / size_per_batch;
        int rem = idx % size_per_batch;
        int i = rem / width;
        int j = rem % width;
        out[idx] = A[b * rows * total_cols + i * total_cols + c0 + j];
    }
}

void slice_cols_fwd(const float* A, float* out, int batch, int rows, int total_cols,
                    int c0, int width) {
    int size = batch * rows * width;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    slice_cols_kernel<<<grid, BLOCK_SIZE>>>(A, out, batch, rows, total_cols, c0, width);
}

__global__ void slice_cols_bwd_kernel(const float* __restrict__ dOut,
                                      float* __restrict__ dA,
                                      int batch, int rows, int total_cols, int c0, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size_per_batch = rows * width;
    if (idx < batch * size_per_batch) {
        int b = idx / size_per_batch;
        int rem = idx % size_per_batch;
        int i = rem / width;
        int j = rem % width;
        atomicAdd(&dA[b * rows * total_cols + i * total_cols + c0 + j], dOut[idx]);
    }
}

void slice_cols_bwd(const float* dOut, float* dA, int batch, int rows, int total_cols,
                    int c0, int width) {
    int size = batch * rows * width;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    slice_cols_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dA, batch, rows, total_cols, c0, width);
}

// ============================================================================
//                        CONCAT COLS (strided copy)
// ============================================================================

__global__ void concat_cols_copy_kernel(const float* __restrict__ part,
                                         float* __restrict__ out,
                                         int batch, int rows, int width, int total_cols, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size_per_batch = rows * width;
    if (idx < batch * size_per_batch) {
        int b = idx / size_per_batch;
        int rem = idx % size_per_batch;
        int i = rem / width;
        int j = rem % width;
        out[b * rows * total_cols + i * total_cols + offset + j] = part[idx];
    }
}

void concat_cols_copy(const float* part, float* out,
                      int batch, int rows, int width, int total_cols, int offset) {
    int size = batch * rows * width;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    concat_cols_copy_kernel<<<grid, BLOCK_SIZE>>>(part, out, batch, rows, width, total_cols, offset);
}

__global__ void concat_cols_bwd_kernel(const float* __restrict__ dOut,
                                        float* __restrict__ dPart,
                                        int batch, int rows, int width, int total_cols, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size_per_batch = rows * width;
    if (idx < batch * size_per_batch) {
        int b = idx / size_per_batch;
        int rem = idx % size_per_batch;
        int i = rem / width;
        int j = rem % width;
        atomicAdd(&dPart[idx], dOut[b * rows * total_cols + i * total_cols + offset + j]);
    }
}

void concat_cols_bwd_part(const float* dOut, float* dPart,
                          int batch, int rows, int width, int total_cols, int offset) {
    int size = batch * rows * width;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    concat_cols_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dPart, batch, rows, width, total_cols, offset);
}

// ============================================================================
//                              CONCAT ROWS
// ============================================================================

__global__ void concat_rows_copy_kernel(const float* __restrict__ src,
                                        float* __restrict__ dst,
                                        int batch, int rows, int cols, int start_row, int dst_total_rows, bool src_batched) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size_per_batch = rows * cols;
    if (idx < batch * size_per_batch) {
        int b = idx / size_per_batch;
        int rem = idx % size_per_batch;
        int i = rem / cols;
        int j = rem % cols;
        int src_idx = src_batched ? idx : rem;
        dst[b * dst_total_rows * cols + (start_row + i) * cols + j] = src[src_idx];
    }
}

void concat_rows_copy(const float* src, float* dst, int batch, int rows, int cols, int start_row, int dst_total_rows, bool src_batched) {
    int size = batch * rows * cols;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    concat_rows_copy_kernel<<<grid, BLOCK_SIZE>>>(src, dst, batch, rows, cols, start_row, dst_total_rows, src_batched);
}

__global__ void concat_rows_bwd_kernel(const float* __restrict__ dOut,
                                       float* __restrict__ dSrc,
                                       int batch, int rows, int cols, int start_row, int dOut_total_rows, bool src_batched) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size_per_batch = rows * cols;
    if (idx < batch * size_per_batch) {
        int b = idx / size_per_batch;
        int rem = idx % size_per_batch;
        int i = rem / cols;
        int j = rem % cols;
        float out_val = dOut[b * dOut_total_rows * cols + (start_row + i) * cols + j];
        if (src_batched) {
            atomicAdd(&dSrc[idx], out_val);
        } else {
            atomicAdd(&dSrc[rem], out_val);
        }
    }
}

void concat_rows_bwd_part(const float* dOut, float* dSrc, int batch, int rows, int cols, int start_row, int dOut_total_rows, bool src_batched) {
    int size = batch * rows * cols;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    concat_rows_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dSrc, batch, rows, cols, start_row, dOut_total_rows, src_batched);
}

// ============================================================================
//                          SELECT ROW
// ============================================================================

__global__ void select_row_kernel(const float* __restrict__ A, float* __restrict__ out,
                                   int batch, int row, int cols, int total_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * cols) {
        int b = idx / cols;
        int j = idx % cols;
        out[idx] = A[b * total_rows * cols + row * cols + j];
    }
}

void select_row_fwd(const float* A, float* out, int batch, int row, int cols, int total_rows) {
    int size = batch * cols;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    select_row_kernel<<<grid, BLOCK_SIZE>>>(A, out, batch, row, cols, total_rows);
}

__global__ void select_row_bwd_kernel(const float* __restrict__ dOut,
                                       float* __restrict__ dA,
                                       int batch, int row, int cols, int total_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * cols) {
        int b = idx / cols;
        int j = idx % cols;
        atomicAdd(&dA[b * total_rows * cols + row * cols + j], dOut[idx]);
    }
}

void select_row_bwd(const float* dOut, float* dA, int batch, int row, int cols, int total_rows) {
    int size = batch * cols;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    select_row_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dA, batch, row, cols, total_rows);
}

// ============================================================================
//                         SOFTMAX (por filas)
// ============================================================================

__global__ void softmax_rows_kernel(const float* __restrict__ A,
                                     float* __restrict__ out,
                                     int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* rowA = A + row * cols;
    float* rowOut = out + row * cols;

    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_max = fmaxf(local_max, rowA[j]);
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = smem[0]; __syncthreads();

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float e = expf(rowA[j] - row_max);
        rowOut[j] = e;
        local_sum += e;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = smem[0];

    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        rowOut[j] /= row_sum;
}

void softmax_rows_fwd(const float* A, float* out, int rows, int cols) {
    int threads = min(cols, 256);
    int t = 1; while (t < threads) t <<= 1;
    threads = min(t, 256);
    softmax_rows_kernel<<<rows, threads, threads * sizeof(float)>>>(A, out, rows, cols);
}

__global__ void softmax_rows_bwd_kernel(const float* __restrict__ dOut,
                                         const float* __restrict__ softmax_out,
                                         float* __restrict__ dA,
                                         int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* dO = dOut + row * cols;
    const float* sO = softmax_out + row * cols;
    float* dArow = dA + row * cols;

    float local_dot = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_dot += dO[j] * sO[j];
    smem[threadIdx.x] = local_dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float dot = smem[0];

    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        atomicAdd(&dArow[j], sO[j] * (dO[j] - dot));
}

void softmax_rows_bwd(const float* dOut, const float* softmax_out,
                      float* dA, int rows, int cols) {
    int threads = min(cols, 256);
    int t = 1; while (t < threads) t <<= 1;
    threads = min(t, 256);
    softmax_rows_bwd_kernel<<<rows, threads, threads * sizeof(float)>>>(dOut, softmax_out, dA, rows, cols);
}

// ============================================================================
//                           LAYER NORM
// ============================================================================

__global__ void layer_norm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ out,
                                   float* __restrict__ mean_out,
                                   float* __restrict__ rstd_out,
                                   float* __restrict__ normed_out,
                                   int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* rowA = A + row * cols;
    float* rowOut = out + row * cols;
    float* rowNorm = normed_out + row * cols;

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_sum += rowA[j];
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smem[0] / cols; __syncthreads();

    float local_var = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float d = rowA[j] - mean;
        local_var += d * d;
    }
    smem[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float var = smem[0] / cols;
    float rstd = rsqrtf(var + eps);

    if (threadIdx.x == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float nh = (rowA[j] - mean) * rstd;
        rowNorm[j] = nh;
        rowOut[j] = nh * gamma[j] + beta[j];
    }
}

void layer_norm_fwd(const float* A, const float* gamma, const float* beta,
                    float* out, float* mean_out, float* rstd_out,
                    float* normed_out, int rows, int cols, float eps) {
    int threads = min(cols, 256);
    int t = 1; while (t < threads) t <<= 1;
    threads = min(t, 256);
    layer_norm_kernel<<<rows, threads, threads * sizeof(float)>>>(
        A, gamma, beta, out, mean_out, rstd_out, normed_out, rows, cols, eps);
}

__global__ void layer_norm_bwd_kernel(const float* __restrict__ dOut,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ normed,
                                       const float* __restrict__ rstd,
                                       float* __restrict__ dA,
                                       float* __restrict__ dgamma,
                                       float* __restrict__ dbeta,
                                       int rows, int cols,
                                       bool A_rg, bool gamma_rg, bool beta_rg) {
    extern __shared__ float smem[];
    float* s_sum_dy = smem;
    float* s_sum_dy_nh = smem + blockDim.x;

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* dO = dOut + row * cols;
    const float* nrow = normed + row * cols;
    float rs = rstd[row];

    float local_dy = 0.0f, local_dy_nh = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float dy = dO[j] * gamma[j];
        local_dy += dy;
        local_dy_nh += dy * nrow[j];
    }
    s_sum_dy[threadIdx.x] = local_dy;
    s_sum_dy_nh[threadIdx.x] = local_dy_nh;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_dy[threadIdx.x] += s_sum_dy[threadIdx.x + s];
            s_sum_dy_nh[threadIdx.x] += s_sum_dy_nh[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_dy = s_sum_dy[0];
    float sum_dy_nh = s_sum_dy_nh[0];

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float doj = dO[j];
        if (A_rg) {
            float dy = doj * gamma[j];
            float dx = rs * (dy - sum_dy / cols - nrow[j] * sum_dy_nh / cols);
            atomicAdd(&dA[row * cols + j], dx);
        }
        if (gamma_rg) atomicAdd(&dgamma[j], doj * nrow[j]);
        if (beta_rg)  atomicAdd(&dbeta[j], doj);
    }
}

void layer_norm_bwd(const float* dOut, const float* gamma,
                    const float* normed, const float* rstd,
                    float* dA, float* dgamma, float* dbeta,
                    int rows, int cols,
                    bool A_rg, bool gamma_rg, bool beta_rg) {
    int threads = min(cols, 256);
    int t = 1; while (t < threads) t <<= 1;
    threads = min(t, 256);
    layer_norm_bwd_kernel<<<rows, threads, 2 * threads * sizeof(float)>>>(
        dOut, gamma, normed, rstd, dA, dgamma, dbeta, rows, cols, A_rg, gamma_rg, beta_rg);
}

// ============================================================================
//                              GELU
// ============================================================================

__global__ void gelu_kernel(const float* __restrict__ A,
                            float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = A[idx];
        const float k0 = 0.7978845608f;
        const float k1 = 0.044715f;
        float x3 = x * x * x;
        float t = tanhf(k0 * (x + k1 * x3));
        out[idx] = 0.5f * x * (1.0f + t);
    }
}

void gelu_fwd(const float* A, float* out, int size) {
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_kernel<<<grid, BLOCK_SIZE>>>(A, out, size);
}

__global__ void gelu_bwd_kernel(const float* __restrict__ dOut,
                                const float* __restrict__ A_data,
                                float* __restrict__ dA, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = A_data[idx];
        const float k0 = 0.7978845608f;
        const float k1 = 0.044715f;
        float x2 = x * x;
        float x3 = x2 * x;
        float inner = k0 * (x + k1 * x3);
        float t = tanhf(inner);
        float sech2 = 1.0f - t * t;
        float dinner = k0 * (1.0f + 3.0f * k1 * x2);
        float dgelu = 0.5f * (1.0f + t) + 0.5f * x * sech2 * dinner;
        atomicAdd(&dA[idx], dOut[idx] * dgelu);
    }
}

void gelu_bwd(const float* dOut, const float* A_data, float* dA, int size) {
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, A_data, dA, size);
}

// ============================================================================
//                    SOFTMAX + CROSS ENTROPY
// ============================================================================

__global__ void softmax_ce_kernel(const float* __restrict__ logits,
                                   const int* __restrict__ labels, int num_classes,
                                   float* __restrict__ loss_out,
                                   float* __restrict__ probs_out) {
    extern __shared__ float smem[];
    int b = blockIdx.x;

    const float* rowLogits = logits + b * num_classes;
    float* rowProbs = probs_out + b * num_classes;
    int label = labels[b];

    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x)
        local_max = fmaxf(local_max, rowLogits[j]);
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float mx = smem[0]; __syncthreads();

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        float e = expf(rowLogits[j] - mx);
        rowProbs[j] = e;
        local_sum += e;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float total = smem[0];

    for (int j = threadIdx.x; j < num_classes; j += blockDim.x)
        rowProbs[j] /= total;

    if (threadIdx.x == 0) {
        float loss = -logf(fmaxf(rowProbs[label], 1e-9f));
        atomicAdd(&loss_out[0], loss / gridDim.x); // Average over batch
    }
}

void softmax_cross_entropy_fwd(const float* logits, const int* labels, int batch, int num_classes,
                               float* loss_out, float* probs_out) {
    int threads = 32;
    softmax_ce_kernel<<<batch, threads, threads * sizeof(float)>>>(
        logits, labels, num_classes, loss_out, probs_out);
}

__global__ void softmax_ce_bwd_kernel(float upstream, const float* __restrict__ probs,
                                       const int* __restrict__ labels, int batch, int num_classes,
                                       float* __restrict__ dLogits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * num_classes) {
        int b = idx / num_classes;
        int j = idx % num_classes;
        int label = labels[b];
        float target = (j == label) ? 1.0f : 0.0f;
        // upstream is already divided by batch if loss is mean over batch, wait no.
        // If forward computes mean over batch, then backward grad should be upstream / batch.
        // Wait, loss = sum(L_i) / B. dL/dL_i = 1/B. So dL_i/dx = (p - y). Total = 1/B * (p - y).
        atomicAdd(&dLogits[idx], (upstream / batch) * (probs[idx] - target));
    }
}

void softmax_cross_entropy_bwd(float upstream, const float* probs, const int* labels,
                               int batch, int num_classes, float* dLogits) {
    int grid = (batch * num_classes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_ce_bwd_kernel<<<grid, BLOCK_SIZE>>>(upstream, probs, labels, batch, num_classes, dLogits);
}

// ============================================================================
//                        ADAM OPTIMIZER
// ============================================================================

__global__ void adam_kernel(float* __restrict__ param,
                           float* __restrict__ grad,
                           float* __restrict__ m_buf,
                           float* __restrict__ v_buf,
                           float lr, float beta1, float beta2, float eps,
                           float bc1, float bc2, float grad_scale,
                           float weight_decay, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] * grad_scale;
        float mi = beta1 * m_buf[idx] + (1.0f - beta1) * g;
        float vi = beta2 * v_buf[idx] + (1.0f - beta2) * g * g;
        m_buf[idx] = mi;
        v_buf[idx] = vi;
        float mhat = mi / bc1;
        float vhat = vi / bc2;
        param[idx] -= lr * mhat / (sqrtf(vhat) + eps);
        if (weight_decay > 0.0f)
            param[idx] -= lr * weight_decay * param[idx];
    }
}

void adam_step(float* param, float* grad, float* m, float* v,
              float lr, float beta1, float beta2, float eps,
              float bc1, float bc2, float grad_scale, float weight_decay,
              int size) {
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adam_kernel<<<grid, BLOCK_SIZE>>>(param, grad, m, v, lr, beta1, beta2, eps, bc1, bc2, grad_scale, weight_decay, size);
}

// ============================================================================
//                          UTILIDADES
// ============================================================================

void zero_memory(float* ptr, int size) {
    CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(float)));
}

}} // namespace vit::cuda

#endif // USE_CUDA

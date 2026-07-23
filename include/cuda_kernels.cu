// cuda_kernels.cu — Implementación de todos los kernels CUDA para el ViT.
//
// Cada kernel es una función __global__ que se ejecuta en la GPU.
// Las funciones wrapper (declaradas en cuda_kernels.cuh) calculan la
// configuración de bloques/hilos y lanzan el kernel.
//
// Convenciones:
//   - Matrices en row-major: A[i*cols + j]
//   - Los kernels backward ACUMULAN gradientes (+=), no los sobreescriben.
//   - El matmul usa tiling con shared memory (TILE_SIZE x TILE_SIZE).

#ifdef USE_CUDA

#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

// Macro para verificar errores de CUDA (solo en debug).
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

// Kernel de matmul con tiling en shared memory.
// out(m,n) = A(m,k) * B(k,n)
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ out,
                              int m, int k, int n) {
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

void matmul_fwd(const float* A, const float* B, float* out, int m, int k, int n) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<grid, block>>>(A, B, out, m, k, n);
}

// Backward respecto a A: dA += dOut * B^T
// dA(m,k) += dOut(m,n) * B^T(n,k)  =>  matmul de dOut(m,n) * B^T(n,k)
// B^T[j,p] = B[p,j], así que en vez de transponer explícitamente,
// accedemos B[p*n + j] como B^T[j*k + p] => B[p, j].
__global__ void matmul_bwd_A_kernel(const float* __restrict__ dOut,
                                     const float* __restrict__ B,
                                     float* __restrict__ dA,
                                     int m, int k, int n) {
    __shared__ float sDO[TILE_SIZE][TILE_SIZE];
    __shared__ float sBT[TILE_SIZE][TILE_SIZE]; // B transpuesta

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int doCol = t * TILE_SIZE + threadIdx.x;
        int btRow = t * TILE_SIZE + threadIdx.y;
        sDO[threadIdx.y][threadIdx.x] = (row < m && doCol < n) ? dOut[row * n + doCol] : 0.0f;
        // B^T[btRow, col] = B[col, btRow]  (col < k, btRow < n)
        sBT[threadIdx.y][threadIdx.x] = (btRow < n && col < k) ? B[col * n + btRow] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += sDO[threadIdx.y][i] * sBT[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < k)
        atomicAdd(&dA[row * k + col], sum);
}

void matmul_bwd_A(const float* dOut, const float* B, float* dA, int m, int k, int n) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((k + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    matmul_bwd_A_kernel<<<grid, block>>>(dOut, B, dA, m, k, n);
}

// Backward respecto a B: dB += A^T * dOut
// dB(k,n) += A^T(k,m) * dOut(m,n)
__global__ void matmul_bwd_B_kernel(const float* __restrict__ A,
                                     const float* __restrict__ dOut,
                                     float* __restrict__ dB,
                                     int m, int k, int n) {
    __shared__ float sAT[TILE_SIZE][TILE_SIZE]; // A transpuesta
    __shared__ float sDO[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // fila de dB (0..k-1)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // col de dB (0..n-1)

    float sum = 0.0f;
    for (int t = 0; t < (m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int atCol = t * TILE_SIZE + threadIdx.x;
        int doRow = t * TILE_SIZE + threadIdx.y;
        // A^T[row, atCol] = A[atCol, row]  (row < k, atCol < m)
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

void matmul_bwd_B(const float* A, const float* dOut, float* dB, int m, int k, int n) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (k + TILE_SIZE - 1) / TILE_SIZE);
    matmul_bwd_B_kernel<<<grid, block>>>(A, dOut, dB, m, k, n);
}

// ============================================================================
//                         ELEMENTWISE ADD
// ============================================================================

__global__ void add_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = A[idx] + B[idx];
}

void add_fwd(const float* A, const float* B, float* out, int size) {
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<<<grid, BLOCK_SIZE>>>(A, B, out, size);
}

__global__ void add_bwd_kernel(const float* __restrict__ dOut, float* dA, float* dB,
                               int size, bool doA, bool doB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = dOut[idx];
        if (doA) atomicAdd(&dA[idx], g);
        if (doB) atomicAdd(&dB[idx], g);
    }
}

void add_bwd(const float* dOut, float* dA, float* dB, int size,
             bool A_rg, bool B_rg) {
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dA, dB, size, A_rg, B_rg);
}

// ============================================================================
//                       ADD ROW BROADCAST (bias)
// ============================================================================

__global__ void add_row_broadcast_kernel(const float* __restrict__ A,
                                         const float* __restrict__ bias,
                                         float* __restrict__ out,
                                         int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int j = idx % cols;
        out[idx] = A[idx] + bias[j];
    }
}

void add_row_broadcast_fwd(const float* A, const float* bias, float* out,
                           int rows, int cols) {
    int size = rows * cols;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_row_broadcast_kernel<<<grid, BLOCK_SIZE>>>(A, bias, out, rows, cols);
}

__global__ void add_row_broadcast_bwd_bias_kernel(const float* __restrict__ dOut,
                                                   float* __restrict__ dbias,
                                                   int rows, int cols) {
    // Cada hilo suma una columna.
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i)
            sum += dOut[i * cols + j];
        atomicAdd(&dbias[j], sum);
    }
}

void add_row_broadcast_bwd(const float* dOut, float* dA, float* dbias,
                           int rows, int cols,
                           bool A_rg, bool bias_rg) {
    int size = rows * cols;
    if (A_rg) {
        // dA += dOut (elemento a elemento)
        int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dA, nullptr, size, true, false);
    }
    if (bias_rg) {
        int grid = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_row_broadcast_bwd_bias_kernel<<<grid, BLOCK_SIZE>>>(dOut, dbias, rows, cols);
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
                                 int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int i = idx / cols;
        int j = idx % cols;
        out[j * rows + i] = A[i * cols + j];
    }
}

void transpose_fwd(const float* A, float* out, int rows, int cols) {
    int size = rows * cols;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transpose_kernel<<<grid, BLOCK_SIZE>>>(A, out, rows, cols);
}

__global__ void transpose_bwd_kernel(const float* __restrict__ dOut,
                                      float* __restrict__ dA,
                                      int rows_A, int cols_A) {
    // dOut tiene forma (cols_A, rows_A). dA tiene forma (rows_A, cols_A).
    // dA[i, j] += dOut[j, i]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows_A * cols_A) {
        int i = idx / cols_A;
        int j = idx % cols_A;
        atomicAdd(&dA[i * cols_A + j], dOut[j * rows_A + i]);
    }
}

void transpose_bwd(const float* dOut, float* dA, int rows_A, int cols_A) {
    int size = rows_A * cols_A;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transpose_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dA, rows_A, cols_A);
}

// ============================================================================
//                          SLICE COLS
// ============================================================================

__global__ void slice_cols_kernel(const float* __restrict__ A, float* __restrict__ out,
                                  int rows, int total_cols, int c0, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * width) {
        int i = idx / width;
        int j = idx % width;
        out[idx] = A[i * total_cols + c0 + j];
    }
}

void slice_cols_fwd(const float* A, float* out, int rows, int total_cols,
                    int c0, int width) {
    int size = rows * width;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    slice_cols_kernel<<<grid, BLOCK_SIZE>>>(A, out, rows, total_cols, c0, width);
}

__global__ void slice_cols_bwd_kernel(const float* __restrict__ dOut,
                                      float* __restrict__ dA,
                                      int rows, int total_cols, int c0, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * width) {
        int i = idx / width;
        int j = idx % width;
        atomicAdd(&dA[i * total_cols + c0 + j], dOut[idx]);
    }
}

void slice_cols_bwd(const float* dOut, float* dA, int rows, int total_cols,
                    int c0, int width) {
    int size = rows * width;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    slice_cols_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dA, rows, total_cols, c0, width);
}

// ============================================================================
//                        CONCAT COLS (strided copy)
// ============================================================================

// Copia part(rows, width) -> out(rows, total_cols) en columnas [offset, offset+width)
__global__ void concat_cols_copy_kernel(const float* __restrict__ part,
                                         float* __restrict__ out,
                                         int rows, int width, int total_cols, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * width) {
        int i = idx / width;
        int j = idx % width;
        out[i * total_cols + offset + j] = part[i * width + j];
    }
}

// Backward: acumula dOut(rows, total_cols)[:, offset:offset+width] en dPart(rows, width)
__global__ void concat_cols_bwd_kernel(const float* __restrict__ dOut,
                                        float* __restrict__ dPart,
                                        int rows, int width, int total_cols, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * width) {
        int i = idx / width;
        int j = idx % width;
        atomicAdd(&dPart[i * width + j], dOut[i * total_cols + offset + j]);
    }
}

// Wrappers host
void concat_cols_copy(const float* part, float* out,
                      int rows, int width, int total_cols, int offset) {
    int size = rows * width;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    concat_cols_copy_kernel<<<grid, BLOCK_SIZE>>>(part, out, rows, width, total_cols, offset);
}

void concat_cols_bwd_part(const float* dOut, float* dPart,
                          int rows, int width, int total_cols, int offset) {
    int size = rows * width;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    concat_cols_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dPart, rows, width, total_cols, offset);
}

// ============================================================================
//                          SELECT ROW
// ============================================================================

__global__ void select_row_kernel(const float* __restrict__ A, float* __restrict__ out,
                                   int row, int cols, int total_cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) out[j] = A[row * total_cols + j];
}

void select_row_fwd(const float* A, float* out, int row, int cols, int total_cols) {
    int grid = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    select_row_kernel<<<grid, BLOCK_SIZE>>>(A, out, row, cols, total_cols);
}

__global__ void select_row_bwd_kernel(const float* __restrict__ dOut,
                                       float* __restrict__ dA,
                                       int row, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) atomicAdd(&dA[row * cols + j], dOut[j]);
}

void select_row_bwd(const float* dOut, float* dA, int row, int cols) {
    int grid = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    select_row_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dA, row, cols);
}

// ============================================================================
//                         SOFTMAX (por filas)
// ============================================================================

// Un bloque por fila. Cada hilo maneja algunos elementos de la fila.
// Usamos shared memory para las reducciones (max y sum).
__global__ void softmax_rows_kernel(const float* __restrict__ A,
                                     float* __restrict__ out,
                                     int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* rowA = A + row * cols;
    float* rowOut = out + row * cols;

    // Paso 1: encontrar el máximo de la fila (reducción)
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

    // Paso 2: exp(x - max) y sumar
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

    // Paso 3: normalizar
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        rowOut[j] /= row_sum;
}

void softmax_rows_fwd(const float* A, float* out, int rows, int cols) {
    int threads = min(cols, 256);
    // Redondear threads a potencia de 2 para la reducción
    int t = 1; while (t < threads) t <<= 1;
    threads = min(t, 256);
    softmax_rows_kernel<<<rows, threads, threads * sizeof(float)>>>(A, out, rows, cols);
}

// Backward: dA[i,j] += s[i,j] * (dOut[i,j] - dot(dOut[i,:], s[i,:]))
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

    // dot = sum_j dOut[j] * softmax[j]
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

// Un bloque por fila. Calcula mean, variance, normaliza, aplica gamma/beta.
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

    // Media
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

    // Varianza
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

    // Normalizar y aplicar gamma/beta
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

// Backward de LayerNorm
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
    // smem: [0..blockDim-1] para sum_dy, [blockDim..2*blockDim-1] para sum_dy_nh
    float* s_sum_dy = smem;
    float* s_sum_dy_nh = smem + blockDim.x;

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* dO = dOut + row * cols;
    const float* nrow = normed + row * cols;
    float rs = rstd[row];

    // Calcular sum_dy y sum_dy_nh para esta fila
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
//                              CONCAT ROWS
// ============================================================================
__global__ void concat_rows_copy_kernel(const float* __restrict__ src,
                                        float* __restrict__ dst,
                                        int rows, int cols, int start_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        dst[start_row * cols + idx] = src[idx];
    }
}

__global__ void concat_rows_bwd_kernel(const float* __restrict__ dOut,
                                       float* __restrict__ dSrc,
                                       int rows, int cols, int start_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        atomicAdd(&dSrc[idx], dOut[start_row * cols + idx]);
    }
}

void concat_rows_copy(const float* src, float* dst, int rows, int cols, int start_row) {
    int size = rows * cols;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    concat_rows_copy_kernel<<<grid, BLOCK_SIZE>>>(src, dst, rows, cols, start_row);
}

void concat_rows_bwd_part(const float* dOut, float* dSrc, int rows, int cols, int start_row) {
    int size = rows * cols;
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    concat_rows_bwd_kernel<<<grid, BLOCK_SIZE>>>(dOut, dSrc, rows, cols, start_row);
}

// ============================================================================
//                              GELU
// ============================================================================

__global__ void gelu_kernel(const float* __restrict__ A,
                            float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = A[idx];
        const float k0 = 0.7978845608f; // sqrt(2/pi)
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

// Se ejecuta con un solo bloque porque logits es (1, num_classes) con num_classes=10.
__global__ void softmax_ce_kernel(const float* __restrict__ logits,
                                   int label, int num_classes,
                                   float* __restrict__ loss_out,
                                   float* __restrict__ probs_out) {
    extern __shared__ float smem[];

    // Encontrar max
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x)
        local_max = fmaxf(local_max, logits[j]);
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float mx = smem[0]; __syncthreads();

    // exp y sum
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
        float e = expf(logits[j] - mx);
        probs_out[j] = e;
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
        probs_out[j] /= total;

    if (threadIdx.x == 0)
        loss_out[0] = -logf(fmaxf(probs_out[label], 1e-9f));
}

void softmax_cross_entropy_fwd(const float* logits, int label, int num_classes,
                               float* loss_out, float* probs_out) {
    int threads = 32; // num_classes es pequeño (10)
    softmax_ce_kernel<<<1, threads, threads * sizeof(float)>>>(
        logits, label, num_classes, loss_out, probs_out);
}

__global__ void softmax_ce_bwd_kernel(float upstream, const float* __restrict__ probs,
                                       int label, int num_classes,
                                       float* __restrict__ dLogits) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < num_classes) {
        float target = (j == label) ? 1.0f : 0.0f;
        atomicAdd(&dLogits[j], upstream * (probs[j] - target));
    }
}

void softmax_cross_entropy_bwd(float upstream, const float* probs, int label,
                               int num_classes, float* dLogits) {
    int grid = (num_classes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_ce_bwd_kernel<<<grid, BLOCK_SIZE>>>(upstream, probs, label, num_classes, dLogits);
}

// ============================================================================
//                        ADAM OPTIMIZER
// ============================================================================

__global__ void adam_kernel(float* __restrict__ param,
                           float* __restrict__ grad,
                           float* __restrict__ m_buf,
                           float* __restrict__ v_buf,
                           float lr, float beta1, float beta2, float eps,
                           float bc1, float bc2, float grad_scale, int size) {
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
    }
}

void adam_step(float* param, float* grad, float* m, float* v,
              float lr, float beta1, float beta2, float eps,
              float bc1, float bc2, float grad_scale, int size) {
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adam_kernel<<<grid, BLOCK_SIZE>>>(param, grad, m, v, lr, beta1, beta2, eps, bc1, bc2, grad_scale, size);
}

// ============================================================================
//                          UTILIDADES
// ============================================================================

void zero_memory(float* ptr, int size) {
    CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(float)));
}

}} // namespace vit::cuda

#endif // USE_CUDA

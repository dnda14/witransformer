// tensor.hpp
//
// Motor mínimo de álgebra lineal + autograd (diferenciación automática en modo
// inverso), escrito desde cero sin dependencias externas de ML.
//
// Idea central: cada Tensor es una matriz 2D (rows x cols) que, si participa
// en operaciones, va construyendo un grafo computacional. Cada nodo guarda
// una función `backward_fn` que sabe cómo propagar el gradiente hacia sus
// "padres". Tensor::backward() hace un orden topológico del grafo y llama a
// cada backward_fn en orden inverso (de la salida hacia las entradas), que es
// exactamente la regla de la cadena aplicada nodo por nodo.
//
// Todo esto reemplaza a un framework como libtorch: aquí NO hay autograd
// mágico, cada operación (matmul, softmax, layernorm, gelu, etc.) implementa
// su propia derivada a mano en este archivo.
//
// Cuando USE_CUDA está definido, cada tensor puede almacenar datos tanto en
// CPU (std::vector) como en GPU (punteros device). Las funciones to_device()
// y to_host() transfieren datos entre ambos. Las operaciones en ops.hpp
// eligen automáticamente el path CPU o GPU.

#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <map>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace vit {

#ifdef USE_CUDA
// Caching allocator súper simple para evitar fragmentación de memoria y CUDA overhead
struct CachingAllocator {
    inline static std::map<size_t, std::vector<float*>> pool;
    static float* allocate(size_t bytes) {
        if (!pool[bytes].empty()) {
            float* ptr = pool[bytes].back();
            pool[bytes].pop_back();
            return ptr;
        }
        float* ptr = nullptr;
        cudaMalloc(&ptr, bytes);
        return ptr;
    }
    static void free(float* ptr, size_t bytes) {
        if (ptr) pool[bytes].push_back(ptr);
    }
};
#endif

struct TensorImpl;
using Tensor = std::shared_ptr<TensorImpl>;

struct TensorImpl : std::enable_shared_from_this<TensorImpl> {
    int rows = 0, cols = 0;
    std::vector<float> data;   // tamaño rows*cols, almacenamiento row-major
    std::vector<float> grad;   // mismo tamaño que data, acumulador de gradiente
    bool requires_grad = false;

    // Grafo computacional: de qué tensores viene este y cómo propagar el
    // gradiente hacia ellos.
    std::vector<Tensor> parents;
    std::function<void()> backward_fn; // usa 'this->grad' y escribe en parents[i]->grad

#ifdef USE_CUDA
    float* d_data = nullptr;   // puntero a VRAM (device memory)
    float* d_grad = nullptr;   // gradientes en VRAM
    bool on_device = false;    // true si los datos están en GPU

    // Copia data y grad de CPU a GPU. Si los buffers device no existen, los
    // aloja con CachingAllocator.
    void to_device() {
        size_t bytes = data.size() * sizeof(float);
        if (!d_data) d_data = CachingAllocator::allocate(bytes);
        if (!d_grad) d_grad = CachingAllocator::allocate(bytes);
        cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad, grad.data(), bytes, cudaMemcpyHostToDevice);
        on_device = true;
    }

    // Copia data y grad de GPU a CPU.
    void to_host() {
        if (!d_data) return;
        size_t bytes = data.size() * sizeof(float);
        cudaMemcpy(data.data(), d_data, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad.data(), d_grad, bytes, cudaMemcpyDeviceToHost);
    }

    // Aloja buffers en GPU sin copiar datos de CPU.
    void alloc_device() {
        size_t bytes = data.size() * sizeof(float);
        if (!d_data) d_data = CachingAllocator::allocate(bytes);
        if (!d_grad) d_grad = CachingAllocator::allocate(bytes);
        cudaMemset(d_data, 0, bytes);
        cudaMemset(d_grad, 0, bytes);
        on_device = true;
    }

    // Pone a cero los gradientes en GPU.
    void zero_grad_device() {
        if (d_grad) cudaMemset(d_grad, 0, data.size() * sizeof(float));
    }

    void free_device() {
        size_t bytes = data.size() * sizeof(float);
        if (d_data) { CachingAllocator::free(d_data, bytes); d_data = nullptr; }
        if (d_grad) { CachingAllocator::free(d_grad, bytes); d_grad = nullptr; }
        on_device = false;
    }
#endif

    TensorImpl(int r, int c, bool rg = false)
        : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0f),
          grad(static_cast<size_t>(r) * c, 0.0f), requires_grad(rg) {}

#ifdef USE_CUDA
    ~TensorImpl() { free_device(); }
#endif

    inline float& at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    inline float at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
    inline float& g(int r, int c) { return grad[static_cast<size_t>(r) * cols + c]; }

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
#ifdef USE_CUDA
        zero_grad_device();
#endif
    }
};

// ---------- Construcción básica ----------

inline Tensor make_tensor(int rows, int cols, bool requires_grad = false) {
    auto t = std::make_shared<TensorImpl>(rows, cols, requires_grad);
#ifdef USE_CUDA
    t->alloc_device();
#endif
    return t;
}

inline Tensor from_vector(int rows, int cols, const std::vector<float>& values,
                           bool requires_grad = false) {
    auto t = std::make_shared<TensorImpl>(rows, cols, requires_grad);
    if (values.size() != t->data.size())
        throw std::runtime_error("from_vector: tamaño no coincide con rows*cols");
    t->data = values;
#ifdef USE_CUDA
    t->to_device();
#endif
    return t;
}

// Inicialización Xavier/Glorot uniforme, estándar para capas lineales.
inline Tensor random_tensor(int rows, int cols, bool requires_grad, std::mt19937& rng) {
    auto t = std::make_shared<TensorImpl>(rows, cols, requires_grad);
    float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& v : t->data) v = dist(rng);
#ifdef USE_CUDA
    t->to_device();
#endif
    return t;
}

inline Tensor zeros(int rows, int cols, bool requires_grad = false) {
    auto t = std::make_shared<TensorImpl>(rows, cols, requires_grad);
#ifdef USE_CUDA
    t->alloc_device();
#endif
    return t;
}

// ---------- Backward: orden topológico + regla de la cadena ----------

inline void build_topo(const Tensor& t, std::vector<Tensor>& order,
                        std::vector<TensorImpl*>& visited) {
    if (std::find(visited.begin(), visited.end(), t.get()) != visited.end()) return;
    visited.push_back(t.get());
    for (auto& p : t->parents) build_topo(p, order, visited);
    order.push_back(t);
}

// Llamar solo sobre un tensor escalar (1x1), típicamente la pérdida.
inline void backward(const Tensor& loss) {
    if (loss->rows != 1 || loss->cols != 1)
        throw std::runtime_error("backward() debe llamarse sobre un escalar 1x1");
    std::vector<Tensor> order;
    std::vector<TensorImpl*> visited;
    build_topo(loss, order, visited);
    loss->grad[0] = 1.0f; // d(loss)/d(loss) = 1
#ifdef USE_CUDA
    if (loss->d_grad) {
        float one = 1.0f;
        cudaMemcpy(loss->d_grad, &one, sizeof(float), cudaMemcpyHostToDevice);
    }
#endif
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if ((*it)->backward_fn) (*it)->backward_fn();
    }
}

} // namespace vit

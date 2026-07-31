/**
 * @file tensor.hpp
 * @brief Motor mínimo de álgebra lineal + autograd (diferenciación automática en modo inverso).
 *
 * Escrito desde cero sin dependencias externas de ML. Reemplaza la funcionalidad
 * de un framework como libtorch: aquí NO hay autograd mágico, cada operación
 * (matmul, softmax, layernorm, gelu, etc.) implementa su propia derivada a mano
 * en ops.hpp.
 *
 * @details
 * **Idea central:** cada Tensor es una matriz 2D (rows × cols) que, si participa
 * en operaciones, va construyendo un **grafo computacional**. Cada nodo guarda
 * una función `backward_fn` que sabe cómo propagar el gradiente hacia sus
 * "padres". La función backward() hace un orden topológico del grafo y llama a
 * cada backward_fn en orden inverso (de la salida hacia las entradas), que es
 * exactamente la **regla de la cadena** aplicada nodo por nodo.
 *
 * **Soporte CUDA:** Cuando `USE_CUDA` está definido, cada tensor puede almacenar
 * datos tanto en CPU (`std::vector`) como en GPU (punteros device). Las funciones
 * `to_device()` y `to_host()` transfieren datos entre ambos. Las operaciones en
 * ops.hpp eligen automáticamente el path CPU o GPU.
 */

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
/**
 * @brief Pool de memoria GPU simple para evitar fragmentación y overhead de cudaMalloc/cudaFree.
 *
 * En lugar de llamar a cudaMalloc/cudaFree repetidamente (que son lentos),
 * reutiliza buffers previamente liberados agrupados por tamaño en bytes.
 * Esto reduce significativamente el overhead durante el entrenamiento.
 */
struct CachingAllocator {
    inline static std::map<size_t, std::vector<float*>> pool; ///< Pool de buffers libres, indexados por tamaño en bytes.

    /**
     * @brief Obtiene un buffer de GPU del pool, o aloja uno nuevo si no hay disponible.
     * @param bytes Tamaño del buffer en bytes.
     * @return Puntero al buffer en memoria de GPU (device memory).
     */
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

    /**
     * @brief Devuelve un buffer al pool para su reutilización (no lo libera realmente).
     * @param ptr   Puntero al buffer en GPU.
     * @param bytes Tamaño del buffer en bytes (debe coincidir con el usado en allocate).
     */
    static void free(float* ptr, size_t bytes) {
        if (ptr) pool[bytes].push_back(ptr);
    }
};
#endif

struct TensorImpl;

/// Alias: un Tensor es un puntero compartido (shared_ptr) a TensorImpl.
using Tensor = std::shared_ptr<TensorImpl>;

/**
 * @brief Estructura principal del proyecto. Representa un tensor 2D (matriz).
 *
 * Almacena los datos numéricos, los gradientes acumulados y la información
 * necesaria para recorrer el grafo computacional durante backpropagation.
 * Hereda de `enable_shared_from_this` para poder obtener un shared_ptr a sí mismo.
 */
struct TensorImpl : std::enable_shared_from_this<TensorImpl> {
    int batch = 1;             ///< Tamaño del batch (por defecto 1 para compatibilidad).
    int rows = 0;              ///< Número de filas de la matriz.
    int cols = 0;              ///< Número de columnas de la matriz.
    std::vector<float> data;   ///< Datos de la matriz, tamaño batch×rows×cols, almacenamiento row-major.
    std::vector<float> grad;   ///< Gradientes acumulados, mismo tamaño que data.
    bool requires_grad = false; ///< Si es true, este tensor participa en backpropagation.

    /// @name Grafo computacional
    /// @{
    std::vector<Tensor> parents;        ///< Tensores de los que proviene este (entradas de la operación).
    std::function<void()> backward_fn;  ///< Función que propaga gradientes de `this->grad` hacia `parents[i]->grad`.
    /// @}

#ifdef USE_CUDA
    /// @name Almacenamiento en GPU (solo cuando USE_CUDA está definido)
    /// @{
    float* d_data = nullptr;   ///< Puntero a los datos en VRAM (device memory).
    float* d_grad = nullptr;   ///< Puntero a los gradientes en VRAM.
    bool on_device = false;    ///< True si los datos están actualmente en GPU.
    /// @}

    /**
     * @brief Copia data y grad de CPU a GPU.
     *
     * Si los buffers en GPU no existen, los aloja usando CachingAllocator.
     * Después de esta llamada, on_device será true.
     */
    void to_device() {
        size_t bytes = data.size() * sizeof(float);
        if (!d_data) d_data = CachingAllocator::allocate(bytes);
        if (!d_grad) d_grad = CachingAllocator::allocate(bytes);
        cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad, grad.data(), bytes, cudaMemcpyHostToDevice);
        on_device = true;
    }

    /**
     * @brief Copia data y grad de GPU a CPU.
     *
     * Los datos en GPU no se liberan; solo se copian a los vectores de CPU.
     */
    void to_host() {
        if (!d_data) return;
        size_t bytes = data.size() * sizeof(float);
        cudaMemcpy(data.data(), d_data, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad.data(), d_grad, bytes, cudaMemcpyDeviceToHost);
    }

    /**
     * @brief Aloja buffers en GPU inicializados a cero, sin copiar datos de CPU.
     *
     * Útil para tensores de salida que serán escritos directamente por kernels CUDA.
     */
    void alloc_device() {
        size_t bytes = data.size() * sizeof(float);
        if (!d_data) d_data = CachingAllocator::allocate(bytes);
        if (!d_grad) d_grad = CachingAllocator::allocate(bytes);
        cudaMemset(d_data, 0, bytes);
        cudaMemset(d_grad, 0, bytes);
        on_device = true;
    }

    /** @brief Pone a cero los gradientes en GPU (d_grad). */
    void zero_grad_device() {
        if (d_grad) cudaMemset(d_grad, 0, data.size() * sizeof(float));
    }

    /** @brief Devuelve los buffers de GPU al CachingAllocator y marca on_device = false. */
    void free_device() {
        size_t bytes = data.size() * sizeof(float);
        if (d_data) { CachingAllocator::free(d_data, bytes); d_data = nullptr; }
        if (d_grad) { CachingAllocator::free(d_grad, bytes); d_grad = nullptr; }
        on_device = false;
    }
#endif

    /**
     * @brief Construye un tensor de dimensiones b × r × c, inicializado a ceros.
     * @param b  Tamaño del batch.
     * @param r  Número de filas.
     * @param c  Número de columnas.
     * @param rg Si es true, el tensor participará en backpropagation.
     */
    TensorImpl(int b, int r, int c, bool rg = false)
        : batch(b), rows(r), cols(c), data(static_cast<size_t>(b) * r * c, 0.0f),
          grad(static_cast<size_t>(b) * r * c, 0.0f), requires_grad(rg) {}

#ifdef USE_CUDA
    /** @brief Destructor: devuelve los buffers de GPU al pool. */
    ~TensorImpl() { free_device(); }
#endif

    /** @brief Acceso mutable al elemento en la fila r, columna c (para batch=1). */
    inline float& at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }

    /** @brief Acceso de solo lectura al elemento en la fila r, columna c (para batch=1). */
    inline float at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }

    /** @brief Acceso mutable al gradiente en la fila r, columna c (para batch=1). */
    inline float& g(int r, int c) { return grad[static_cast<size_t>(r) * cols + c]; }

    /** @brief Acceso mutable al elemento en (b, r, c). */
    inline float& at(int b, int r, int c) { return data[(static_cast<size_t>(b) * rows + r) * cols + c]; }

    /** @brief Acceso de solo lectura al elemento en (b, r, c). */
    inline float at(int b, int r, int c) const { return data[(static_cast<size_t>(b) * rows + r) * cols + c]; }

    /** @brief Acceso mutable al gradiente en (b, r, c). */
    inline float& g(int b, int r, int c) { return grad[(static_cast<size_t>(b) * rows + r) * cols + c]; }

    /**
     * @brief Pone a cero todos los gradientes (CPU y GPU si aplica).
     *
     * Se llama antes de cada paso de optimización para evitar acumular
     * gradientes de iteraciones anteriores.
     */
    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
#ifdef USE_CUDA
        zero_grad_device();
#endif
    }
};

// ======================= Construcción de Tensores =======================

/**
     * @brief Crea un tensor vacío (lleno de ceros) en 3D.
     */
inline Tensor make_tensor(int batch, int rows, int cols, bool requires_grad = false) {
    auto t = std::make_shared<TensorImpl>(batch, rows, cols, requires_grad);
#ifdef USE_CUDA
    t->alloc_device();
#endif
    return t;
}

/**
 * @brief Sobrecarga para crear un tensor 2D (batch = 1).
 */
inline Tensor make_tensor(int rows, int cols, bool requires_grad = false) {
    return make_tensor(1, rows, cols, requires_grad);
}

/**
 * @brief Crea un tensor 3D a partir de un vector.
 */
inline Tensor from_vector(int batch, int rows, int cols, const std::vector<float>& values,
                           bool requires_grad = false) {
    auto t = std::make_shared<TensorImpl>(batch, rows, cols, requires_grad);
    if (values.size() != t->data.size())
        throw std::runtime_error("from_vector: tamaño no coincide con batch*rows*cols");
    t->data = values;
#ifdef USE_CUDA
    t->to_device();
#endif
    return t;
}

/**
 * @brief Sobrecarga para crear un tensor 2D desde vector (batch = 1).
 */
inline Tensor from_vector(int rows, int cols, const std::vector<float>& values,
                           bool requires_grad = false) {
    return from_vector(1, rows, cols, values, requires_grad);
}

/**
 * @brief Crea un tensor aleatorio 3D.
 */
inline Tensor random_tensor(int batch, int rows, int cols, bool requires_grad, std::mt19937& rng) {
    auto t = std::make_shared<TensorImpl>(batch, rows, cols, requires_grad);
    float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& v : t->data) v = dist(rng);
#ifdef USE_CUDA
    t->to_device();
#endif
    return t;
}

/**
 * @brief Sobrecarga para tensor aleatorio 2D (batch = 1).
 */
inline Tensor random_tensor(int rows, int cols, bool requires_grad, std::mt19937& rng) {
    return random_tensor(1, rows, cols, requires_grad, rng);
}

/**
 * @brief Crea un tensor lleno de ceros 3D.
 */
inline Tensor zeros(int batch, int rows, int cols, bool requires_grad = false) {
    return make_tensor(batch, rows, cols, requires_grad);
}

/**
 * @brief Sobrecarga para tensor de ceros 2D (batch = 1).
 */
inline Tensor zeros(int rows, int cols, bool requires_grad = false) {
    return make_tensor(1, rows, cols, requires_grad);
}

// ======================= Backward (Backpropagation) =======================

/**
 * @brief Construye un orden topológico del grafo computacional mediante DFS.
 *
 * Recorre recursivamente los padres de cada tensor y los agrega al vector
 * `order` en orden topológico (padres antes que hijos). El vector `visited`
 * evita procesar un nodo más de una vez.
 *
 * @param t       Tensor raíz desde donde empezar el recorrido.
 * @param order   [out] Vector donde se acumula el orden topológico.
 * @param visited [out] Conjunto de nodos ya visitados (para evitar ciclos).
 */
inline void build_topo(const Tensor& t, std::vector<Tensor>& order,
                        std::vector<TensorImpl*>& visited) {
    if (std::find(visited.begin(), visited.end(), t.get()) != visited.end()) return;
    visited.push_back(t.get());
    for (auto& p : t->parents) build_topo(p, order, visited);
    order.push_back(t);
}

/**
 * @brief Ejecuta backpropagation desde un escalar de pérdida.
 *
 * Calcula los gradientes de todos los tensores en el grafo computacional
 * aplicando la **regla de la cadena** en orden inverso (del loss hacia las entradas).
 *
 * Algoritmo:
 * 1. Construye el orden topológico del grafo con build_topo().
 * 2. Inicializa d(loss)/d(loss) = 1.0.
 * 3. Recorre el grafo en orden inverso, llamando a backward_fn de cada nodo.
 *
 * @param loss Tensor escalar (1×1) con el valor de la pérdida.
 * @throws std::runtime_error Si loss no es un escalar 1×1.
 */
inline void backward(const Tensor& loss) {
    if (loss->batch != 1 || loss->rows != 1 || loss->cols != 1)
        throw std::runtime_error("backward() debe llamarse sobre un escalar 1x1x1");
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

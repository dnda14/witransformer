/**
 * @file nn.hpp
 * @brief Capas de red neuronal construidas sobre las operaciones de ops.hpp.
 *
 * Define las capas fundamentales del Vision Transformer:
 * - **Linear:** Transformación afín y = x·W + b.
 * - **LayerNorm:** Normalización por capa para estabilizar el entrenamiento.
 * - **MultiHeadSelfAttention:** Mecanismo de atención multi-cabeza.
 * - **MLP:** Perceptrón multicapa con activación GELU.
 * - **TransformerBlock:** Bloque completo pre-norm con conexiones residuales.
 *
 * Todas las capas heredan de Module, que provee una interfaz uniforme
 * para recolectar parámetros entrenables (usada por el optimizador).
 */
#pragma once

#include "ops.hpp"
#include <vector>
#include <random>
#include <cmath>

namespace vit {

/**
 * @brief Interfaz base para todas las capas de la red neuronal.
 *
 * Toda capa debe implementar parameters() para que el optimizador pueda
 * acceder a sus pesos y actualizarlos durante el entrenamiento.
 */
struct Module {
    /**
     * @brief Retorna todos los tensores entrenables de esta capa.
     * @return Vector de Tensores con requires_grad = true.
     */
    virtual std::vector<Tensor> parameters() = 0;
    virtual ~Module() = default;
};

/**
 * @brief Capa Lineal (fully-connected): y = x·W + b.
 *
 * Realiza una transformación afín: multiplica la entrada por una matriz de
 * pesos W y suma un vector de sesgo (bias) b. Es el bloque fundamental
 * de las redes neuronales.
 *
 * - W se inicializa con Xavier/Glorot (ver random_tensor).
 * - b se inicializa a cero.
 */
struct Linear : Module {
    Tensor W;      ///< Matriz de pesos con forma (in_dim, out_dim).
    Tensor b;      ///< Vector de sesgo con forma (1, out_dim).
    int in_dim;    ///< Dimensión de entrada.
    int out_dim;   ///< Dimensión de salida.

    /**
     * @brief Construye una capa lineal con pesos inicializados aleatoriamente.
     * @param in_dim_  Número de características de entrada.
     * @param out_dim_ Número de características de salida.
     * @param rng      Generador de números aleatorios para la inicialización.
     */
    Linear(int in_dim_, int out_dim_, std::mt19937& rng) : in_dim(in_dim_), out_dim(out_dim_) {
        W = random_tensor(in_dim, out_dim, true, rng);
        b = zeros(1, out_dim, true);
    }

    /**
     * @brief Forward pass: calcula y = x·W + b.
     * @param x Tensor de entrada con forma (batch, in_dim).
     * @return  Tensor de salida con forma (batch, out_dim).
     */
    Tensor forward(const Tensor& x) { return add_row_broadcast(matmul(x, W), b); }

    std::vector<Tensor> parameters() override { return {W, b}; }
};

/**
 * @brief Layer Normalization (Ba et al., 2016).
 *
 * Normaliza cada fila (muestra) a media 0 y varianza 1, luego aplica
 * una escala (gamma) y desplazamiento (beta) aprendibles:
 *   out[i,j] = gamma[j] × (x[i,j] - mean_i) / √(var_i + ε) + beta[j]
 *
 * Esto estabiliza el entrenamiento evitando que las activaciones crezcan
 * o se desvanezcan entre capas.
 */
struct LayerNorm : Module {
    Tensor gamma;  ///< Factor de escala aprendible, forma (1, dim), inicializado a 1.
    Tensor beta;   ///< Factor de desplazamiento aprendible, forma (1, dim), inicializado a 0.
    int dim;       ///< Dimensión de la normalización (número de columnas).

    /**
     * @brief Construye una LayerNorm para vectores de dimensión dim_.
     * @param dim_ Dimensión de las características a normalizar.
     */
    explicit LayerNorm(int dim_) : dim(dim_) {
        gamma = from_vector(1, dim, std::vector<float>(dim, 1.0f), true);
        beta = zeros(1, dim, true);
    }

    /**
     * @brief Forward pass: normaliza la entrada por filas.
     * @param x Tensor de entrada con forma (seq_len, dim).
     * @return  Tensor normalizado con forma (seq_len, dim).
     */
    Tensor forward(const Tensor& x) { return layer_norm(x, gamma, beta); }

    std::vector<Tensor> parameters() override { return {gamma, beta}; }
};

/**
 * @brief Multi-Head Self-Attention (Vaswani et al., 2017).
 *
 * Permite a la red evaluar la relación entre todos los pares de tokens
 * (parches de imagen) simultáneamente. Divide la atención en múltiples
 * "cabezas" para capturar diferentes tipos de relaciones.
 *
 * Flujo interno:
 * 1. Proyecta la entrada a Q (queries), K (keys) y V (values).
 * 2. Divide Q, K, V en num_heads cabezas de dimensión head_dim.
 * 3. Para cada cabeza: scores = softmax(Q·Kᵀ / √head_dim), out = scores·V.
 * 4. Concatena las salidas de todas las cabezas.
 * 5. Aplica una proyección lineal final.
 *
 * Fórmula: Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
 */
struct MultiHeadSelfAttention : Module {
    int dim;       ///< Dimensión total del modelo (embed_dim).
    int num_heads; ///< Número de cabezas de atención.
    int head_dim;  ///< Dimensión por cabeza (dim / num_heads).
    Linear q_proj; ///< Proyección lineal para Queries.
    Linear k_proj; ///< Proyección lineal para Keys.
    Linear v_proj; ///< Proyección lineal para Values.
    Linear out_proj; ///< Proyección lineal de salida (después de concatenar cabezas).

    /**
     * @brief Construye el módulo de atención multi-cabeza.
     * @param dim_       Dimensión del modelo (debe ser divisible por num_heads_).
     * @param num_heads_ Número de cabezas de atención.
     * @param rng        Generador de números aleatorios.
     * @throws std::runtime_error Si dim_ no es divisible por num_heads_.
     */
    MultiHeadSelfAttention(int dim_, int num_heads_, std::mt19937& rng)
        : dim(dim_), num_heads(num_heads_), head_dim(dim_ / num_heads_),
          q_proj(dim_, dim_, rng), k_proj(dim_, dim_, rng),
          v_proj(dim_, dim_, rng), out_proj(dim_, dim_, rng) {
        if (dim_ % num_heads_ != 0)
            throw std::runtime_error("dim debe ser divisible entre num_heads");
    }

    /**
     * @brief Forward pass: aplica self-attention multi-cabeza.
     * @param x Tensor de entrada con forma (seq_len, dim).
     * @return  Tensor de salida con forma (seq_len, dim).
     */
    Tensor forward(const Tensor& x) {
        Tensor Q = q_proj.forward(x);
        Tensor K = k_proj.forward(x);
        Tensor V = v_proj.forward(x);

        float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<Tensor> head_outputs;
        head_outputs.reserve(num_heads);
        for (int h = 0; h < num_heads; ++h) {
            Tensor Qh = slice_cols(Q, h * head_dim, head_dim); // (seq, head_dim)
            Tensor Kh = slice_cols(K, h * head_dim, head_dim);
            Tensor Vh = slice_cols(V, h * head_dim, head_dim);

            Tensor scores = scale(matmul(Qh, transpose(Kh)), scale_factor); // (seq, seq)
            Tensor attn = softmax_rows(scores);       // probabilidades de atención
            Tensor headOut = matmul(attn, Vh);         // (seq, head_dim)
            head_outputs.push_back(headOut);
        }
        Tensor concat = concat_cols(head_outputs); // (seq, dim)
        return out_proj.forward(concat);
    }

    std::vector<Tensor> parameters() override {
        std::vector<Tensor> p;
        for (auto* lin : {&q_proj, &k_proj, &v_proj, &out_proj})
            for (auto& t : lin->parameters()) p.push_back(t);
        return p;
    }
};

/**
 * @brief Perceptrón Multicapa (MLP) con activación GELU.
 *
 * Red feed-forward de dos capas lineales con GELU en el medio:
 *   MLP(x) = fc2(GELU(fc1(x)))
 *
 * En el ViT, el MLP expande la dimensión al cuádruple (hidden_dim) y
 * luego la reduce de vuelta a dim.
 */
struct MLP : Module {
    Linear fc1; ///< Primera capa lineal: dim → hidden_dim.
    Linear fc2; ///< Segunda capa lineal: hidden_dim → dim.

    /**
     * @brief Construye el MLP.
     * @param dim        Dimensión de entrada y salida.
     * @param hidden_dim Dimensión intermedia (típicamente 4 × dim en ViT).
     * @param rng        Generador de números aleatorios.
     */
    MLP(int dim, int hidden_dim, std::mt19937& rng) : fc1(dim, hidden_dim, rng), fc2(hidden_dim, dim, rng) {}

    /**
     * @brief Forward pass: fc1 → GELU → fc2.
     * @param x Tensor de entrada con forma (seq_len, dim).
     * @return  Tensor de salida con forma (seq_len, dim).
     */
    Tensor forward(const Tensor& x) { return fc2.forward(gelu(fc1.forward(x))); }

    std::vector<Tensor> parameters() override {
        auto p1 = fc1.parameters(), p2 = fc2.parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }
};

/**
 * @brief Bloque Transformer con pre-normalización y conexiones residuales.
 *
 * Implementa un bloque estándar de Vision Transformer (pre-norm, como en
 * el paper original de ViT):
 *
 *   x₂ = x + Attention(LayerNorm(x))     ← conexión residual 1
 *   x₃ = x₂ + MLP(LayerNorm(x₂))        ← conexión residual 2
 *
 * La pre-normalización (LayerNorm antes de Attention/MLP) es más estable
 * que la post-normalización original de Vaswani et al.
 */
struct TransformerBlock : Module {
    LayerNorm ln1; ///< LayerNorm antes de la atención.
    LayerNorm ln2; ///< LayerNorm antes del MLP.
    MultiHeadSelfAttention attn; ///< Módulo de atención multi-cabeza.
    MLP mlp;       ///< Perceptrón multicapa.

    /**
     * @brief Construye un bloque Transformer.
     * @param dim        Dimensión del modelo.
     * @param num_heads  Número de cabezas de atención.
     * @param mlp_hidden Dimensión oculta del MLP (típicamente 4 × dim).
     * @param rng        Generador de números aleatorios.
     */
    TransformerBlock(int dim, int num_heads, int mlp_hidden, std::mt19937& rng)
        : ln1(dim), ln2(dim), attn(dim, num_heads, rng), mlp(dim, mlp_hidden, rng) {}

    /**
     * @brief Forward pass: pre-norm attention + residual, pre-norm MLP + residual.
     * @param x Tensor de entrada con forma (seq_len, dim).
     * @return  Tensor de salida con forma (seq_len, dim).
     */
    Tensor forward(const Tensor& x) {
        Tensor a = attn.forward(ln1.forward(x));
        Tensor x2 = add(x, a);              // conexión residual 1
        Tensor m = mlp.forward(ln2.forward(x2));
        Tensor x3 = add(x2, m);             // conexión residual 2
        return x3;
    }

    std::vector<Tensor> parameters() override {
        std::vector<Tensor> p;
        for (auto& t : ln1.parameters()) p.push_back(t);
        for (auto& t : attn.parameters()) p.push_back(t);
        for (auto& t : ln2.parameters()) p.push_back(t);
        for (auto& t : mlp.parameters()) p.push_back(t);
        return p;
    }
};

} // namespace vit

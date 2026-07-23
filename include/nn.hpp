// nn.hpp — Capas de red neuronal construidas sobre las primitivas de ops.hpp.
#pragma once

#include "ops.hpp"
#include <vector>
#include <random>
#include <cmath>

namespace vit {

// Interfaz común: toda capa puede exponer sus parámetros para el optimizador
// y para guardar/cargar pesos.
struct Module {
    virtual std::vector<Tensor> parameters() = 0;
    virtual ~Module() = default;
};

// -------------------- Linear: y = x W + b --------------------
struct Linear : Module {
    Tensor W; // (in, out)
    Tensor b; // (1, out)
    int in_dim, out_dim;

    Linear(int in_dim_, int out_dim_, std::mt19937& rng) : in_dim(in_dim_), out_dim(out_dim_) {
        W = random_tensor(in_dim, out_dim, true, rng);
        b = zeros(1, out_dim, true);
    }

    Tensor forward(const Tensor& x) { return add_row_broadcast(matmul(x, W), b); }

    std::vector<Tensor> parameters() override { return {W, b}; }
};

// -------------------- LayerNorm --------------------
struct LayerNorm : Module {
    Tensor gamma, beta;
    int dim;
    explicit LayerNorm(int dim_) : dim(dim_) {
        gamma = from_vector(1, dim, std::vector<float>(dim, 1.0f), true);
        beta = zeros(1, dim, true);
    }
    Tensor forward(const Tensor& x) { return layer_norm(x, gamma, beta); }
    std::vector<Tensor> parameters() override { return {gamma, beta}; }
};

// -------------------- Multi-Head Self-Attention --------------------
struct MultiHeadSelfAttention : Module {
    int dim, num_heads, head_dim;
    Linear q_proj, k_proj, v_proj, out_proj;

    MultiHeadSelfAttention(int dim_, int num_heads_, std::mt19937& rng)
        : dim(dim_), num_heads(num_heads_), head_dim(dim_ / num_heads_),
          q_proj(dim_, dim_, rng), k_proj(dim_, dim_, rng),
          v_proj(dim_, dim_, rng), out_proj(dim_, dim_, rng) {
        if (dim_ % num_heads_ != 0)
            throw std::runtime_error("dim debe ser divisible entre num_heads");
    }

    // x: (seq_len, dim) -> (seq_len, dim)
    Tensor forward(const Tensor& x) {
        Tensor Q = q_proj.forward(x);
        Tensor K = k_proj.forward(x);
        Tensor V = v_proj.forward(x);

        float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<Tensor> head_outputs;
        head_outputs.reserve(num_heads);
        for (int h = 0; h < num_heads; ++h) {
            Tensor Qh = slice_cols(Q, h * head_dim, head_dim);
            Tensor Kh = slice_cols(K, h * head_dim, head_dim);
            Tensor Vh = slice_cols(V, h * head_dim, head_dim);

            Tensor scores = scale(matmul(Qh, transpose(Kh)), scale_factor); // (seq, seq)
            Tensor attn = softmax_rows(scores);
            Tensor headOut = matmul(attn, Vh); // (seq, head_dim)
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

// -------------------- MLP (feed-forward) --------------------
struct MLP : Module {
    Linear fc1, fc2;
    MLP(int dim, int hidden_dim, std::mt19937& rng) : fc1(dim, hidden_dim, rng), fc2(hidden_dim, dim, rng) {}
    Tensor forward(const Tensor& x) { return fc2.forward(gelu(fc1.forward(x))); }
    std::vector<Tensor> parameters() override {
        auto p1 = fc1.parameters(), p2 = fc2.parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }
};

// -------------------- Bloque Transformer (pre-norm, como en ViT) --------------------
struct TransformerBlock : Module {
    LayerNorm ln1, ln2;
    MultiHeadSelfAttention attn;
    MLP mlp;

    TransformerBlock(int dim, int num_heads, int mlp_hidden, std::mt19937& rng)
        : ln1(dim), ln2(dim), attn(dim, num_heads, rng), mlp(dim, mlp_hidden, rng) {}

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

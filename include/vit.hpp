// vit.hpp — Ensambla el Vision Transformer completo para MNIST.
#pragma once

#include "nn.hpp"
#include <vector>
#include <fstream>
#include <stdexcept>

namespace vit {

struct ViTConfig {
    int image_size = 28;
    int patch_size = 7;     // 28/7 = 4 -> 4x4 = 16 parches
    int in_channels = 1;
    int embed_dim = 64;
    int depth = 4;           // número de bloques Transformer
    int num_heads = 4;
    int mlp_hidden = 128;
    int num_classes = 10;

    int num_patches() const {
        int side = image_size / patch_size;
        return side * side;
    }
    int patch_dim() const { return patch_size * patch_size * in_channels; }
};

// Convierte una imagen 28x28 (784 floats, orden fila-mayor) en una matriz
// (num_patches, patch_dim), recorriendo la imagen en una grilla de parches.
inline Tensor patchify(const std::vector<float>& image, const ViTConfig& cfg) {
    int side = cfg.image_size / cfg.patch_size;
    auto out = make_tensor(cfg.num_patches(), cfg.patch_dim(), false);
    int patch_idx = 0;
    for (int py = 0; py < side; ++py) {
        for (int px = 0; px < side; ++px) {
            int col = 0;
            for (int iy = 0; iy < cfg.patch_size; ++iy) {
                for (int ix = 0; ix < cfg.patch_size; ++ix) {
                    int y = py * cfg.patch_size + iy;
                    int x = px * cfg.patch_size + ix;
                    out->at(patch_idx, col++) = image[y * cfg.image_size + x];
                }
            }
            ++patch_idx;
        }
    }
    return out;
}

struct VisionTransformer : Module {
    ViTConfig cfg;
    Linear patch_embed;
    Tensor cls_token;   // (1, embed_dim)
    Tensor pos_embed;   // (num_patches+1, embed_dim)
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    LayerNorm final_ln;
    Linear head;

    VisionTransformer(const ViTConfig& cfg_, std::mt19937& rng)
        : cfg(cfg_),
          patch_embed(cfg_.patch_dim(), cfg_.embed_dim, rng),
          final_ln(cfg_.embed_dim),
          head(cfg_.embed_dim, cfg_.num_classes, rng) {
        cls_token = random_tensor(1, cfg.embed_dim, true, rng);
        pos_embed = random_tensor(cfg.num_patches() + 1, cfg.embed_dim, true, rng);
        for (int i = 0; i < cfg.depth; ++i)
            blocks.push_back(std::make_shared<TransformerBlock>(cfg.embed_dim, cfg.num_heads, cfg.mlp_hidden, rng));
    }

    // image: 784 floats normalizados en [0,1]. Devuelve logits (1, num_classes).
    Tensor forward(const std::vector<float>& image) {
        Tensor patches = patchify(image, cfg);              // (num_patches, patch_dim)
        Tensor embedded = patch_embed.forward(patches);      // (num_patches, embed_dim)
        Tensor with_cls = concat_rows_helper(cls_token, embedded); // (num_patches+1, embed_dim)
        Tensor x = add(with_cls, pos_embed);
        for (auto& blk : blocks) x = blk->forward(x);
        x = final_ln.forward(x);
        Tensor cls_out = select_row(x, 0);                   // (1, embed_dim)
        Tensor logits = head.forward(cls_out);                // (1, num_classes)
        return logits;
    }

    // concatenar cls_token (1,d) con el resto de filas (n,d) -> (n+1,d)
    static Tensor concat_rows_helper(const Tensor& cls, const Tensor& rest) {
        int d = cls->cols;
        auto out = make_tensor(rest->rows + 1, d, cls->requires_grad || rest->requires_grad);
        for (int j = 0; j < d; ++j) out->at(0, j) = cls->at(0, j);
        for (int i = 0; i < rest->rows; ++i)
            for (int j = 0; j < d; ++j)
                out->at(i + 1, j) = rest->at(i, j);
        out->parents = {cls, rest};
        TensorImpl* out_raw = out.get();
        out->backward_fn = [cls, rest, out_raw, d]() {
            if (cls->requires_grad)
                for (int j = 0; j < d; ++j) cls->g(0, j) += out_raw->g(0, j);
            if (rest->requires_grad)
                for (int i = 0; i < rest->rows; ++i)
                    for (int j = 0; j < d; ++j)
                        rest->g(i, j) += out_raw->g(i + 1, j);
        };
        return out;
    }

    std::vector<Tensor> parameters() override {
        std::vector<Tensor> p;
        for (auto& t : patch_embed.parameters()) p.push_back(t);
        p.push_back(cls_token);
        p.push_back(pos_embed);
        for (auto& blk : blocks)
            for (auto& t : blk->parameters()) p.push_back(t);
        for (auto& t : final_ln.parameters()) p.push_back(t);
        for (auto& t : head.parameters()) p.push_back(t);
        return p;
    }

    // ---------- Serialización de pesos ----------
    static constexpr uint32_t MAGIC = 0x56495443; // "VITC"

    void save(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("No se pudo abrir para escritura: " + path);
        uint32_t magic = MAGIC;
        f.write(reinterpret_cast<char*>(&magic), sizeof(magic));
        f.write(reinterpret_cast<char*>(&cfg), sizeof(ViTConfig));
        auto params = parameters();
        uint32_t n = static_cast<uint32_t>(params.size());
        f.write(reinterpret_cast<char*>(&n), sizeof(n));
        for (auto& t : params) {
            f.write(reinterpret_cast<char*>(&t->rows), sizeof(t->rows));
            f.write(reinterpret_cast<char*>(&t->cols), sizeof(t->cols));
            f.write(reinterpret_cast<const char*>(t->data.data()), sizeof(float) * t->data.size());
        }
    }

    // Carga pesos en un modelo ya construido con la MISMA configuración (cfg).
    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("No se pudo abrir para lectura: " + path);
        uint32_t magic = 0;
        f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != MAGIC) throw std::runtime_error("Archivo de pesos inválido (magic incorrecto)");
        ViTConfig file_cfg;
        f.read(reinterpret_cast<char*>(&file_cfg), sizeof(ViTConfig));
        if (file_cfg.embed_dim != cfg.embed_dim || file_cfg.depth != cfg.depth ||
            file_cfg.num_heads != cfg.num_heads || file_cfg.patch_size != cfg.patch_size ||
            file_cfg.image_size != cfg.image_size || file_cfg.num_classes != cfg.num_classes) {
            throw std::runtime_error("La configuración del archivo no coincide con el modelo actual");
        }
        uint32_t n = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        auto params = parameters();
        if (n != params.size()) throw std::runtime_error("Número de tensores de parámetros no coincide");
        for (auto& t : params) {
            int r, c;
            f.read(reinterpret_cast<char*>(&r), sizeof(r));
            f.read(reinterpret_cast<char*>(&c), sizeof(c));
            if (r != t->rows || c != t->cols) throw std::runtime_error("Forma de tensor no coincide al cargar pesos");
            f.read(reinterpret_cast<char*>(t->data.data()), sizeof(float) * t->data.size());
        }
    }
};

} // namespace vit

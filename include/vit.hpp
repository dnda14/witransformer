/**
 * @file vit.hpp
 * @brief Ensambla el Vision Transformer (ViT) completo para clasificación de MNIST.
 *
 * Este archivo une todas las capas de nn.hpp para construir la arquitectura
 * completa del ViT. También implementa la serialización (guardar/cargar)
 * de los pesos del modelo en formato binario.
 *
 * **Flujo del forward pass:**
 * 1. Patchify: imagen 28×28 → 16 parches de 7×7 (aplanados a vectores de 49).
 * 2. Patch Embedding: proyección lineal de cada parche a embed_dim.
 * 3. CLS Token: se prepone un token especial [CLS] a la secuencia.
 * 4. Positional Embedding: se suma información posicional a cada token.
 * 5. Transformer Blocks: se aplican `depth` bloques transformer.
 * 6. Clasificación: se extrae el token [CLS] y se pasa por una capa lineal final.
 */
#pragma once

#include "nn.hpp"
#include <vector>
#include <fstream>
#include <stdexcept>

namespace vit {

/**
 * @brief Configuración de hiperparámetros del Vision Transformer.
 *
 * Contiene todos los hiperparámetros que definen la arquitectura del modelo.
 * Los valores por defecto están optimizados para MNIST (28×28, 1 canal, 10 clases).
 */
struct ViTConfig {
    int image_size = 28;     ///< Tamaño de la imagen de entrada (lado del cuadrado, en píxeles).
    int patch_size = 7;      ///< Tamaño de cada parche (28/7 = 4 → 4×4 = 16 parches).
    int in_channels = 1;     ///< Canales de la imagen (1 para escala de grises, 3 para RGB).
    int embed_dim = 64;      ///< Dimensión del espacio de embeddings (tamaño de cada vector token).
    int depth = 4;           ///< Número de bloques Transformer apilados.
    int num_heads = 4;       ///< Número de cabezas de atención por bloque.
    int mlp_hidden = 128;    ///< Dimensión oculta del MLP dentro de cada bloque (2× embed_dim).
    int num_classes = 10;    ///< Número de clases a clasificar (10 dígitos para MNIST).

    /**
     * @brief Calcula el número total de parches de la imagen.
     * @return (image_size / patch_size)² — ejemplo: (28/7)² = 16.
     */
    int num_patches() const {
        int side = image_size / patch_size;
        return side * side;
    }

    /**
     * @brief Calcula la dimensión de cada parche aplanado.
     * @return patch_size × patch_size × in_channels — ejemplo: 7×7×1 = 49.
     */
    int patch_dim() const { return patch_size * patch_size * in_channels; }
};

/**
 * @brief Convierte una imagen plana en una matriz de parches.
 *
 * Recorre la imagen en una grilla de parches de tamaño patch_size × patch_size,
 * extrayendo cada parche y aplanándolo en un vector fila.
 *
 * @param image Vector de 784 floats (28×28 píxeles) normalizados en [0, 1], orden fila-mayor.
 * @param cfg   Configuración del ViT (para obtener patch_size y num_patches).
 * @return      Tensor con forma (num_patches, patch_dim) — ejemplo: (16, 49).
 */
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
#ifdef USE_CUDA
    out->to_device();
#endif
    return out;
}

/**
 * @brief El Vision Transformer completo para clasificación de imágenes.
 *
 * Ensambla todas las capas previamente definidas en una arquitectura ViT funcional.
 * Implementa el forward pass completo (imagen → predicción de clase) y
 * la serialización de pesos a disco.
 */
struct VisionTransformer : Module {
    ViTConfig cfg;         ///< Configuración de hiperparámetros del modelo.
    Linear patch_embed;    ///< Proyección lineal de parches: (patch_dim → embed_dim).
    Tensor cls_token;      ///< Token [CLS] aprendible, forma (1, embed_dim).
    Tensor pos_embed;      ///< Embeddings posicionales aprendibles, forma (num_patches+1, embed_dim).
    std::vector<std::shared_ptr<TransformerBlock>> blocks; ///< Bloques Transformer apilados.
    LayerNorm final_ln;    ///< LayerNorm final antes de la clasificación.
    Linear head;           ///< Capa de clasificación: (embed_dim → num_classes).

    /**
     * @brief Construye el Vision Transformer con la configuración dada.
     * @param cfg_ Configuración de hiperparámetros.
     * @param rng  Generador de números aleatorios para la inicialización de pesos.
     */
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

    /**
     * @brief Forward pass completo: imagen → logits de clasificación.
     *
     * Flujo:
     * 1. patchify: imagen (784 floats) → parches (16, 49)
     * 2. patch_embed: parches (16, 49) → embeddings (16, 64)
     * 3. concat CLS: (16, 64) → (17, 64) con token [CLS] al inicio
     * 4. + pos_embed: suma embeddings posicionales
     * 5. transformer blocks × depth: (17, 64) → (17, 64)
     * 6. LayerNorm final + seleccionar CLS → (1, 64)
     * 7. head: (1, 64) → (1, 10) logits
     *
     * @param image Vector de 784 floats normalizados en [0, 1].
     * @return      Tensor de logits con forma (1, num_classes).
     */
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

    /**
     * @brief Concatena el token CLS (1, d) con el resto de embeddings (n, d).
     *
     * Crea un tensor de (n+1, d) colocando cls en la fila 0 y rest en las filas 1..n.
     * Implementa su propio backward para propagar gradientes a cls y rest.
     *
     * @param cls  Token CLS con forma (1, d).
     * @param rest Embeddings de parches con forma (n, d).
     * @return     Tensor concatenado con forma (n+1, d).
     */
    static Tensor concat_rows_helper(const Tensor& cls, const Tensor& rest) {
        int d = cls->cols;
        auto out = make_tensor(rest->rows + 1, d, cls->requires_grad || rest->requires_grad);
#ifdef USE_CUDA
        cuda::concat_rows_copy(cls->d_data, out->d_data, 1, d, 0);
        cuda::concat_rows_copy(rest->d_data, out->d_data, rest->rows, d, 1);
#else
        for (int j = 0; j < d; ++j) out->at(0, j) = cls->at(0, j);
        for (int i = 0; i < rest->rows; ++i)
            for (int j = 0; j < d; ++j)
                out->at(i + 1, j) = rest->at(i, j);
#endif
        out->parents = {cls, rest};
        TensorImpl* out_raw = out.get();
        out->backward_fn = [cls, rest, out_raw, d]() {
#ifdef USE_CUDA
            if (cls->requires_grad)
                cuda::concat_rows_bwd_part(out_raw->d_grad, cls->d_grad, 1, d, 0);
            if (rest->requires_grad)
                cuda::concat_rows_bwd_part(out_raw->d_grad, rest->d_grad, rest->rows, d, 1);
#else
            if (cls->requires_grad)
                for (int j = 0; j < d; ++j) cls->g(0, j) += out_raw->g(0, j);
            if (rest->requires_grad)
                for (int i = 0; i < rest->rows; ++i)
                    for (int j = 0; j < d; ++j)
                        rest->g(i, j) += out_raw->g(i + 1, j);
#endif
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

    // ======================= Serialización de Pesos =======================

    static constexpr uint32_t MAGIC = 0x56495443; ///< Número mágico "VITC" para validar archivos de pesos.

    /**
     * @brief Guarda todos los pesos del modelo en un archivo binario.
     *
     * Formato del archivo:
     * 1. Magic number (4 bytes): 0x56495443 ("VITC")
     * 2. ViTConfig (sizeof(ViTConfig) bytes)
     * 3. Número de tensores (4 bytes)
     * 4. Para cada tensor: rows (4B) + cols (4B) + data (rows×cols×4 bytes)
     *
     * @param path Ruta del archivo de salida (ej: "models/vit_mnist.bin").
     * @throws std::runtime_error Si no se puede abrir el archivo.
     */
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
#ifdef USE_CUDA
            t->to_host();
#endif
            f.write(reinterpret_cast<char*>(&t->rows), sizeof(t->rows));
            f.write(reinterpret_cast<char*>(&t->cols), sizeof(t->cols));
            f.write(reinterpret_cast<const char*>(t->data.data()), sizeof(float) * t->data.size());
        }
    }

    /**
     * @brief Carga pesos desde un archivo binario a un modelo ya construido.
     *
     * El modelo debe haberse construido con la **misma configuración** (cfg)
     * que se usó al guardar. Se validan el magic number, la configuración
     * y las dimensiones de cada tensor.
     *
     * @param path Ruta del archivo de pesos (ej: "models/vit_mnist.bin").
     * @throws std::runtime_error Si el archivo es inválido o la configuración no coincide.
     */
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
#ifdef USE_CUDA
            t->to_device();
#endif
        }
    }
};

} // namespace vit

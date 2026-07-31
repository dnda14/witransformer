/**
 * @file mnist.hpp
 * @brief Lector del formato binario IDX usado por el dataset MNIST.
 *
 * MNIST es un dataset clásico de dígitos escritos a mano (0-9) con:
 * - 60,000 imágenes de entrenamiento
 * - 10,000 imágenes de prueba
 * - Cada imagen es de 28×28 píxeles en escala de grises
 *
 * Los archivos usan el formato IDX con bytes en orden big-endian.
 * Este módulo lee ambos archivos (imágenes y etiquetas) y los convierte
 * a vectores de floats normalizados en [0, 1].
 */
#pragma once

#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <string>

namespace vit {

/**
 * @brief Contiene un dataset MNIST completo (imágenes + etiquetas).
 */
struct MnistDataset {
    std::vector<std::vector<float>> images; ///< Imágenes: cada una es un vector de 784 floats en [0, 1].
    std::vector<int> labels;                ///< Etiquetas: un entero (0-9) por imagen.
    int rows = 0;   ///< Filas de cada imagen (28 para MNIST).
    int cols = 0;   ///< Columnas de cada imagen (28 para MNIST).

    /** @brief Retorna el número de imágenes en el dataset. */
    size_t size() const { return images.size(); }
};

/**
 * @brief Lee un entero de 32 bits en formato big-endian desde un archivo binario.
 *
 * El formato IDX almacena todos los enteros en big-endian (byte más significativo
 * primero), independientemente de la arquitectura del sistema.
 *
 * @param f Stream de entrada del archivo binario.
 * @return  Valor del entero en formato nativo del sistema.
 */
inline uint32_t read_be_uint32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
           (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
}

/**
 * @brief Carga el dataset MNIST desde archivos binarios IDX.
 *
 * Lee los archivos de imágenes y etiquetas en formato IDX, valida los
 * magic numbers, y convierte los píxeles (bytes 0-255) a floats
 * normalizados en [0.0, 1.0] dividiendo por 255.
 *
 * @param images_path Ruta al archivo de imágenes (ej: "data/train-images-idx3-ubyte").
 * @param labels_path Ruta al archivo de etiquetas (ej: "data/train-labels-idx1-ubyte").
 * @return MnistDataset con todas las imágenes y etiquetas cargadas.
 * @throws std::runtime_error Si no se pueden abrir los archivos, los magic numbers
 *         son inválidos, o el número de imágenes y etiquetas no coincide.
 */
inline MnistDataset load_mnist(const std::string& images_path, const std::string& labels_path) {
    std::ifstream fi(images_path, std::ios::binary);
    if (!fi) throw std::runtime_error("No se pudo abrir: " + images_path);
    std::ifstream fl(labels_path, std::ios::binary);
    if (!fl) throw std::runtime_error("No se pudo abrir: " + labels_path);

    uint32_t magic_i = read_be_uint32(fi);
    if (magic_i != 0x00000803) throw std::runtime_error("Magic number inválido en archivo de imágenes");
    uint32_t n_images = read_be_uint32(fi);
    uint32_t rows = read_be_uint32(fi);
    uint32_t cols = read_be_uint32(fi);

    uint32_t magic_l = read_be_uint32(fl);
    if (magic_l != 0x00000801) throw std::runtime_error("Magic number inválido en archivo de etiquetas");
    uint32_t n_labels = read_be_uint32(fl);
    if (n_labels != n_images) throw std::runtime_error("El número de imágenes y etiquetas no coincide");

    MnistDataset ds;
    ds.rows = static_cast<int>(rows);
    ds.cols = static_cast<int>(cols);
    ds.images.resize(n_images);
    ds.labels.resize(n_images);

    size_t pixels_per_image = static_cast<size_t>(rows) * cols;
    std::vector<unsigned char> buf(pixels_per_image);
    for (uint32_t i = 0; i < n_images; ++i) {
        fi.read(reinterpret_cast<char*>(buf.data()), pixels_per_image);
        ds.images[i].resize(pixels_per_image);
        // Normalizar píxeles de [0, 255] a [0.0, 1.0]
        for (size_t p = 0; p < pixels_per_image; ++p)
            ds.images[i][p] = static_cast<float>(buf[p]) / 255.0f;
        unsigned char lbl;
        fl.read(reinterpret_cast<char*>(&lbl), 1);
        ds.labels[i] = static_cast<int>(lbl);
    }
    return ds;
}

} // namespace vit

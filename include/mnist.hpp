// mnist.hpp — Lector del formato binario IDX usado por MNIST.
#pragma once

#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <string>

namespace vit {

struct MnistDataset {
    std::vector<std::vector<float>> images; // cada imagen: 784 floats en [0,1]
    std::vector<int> labels;
    int rows = 0, cols = 0;
    size_t size() const { return images.size(); }
};

inline uint32_t read_be_uint32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
           (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
}

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
        for (size_t p = 0; p < pixels_per_image; ++p)
            ds.images[i][p] = static_cast<float>(buf[p]) / 255.0f;
        unsigned char lbl;
        fl.read(reinterpret_cast<char*>(&lbl), 1);
        ds.labels[i] = static_cast<int>(lbl);
    }
    return ds;
}

} // namespace vit

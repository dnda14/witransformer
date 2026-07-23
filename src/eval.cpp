// eval.cpp — Carga un modelo guardado (.bin) y evalúa sobre el set de test,
// mostrando accuracy global y matriz de confusión.
//
// Uso:
//   ./vit_eval --model models/vit_mnist.bin
//              --test-images data/t10k-images-idx3-ubyte
//              --test-labels data/t10k-labels-idx1-ubyte

#include "../include/vit.hpp"
#include "../include/mnist.hpp"
#include <iostream>
#include <algorithm>
#include <map>
#include <iomanip>

using namespace vit;

int main(int argc, char** argv) {
    std::map<std::string, std::string> kv;
    for (int i = 1; i + 1 < argc; i += 2) kv[argv[i]] = argv[i + 1];
    auto get = [&](const std::string& key, std::string def) {
        auto it = kv.find(key); return it != kv.end() ? it->second : def;
    };
    std::string model_path = get("--model", "models/vit_mnist.bin");
    std::string test_images = get("--test-images", "data/t10k-images-idx3-ubyte");
    std::string test_labels = get("--test-labels", "data/t10k-labels-idx1-ubyte");
    int limit = std::stoi(get("--limit-test", "-1"));

    ViTConfig cfg;
    std::mt19937 rng(0);
    VisionTransformer model(cfg, rng);
    model.load(model_path);
    std::cout << "Modelo cargado desde " << model_path << "\n";

    MnistDataset test = load_mnist(test_images, test_labels);
    if (limit > 0 && static_cast<size_t>(limit) < test.size()) {
        test.images.resize(limit);
        test.labels.resize(limit);
    }

    std::vector<std::vector<int>> confusion(10, std::vector<int>(10, 0));
    int correct = 0;
    for (size_t i = 0; i < test.size(); ++i) {
        Tensor logits = model.forward(test.images[i]);
#ifdef USE_CUDA
        logits->to_host();
#endif
        int pred = static_cast<int>(std::max_element(logits->data.begin(), logits->data.end()) - logits->data.begin());
        int truth = test.labels[i];
        confusion[truth][pred]++;
        if (pred == truth) ++correct;
    }

    double acc = static_cast<double>(correct) / test.size();
    std::cout << "Accuracy en test (" << test.size() << " imagenes): " << acc << "\n\n";

    std::cout << "Matriz de confusion (filas = etiqueta real, columnas = prediccion):\n     ";
    for (int j = 0; j < 10; ++j) std::cout << std::setw(5) << j;
    std::cout << "\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(3) << i << ": ";
        for (int j = 0; j < 10; ++j) std::cout << std::setw(5) << confusion[i][j];
        std::cout << "\n";
    }
    return 0;
}

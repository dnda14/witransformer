/**
 * @file eval.cpp
 * @brief Programa de evaluación (inferencia) del Vision Transformer sobre MNIST.
 *
 * Carga un modelo previamente entrenado y guardado en formato .bin, lo evalúa
 * sobre el conjunto de prueba de MNIST, y muestra:
 * - Accuracy global (porcentaje de predicciones correctas).
 * - Matriz de confusión 10×10 (filas = etiqueta real, columnas = predicción).
 *
 * @par Uso:
 * @code
 * ./vit_eval --model models/vit_mnist.bin
 *            --test-images data/t10k-images-idx3-ubyte
 *            --test-labels data/t10k-labels-idx1-ubyte
 * @endcode
 */

#include "../include/vit.hpp"
#include "../include/mnist.hpp"
#include <iostream>
#include <algorithm>
#include <map>
#include <iomanip>

using namespace vit;

/**
 * @brief Punto de entrada principal del programa de evaluación.
 *
 * Flujo:
 * 1. Parsear argumentos de CLI (--model, --test-images, --test-labels, --limit-test).
 * 2. Construir el modelo ViT con la configuración por defecto y cargar pesos.
 * 3. Cargar el dataset de prueba.
 * 4. Ejecutar inferencia (forward pass) en cada imagen.
 * 5. Calcular accuracy y construir la matriz de confusión.
 * 6. Imprimir resultados.
 */
int main(int argc, char** argv) {
    // --- Parsear argumentos ---
    std::map<std::string, std::string> kv;
    for (int i = 1; i + 1 < argc; i += 2) kv[argv[i]] = argv[i + 1];
    auto get = [&](const std::string& key, std::string def) {
        auto it = kv.find(key); return it != kv.end() ? it->second : def;
    };
    std::string model_path = get("--model", "models/vit_mnist.bin");
    std::string test_images = get("--test-images", "data/t10k-images-idx3-ubyte");
    std::string test_labels = get("--test-labels", "data/t10k-labels-idx1-ubyte");
    int limit = std::stoi(get("--limit-test", "-1"));

    // --- Cargar modelo ---
    ViTConfig cfg;
    std::mt19937 rng(0); // seed no importa, los pesos se sobreescriben con load()
    VisionTransformer model(cfg, rng);
    model.load(model_path);
    std::cout << "Modelo cargado desde " << model_path << "\n";

    // --- Cargar datos de prueba ---
    MnistDataset test = load_mnist(test_images, test_labels);
    if (limit > 0 && static_cast<size_t>(limit) < test.size()) {
        test.images.resize(limit);
        test.labels.resize(limit);
    }

    // --- Inferencia y evaluación ---
    std::vector<std::vector<int>> confusion(10, std::vector<int>(10, 0));
    int correct = 0;
    for (size_t i = 0; i < test.size(); ++i) {
        Tensor logits = model.forward(test.images[i]);
#ifdef USE_CUDA
        logits->to_host(); // traer logits de GPU a CPU para calcular argmax
#endif
        int pred = static_cast<int>(std::max_element(logits->data.begin(), logits->data.end()) - logits->data.begin());
        int truth = test.labels[i];
        confusion[truth][pred]++;
        if (pred == truth) ++correct;
    }

    // --- Resultados ---
    double acc = static_cast<double>(correct) / test.size();
    std::cout << "Accuracy en test (" << test.size() << " imagenes): " << acc << "\n\n";

    // Imprimir matriz de confusión
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

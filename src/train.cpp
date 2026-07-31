/**
 * @file train.cpp
 * @brief Programa principal de entrenamiento del Vision Transformer sobre MNIST.
 *
 * Entrena el ViT desde cero usando backpropagation manual y el optimizador AdamW.
 * Durante el entrenamiento:
 * - Mezcla el dataset en cada época.
 * - Procesa los datos en mini-batches.
 * - Evalúa en el conjunto de prueba al final de cada época.
 * - Guarda automáticamente el mejor modelo (por test_acc).
 * - Registra métricas en un archivo CSV.
 *
 * @par Uso:
 * @code
 * ./vit_train --train-images data/train-images-idx3-ubyte
 *             --train-labels data/train-labels-idx1-ubyte
 *             --test-images  data/t10k-images-idx3-ubyte
 *             --test-labels  data/t10k-labels-idx1-ubyte
 *             --epochs 10 --batch-size 32 --lr 0.001
 *             --weight-decay 0.01
 *             --model-out models/vit_mnist.bin
 *             --metrics-out metrics/metrics.csv
 * @endcode
 * Todos los argumentos tienen valores por defecto razonables (ver Args).
 */

#include "../include/vit.hpp"
#include "../include/optimizer.hpp"
#include "../include/mnist.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <string>
#include <map>

using namespace vit;

/**
 * @brief Argumentos de línea de comandos para el entrenamiento.
 *
 * Todos los campos tienen valores por defecto razonables para MNIST.
 * Se pueden sobreescribir pasando --nombre valor en la línea de comandos.
 */
struct Args {
    std::string train_images = "data/train-images-idx3-ubyte"; ///< Ruta al archivo de imágenes de entrenamiento.
    std::string train_labels = "data/train-labels-idx1-ubyte"; ///< Ruta al archivo de etiquetas de entrenamiento.
    std::string test_images  = "data/t10k-images-idx3-ubyte";  ///< Ruta al archivo de imágenes de prueba.
    std::string test_labels  = "data/t10k-labels-idx1-ubyte";  ///< Ruta al archivo de etiquetas de prueba.
    std::string model_out    = "models/vit_mnist.bin";          ///< Ruta donde guardar el mejor modelo.
    std::string metrics_out  = "metrics/metrics.csv";           ///< Ruta donde guardar las métricas por época.
    int epochs = 5;          ///< Número de épocas de entrenamiento.
    int batch_size = 32;     ///< Tamaño del mini-batch.
    float lr = 1e-3f;        ///< Learning rate (tasa de aprendizaje).
    float weight_decay = 0.0f; ///< Coeficiente de weight decay para AdamW (0 = Adam puro).
    int limit_train = -1;    ///< Limitar el dataset de train a N imágenes (-1 = usar todo).
    int limit_test  = -1;    ///< Limitar el dataset de test a N imágenes (-1 = usar todo).
    unsigned seed = 42;      ///< Semilla para reproducibilidad.
};

/**
 * @brief Parsea los argumentos de línea de comandos en formato --clave valor.
 * @param argc Número de argumentos.
 * @param argv Array de argumentos.
 * @return Struct Args con los valores parseados.
 */
static Args parse_args(int argc, char** argv) {
    Args a;
    std::map<std::string, std::string> kv;
    for (int i = 1; i + 1 < argc; i += 2) kv[argv[i]] = argv[i + 1];
    auto get = [&](const std::string& key, std::string def) {
        auto it = kv.find(key); return it != kv.end() ? it->second : def;
    };
    a.train_images = get("--train-images", a.train_images);
    a.train_labels = get("--train-labels", a.train_labels);
    a.test_images  = get("--test-images", a.test_images);
    a.test_labels  = get("--test-labels", a.test_labels);
    a.model_out    = get("--model-out", a.model_out);
    a.metrics_out  = get("--metrics-out", a.metrics_out);
    a.epochs       = std::stoi(get("--epochs", std::to_string(a.epochs)));
    a.batch_size   = std::stoi(get("--batch-size", std::to_string(a.batch_size)));
    a.lr           = std::stof(get("--lr", std::to_string(a.lr)));
    a.weight_decay = std::stof(get("--weight-decay", std::to_string(a.weight_decay)));
    a.limit_train  = std::stoi(get("--limit-train", std::to_string(a.limit_train)));
    a.limit_test   = std::stoi(get("--limit-test", std::to_string(a.limit_test)));
    a.seed         = static_cast<unsigned>(std::stoi(get("--seed", std::to_string(a.seed))));
    return a;
}

/**
 * @brief Retorna el índice del elemento máximo de un vector (usado para obtener la clase predicha).
 * @param v Vector de probabilidades o logits.
 * @return Índice del elemento con mayor valor.
 */
static int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

/**
 * @brief Punto de entrada principal del programa de entrenamiento.
 *
 * Flujo:
 * 1. Parsear argumentos de CLI.
 * 2. Cargar dataset MNIST (train + test).
 * 3. Construir el modelo ViT con la configuración por defecto.
 * 4. Crear el optimizador AdamW.
 * 5. Para cada época:
 *    a. Mezclar índices de entrenamiento.
 *    b. Iterar mini-batches: forward → loss → backward → optimizer step.
 *    c. Evaluar en el conjunto de test.
 *    d. Guardar métricas y el mejor modelo.
 */
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::mt19937 rng(args.seed);

    // --- Carga de datos ---
    std::cout << "Cargando MNIST...\n";
    MnistDataset train = load_mnist(args.train_images, args.train_labels);
    MnistDataset test  = load_mnist(args.test_images, args.test_labels);
    if (args.limit_train > 0 && static_cast<size_t>(args.limit_train) < train.size()) {
        train.images.resize(args.limit_train);
        train.labels.resize(args.limit_train);
    }
    if (args.limit_test > 0 && static_cast<size_t>(args.limit_test) < test.size()) {
        test.images.resize(args.limit_test);
        test.labels.resize(args.limit_test);
    }
    std::cout << "Train: " << train.size() << " imagenes | Test: " << test.size() << " imagenes\n";

    // --- Construcción del modelo ---
    ViTConfig cfg; // valores por defecto: patch 7x7, embed_dim 64, depth 4, heads 4
    VisionTransformer model(cfg, rng);
    auto params = model.parameters();
    size_t total_params = 0;
    for (auto& p : params) total_params += p->data.size();
    std::cout << "Modelo ViT: " << params.size() << " tensores, " << total_params << " parametros totales\n";

    // --- Optimizador ---
    Adam opt(params, args.lr, 0.9f, 0.999f, 1e-8f, args.weight_decay);
    if (args.weight_decay > 0.0f)
        std::cout << "AdamW weight_decay=" << args.weight_decay << "\n";

    // --- Archivo de métricas ---
    std::ofstream metrics(args.metrics_out);
    metrics << "epoch,train_loss,train_acc,test_loss,test_acc,seconds\n";

    std::vector<int> indices(train.size());
    std::iota(indices.begin(), indices.end(), 0);

    float best_test_acc = -1.0f;

    // --- Bucle de entrenamiento ---
    for (int epoch = 1; epoch <= args.epochs; ++epoch) {
        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(indices.begin(), indices.end(), rng);

        double running_loss = 0.0;
        int correct = 0;
        int n = static_cast<int>(indices.size());

        // Iterar mini-batches
        for (int start = 0; start < n; start += args.batch_size) {
            int end = std::min(start + args.batch_size, n);
            int bs = end - start;
            opt.zero_grad();
            // Acumular gradientes del mini-batch
            for (int bi = start; bi < end; ++bi) {
                int idx = indices[bi];
                Tensor logits = model.forward(train.images[idx]);
                std::vector<float> probs;
                Tensor loss = softmax_cross_entropy(logits, train.labels[idx], &probs);
                backward(loss);
                running_loss += loss->data[0];
                if (argmax(probs) == train.labels[idx]) ++correct;
            }
            // Actualizar pesos (grad_scale = 1/batch_size para promediar gradientes)
            opt.step(1.0f / static_cast<float>(bs));

            // Progreso
            int done = end;
            if ((done / args.batch_size) % 20 == 0 || done == n) {
                std::cout << "\r  epoca " << epoch << ": " << done << "/" << n
                          << " ejemplos procesados" << std::flush;
            }
        }
        std::cout << "\n";

        double train_loss = running_loss / n;
        double train_acc = static_cast<double>(correct) / n;

        // --- Evaluación en test ---
        double test_running_loss = 0.0;
        int test_correct = 0;
        for (size_t i = 0; i < test.size(); ++i) {
            Tensor logits = model.forward(test.images[i]);
            std::vector<float> probs;
            Tensor loss = softmax_cross_entropy(logits, test.labels[i], &probs);
            test_running_loss += loss->data[0];
            if (argmax(probs) == test.labels[i]) ++test_correct;
        }
        double test_loss = test_running_loss / test.size();
        double test_acc = static_cast<double>(test_correct) / test.size();

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Epoca " << epoch << "/" << args.epochs
                  << " | train_loss=" << train_loss << " train_acc=" << train_acc
                  << " | test_loss=" << test_loss << " test_acc=" << test_acc
                  << " | " << secs << "s\n";

        // Guardar métricas en CSV
        metrics << epoch << "," << train_loss << "," << train_acc << ","
                << test_loss << "," << test_acc << "," << secs << "\n";
        metrics.flush();

        // Guardar mejor modelo
        if (test_acc > best_test_acc) {
            best_test_acc = test_acc;
            model.save(args.model_out);
            std::cout << "  -> Nuevo mejor modelo guardado en " << args.model_out
                      << " (test_acc=" << test_acc << ")\n";
        }
    }

    std::cout << "Entrenamiento terminado. Mejor test_acc=" << best_test_acc << "\n";
    std::cout << "Metricas guardadas en " << args.metrics_out << "\n";
    return 0;
}

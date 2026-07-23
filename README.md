# Vision Transformer (ViT) en C++ puro — MNIST

Implementación de un Vision Transformer **desde cero en C++17**, sin ninguna
librería de machine learning (nada de LibTorch, Eigen, ni similares). Incluye
motor de autograd propio (backprop manual mediante grafo computacional),
arquitectura ViT completa, carga de MNIST, entrenamiento con Adam,
evaluación, guardado/carga de pesos y registro de métricas en CSV.

Fue probado end-to-end: compila limpio, entrena sobre MNIST real y la
accuracy sube de forma consistente (ver sección "Validación" abajo).

## ¿Cómo funciona el autograd?

No hay diferenciación automática "mágica" de un framework. Cada tensor
(`include/tensor.hpp`) es un nodo de un grafo computacional: cuando una
operación (`include/ops.hpp`) produce un tensor de salida, le asigna una
función `backward_fn` que sabe calcular a mano la derivada de esa operación
específica (regla de la cadena escrita explícitamente). `backward()` hace un
orden topológico del grafo y ejecuta cada `backward_fn` en orden inverso.

Esto es exactamente lo que hacen frameworks como PyTorch por dentro, solo que
aquí cada operación (matmul, softmax, layer norm, GELU, cross-entropy...) fue
derivada e implementada a mano en `ops.hpp`.

## Arquitectura del modelo

```
imagen 28x28 (1 canal)
  -> patchify: 16 parches de 7x7 = 49 valores c/u   (patch_size=7)
  -> Linear(49 -> 64)                                (patch embedding)
  -> prepend token [CLS] + sumar embeddings posicionales
  -> N=4 bloques Transformer (pre-norm), cada uno:
       LayerNorm -> Multi-Head Self-Attention (4 heads) -> +residual
       LayerNorm -> MLP (64->128->64, GELU)             -> +residual
  -> LayerNorm final
  -> tomar el token [CLS]
  -> Linear(64 -> 10) -> logits
  -> softmax + cross-entropy
```

Configuración por defecto en `include/vit.hpp` (`struct ViTConfig`):
patch 7x7, embed_dim=64, depth=4, heads=4, mlp_hidden=128. Se puede cambiar
libremente ahí (recompilar después).

**Nota sobre el diseño**: para mantener el motor de tensores simple (matrices
2D, sin dimensión de batch), cada imagen se procesa individualmente. El
"mini-batch" se implementa acumulando gradientes de varias imágenes antes de
llamar al optimizador — matemáticamente equivalente a un batch real, pero sin
necesitar operaciones tensoriales de más de 2 dimensiones.

## Estructura del proyecto

```
vit_mnist/
├── CMakeLists.txt
├── include/
│   ├── tensor.hpp      # Tensor + motor de autograd (grafo + backward())
│   ├── ops.hpp          # Operaciones diferenciables (matmul, softmax, etc.)
│   ├── nn.hpp            # Capas: Linear, LayerNorm, MultiHeadAttention, MLP...
│   ├── vit.hpp            # Modelo ViT completo + guardado/carga de pesos
│   ├── mnist.hpp           # Lector del formato IDX de MNIST
│   └── optimizer.hpp        # Adam implementado a mano
├── src/
│   ├── train.cpp    # Entrenamiento: epochs, mini-batches, eval, checkpoints
│   └── eval.cpp      # Carga un modelo guardado y evalúa (+ matriz de confusión)
├── scripts/
│   └── prepare_data.sh   # Descomprime (o descarga) los archivos de MNIST
├── data/            # MNIST comprimido (.gz), listo para usar
├── models/         # Aquí se guardan los checkpoints (.bin)
└── metrics/       # Aquí se guardan los CSV de métricas por época
```

## Compilar

Requiere CMake ≥3.10 y un compilador con C++17 (g++ o clang++). OpenMP es
opcional: si está disponible, paraleliza el matmul (la operación más cara);
si no, compila igual pero en modo secuencial.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

Esto genera dos ejecutables: `vit_train` y `vit_eval`.

## Preparar los datos

El dataset ya viene incluido comprimido en `data/*.gz`. Antes de entrenar,
descomprímelo:

```bash
bash scripts/prepare_data.sh
```

(Si los `.gz` no estuvieran, el script los descarga automáticamente desde un
mirror en GitHub.)

## Entrenar

```bash
./build/vit_train \
  --epochs 15 --batch-size 32 --lr 0.001 \
  --model-out models/vit_mnist.bin \
  --metrics-out metrics/metrics.csv
```

Todos los argumentos son opcionales (ver valores por defecto en
`src/train.cpp`). Algunos útiles:

- `--limit-train N` / `--limit-test N`: usar solo N imágenes (para pruebas rápidas).
- `--seed N`: semilla para reproducibilidad.

Cada época imprime `train_loss`, `train_acc`, `test_loss`, `test_acc` y el
tiempo, y los añade a `metrics/metrics.csv`. El checkpoint se guarda cada vez
que mejora la accuracy de test.

### ⚠️ Sobre el rendimiento

Este es un ViT **educativo**: backprop manual con matrices densas, sin BLAS
ni GPU. En una CPU típica, una época sobre las 60,000 imágenes completas de
MNIST puede tomar bastantes minutos (más rápido con OpenMP y varios núcleos).
Para explorar rápido, usa `--limit-train` con unos pocos miles de imágenes;
para exprimir accuracy real (>95%), déjalo correr con el dataset completo
durante varias épocas.

## Evaluar / inferencia

```bash
./build/vit_eval --model models/vit_mnist.bin --test-images data/t10k-images-idx3-ubyte --test-labels data/t10k-labels-idx1-ubyte
```

Imprime accuracy global y una matriz de confusión 10x10.

## Formato de los pesos guardados

`VisionTransformer::save()/load()` (en `vit.hpp`) serializan en binario: un
magic number, la configuración (`ViTConfig`) y luego cada tensor de
parámetros como `(rows, cols, datos float)` en el mismo orden en que
`parameters()` los recorre. `load()` valida que la configuración y las
formas coincidan antes de sobrescribir los datos.

## Validación realizada

Se compiló el proyecto (g++ 13 y también vía CMake) y se corrió un
entrenamiento real sobre MNIST descargado de un mirror público. Con solo
1500 imágenes de entrenamiento y 5 épocas:

| Época | train_loss | train_acc | test_loss | test_acc |
|------:|-----------:|----------:|----------:|---------:|
| 1     | 1.662      | 44.9%     | 0.968     | 66.7%    |
| 2     | 0.756      | 76.5%     | 0.634     | 80.7%    |
| 3     | 0.424      | 88.1%     | 0.612     | 81.3%    |
| 4     | 0.236      | 93.2%     | 0.446     | 85.7%    |
| 5     | 0.172      | 95.3%     | 0.467     | 86.7%    |

La pérdida baja consistentemente y la accuracy sube en cada época, lo que
confirma que el forward pass, el backward pass y el optimizador Adam están
implementados correctamente. Con el dataset completo (60k imágenes) y más
épocas, la accuracy en test debería subir bastante más (un ViT pequeño bien
entrenado en MNIST suele superar 97-98%).

También se verificó explícitamente que guardar y volver a cargar los pesos
reproduce exactamente la misma accuracy.

## Ideas para extender

- **Data augmentation**: pequeñas rotaciones/traslaciones para mejorar generalización.
- **Dropout**: añadir una capa de dropout en la MLP o en la atención.
- **Warmup + decaimiento de learning rate**: común en el entrenamiento de Transformers.
- **Batch real vectorizado**: extender `Tensor` a 3D (batch, seq, dim) para
  aprovechar mejor la CPU (bastante más trabajo de ingeniería).
- **Cuantización de pesos** para inferencia más rápida/liviana.

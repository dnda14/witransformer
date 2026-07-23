#!/usr/bin/env bash
# prepare_data.sh — deja los 4 archivos IDX de MNIST listos en data/.
# El repo ya incluye los .gz; este script solo los descomprime.
# Si por alguna razón no están, intenta descargarlos de un mirror en GitHub.
set -e
cd "$(dirname "$0")/.."

FILES="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"
MIRROR="https://raw.githubusercontent.com/fgnt/mnist/master"

mkdir -p data
for f in $FILES; do
    if [ -f "data/$f" ]; then
        echo "OK: data/$f ya existe"
        continue
    fi
    if [ -f "data/$f.gz" ]; then
        echo "Descomprimiendo data/$f.gz..."
        gunzip -k "data/$f.gz"
        continue
    fi
    echo "Descargando $f.gz..."
    curl -sL "$MIRROR/$f.gz" -o "data/$f.gz"
    gunzip -k "data/$f.gz"
done

echo "Listo. Archivos en data/:"
ls -la data/*idx*

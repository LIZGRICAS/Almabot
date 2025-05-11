#!/bin/bash
set -e

# Si el directorio de modelos está vacío, copiamos los modelos de ejemplo
if [ -z "$(ls -A /app/data/models)" ]; then
    echo "No se encontraron modelos. Copiando modelos de ejemplo..."
    mkdir -p /app/data/models
    cp -r /app/data/example_models/* /app/data/models/
    chown -R appuser:appuser /app/data/models
fi

# Verificar que el modelo existe
if [ ! -f "/app/data/models/bullying_detection_model.joblib" ]; then
    echo "ERROR: No se encontró el modelo entrenado. Por favor, ejecute el entrenamiento primero."
    echo "Puede ejecutar el entrenamiento con: docker compose --profile train up training"
    exit 1
fi

# Iniciar la API
echo "Iniciando la API..."
exec "$@"

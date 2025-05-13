#!/bin/bash

# Script para inicializar Ollama y descargar el modelo Mistral 7B
# Este script se ejecutará después de que el contenedor de Ollama esté en funcionamiento

echo "=== Inicializando Ollama ==="

# Esperar a que el servicio Ollama esté disponible
echo "Esperando a que el servicio Ollama esté disponible..."
MAX_RETRIES=60
RETRY_COUNT=0

# Función para verificar si Ollama está disponible
check_ollama() {
    curl -s -m 5 http://ollama_8001:11434/ > /dev/null
    return $?
}

# Esperar a que Ollama esté disponible
while ! check_ollama; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ADVERTENCIA: No se pudo conectar al servicio Ollama después de $MAX_RETRIES intentos."
        echo "Continuando de todos modos, pero es posible que el modelo no se descargue correctamente."
        break
    fi
    echo "Esperando a que Ollama esté disponible... Intento $RETRY_COUNT de $MAX_RETRIES"
    sleep 5
done

echo "Intentando descargar el modelo Mistral 7B..."

# Intentar descargar el modelo Mistral 7B con reintentos
MAX_PULL_RETRIES=3
PULL_RETRY_COUNT=0
PULL_SUCCESS=false

while [ $PULL_RETRY_COUNT -lt $MAX_PULL_RETRIES ] && [ "$PULL_SUCCESS" = "false" ]; do
    PULL_RETRY_COUNT=$((PULL_RETRY_COUNT+1))
    echo "Intento $PULL_RETRY_COUNT de $MAX_PULL_RETRIES para descargar el modelo..."
    
    # Intentar descargar el modelo
    RESPONSE=$(curl -s -X POST http://ollama_8001:11434/api/pull -d '{"name": "mistral:7b-instruct"}' || echo '{"error": "connection failed"}')
    
    # Verificar si la respuesta contiene un error
    if echo "$RESPONSE" | grep -q "error"; then
        ERROR=$(echo "$RESPONSE" | grep -o '"error":"[^"]*"' | cut -d '"' -f 4)
        echo "Error al descargar el modelo: $ERROR"
        echo "Esperando antes de reintentar..."
        sleep 10
    else
        echo "Modelo Mistral 7B descargado correctamente o descarga iniciada."
        PULL_SUCCESS=true
        break
    fi
done

if [ "$PULL_SUCCESS" = "true" ]; then
    echo "=== Inicialización de Ollama completada con éxito ==="
    exit 0
else
    echo "=== Inicialización de Ollama completada con advertencias ==="
    echo "No se pudo descargar el modelo, pero la API debería estar disponible."
    # Salimos con código 0 para que el contenedor no falle
    exit 0
fi

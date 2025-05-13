#!/bin/bash

# Script para iniciar el sistema AlmaBot completo
# Este script inicia todos los servicios necesarios y realiza pruebas básicas

echo "=== Iniciando AlmaBot con integración de Ollama ==="

# Verificar si Docker está instalado y en ejecución
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker y/o docker-compose no están instalados."
    echo "Por favor, instale Docker y docker-compose antes de continuar."
    exit 1
fi

# Verificar si Docker está en ejecución
if ! docker info &> /dev/null; then
    echo "Error: El servicio Docker no está en ejecución."
    echo "Por favor, inicie el servicio Docker antes de continuar."
    exit 1
fi

# Función para limpiar al salir
cleanup() {
    echo -e "\n\nDeteniendo servicios..."
    docker-compose -f docker-compose.8001.yml down
    echo "Servicios detenidos."
    exit 0
}

# Capturar señales para limpiar al salir
trap cleanup SIGINT SIGTERM

# Iniciar los servicios
echo "Iniciando servicios con docker-compose..."
docker-compose -f docker-compose.8001.yml up -d

# Esperar a que los servicios estén disponibles
echo "Esperando a que los servicios estén disponibles..."
MAX_RETRIES=30
RETRY_COUNT=0

while ! curl -s http://localhost:8001/health &> /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Error: No se pudo conectar a la API después de $MAX_RETRIES intentos."
        echo "Verificando logs de los contenedores..."
        docker-compose -f docker-compose.8001.yml logs --tail=50 api_8001
        echo "Deteniendo servicios..."
        docker-compose -f docker-compose.8001.yml down
        exit 1
    fi
    echo "Esperando a que la API esté disponible... Intento $RETRY_COUNT de $MAX_RETRIES"
    sleep 10
done

# Verificar el estado de salud de la API
echo "Verificando el estado de salud de la API..."
HEALTH_STATUS=$(curl -s http://localhost:8001/health)
echo "Estado de salud: $HEALTH_STATUS"

# Ejecutar prueba del chatbot
echo "Ejecutando prueba del chatbot mejorado..."
docker exec almabot-api-8001 python /app/test_enhanced_chatbot.py

# Mostrar información sobre cómo acceder a la API
echo -e "\n=== AlmaBot está en ejecución ==="
echo "API disponible en: http://localhost:8001"
echo "Endpoint de salud: http://localhost:8001/health"
echo "Ollama disponible en: http://localhost:11434"
echo -e "\nPresione Ctrl+C para detener los servicios."

# Mantener el script en ejecución para que el usuario pueda ver los logs
docker-compose -f docker-compose.8001.yml logs -f

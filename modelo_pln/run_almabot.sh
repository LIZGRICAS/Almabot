#!/bin/bash

# Script para iniciar AlmaBot con opciones de entrenamiento
# Uso: ./run_almabot.sh [--retrain]
#   --retrain: Fuerza el reentrenamiento del modelo aunque ya exista

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Verificar si Docker está instalado y en ejecución
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker no está instalado.${NC}"
    echo "Por favor, instale Docker antes de continuar."
    exit 1
fi

# Verificar si el comando docker compose funciona (Docker CLI Plugin)
if ! docker compose version &> /dev/null; then
    echo -e "${YELLOW}Advertencia: El comando 'docker compose' no está disponible.${NC}"
    echo "Se intentará usar 'docker-compose' como alternativa."
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Error: Ni 'docker compose' ni 'docker-compose' están disponibles.${NC}"
        echo "Por favor, actualice Docker o instale docker-compose antes de continuar."
        exit 1
    fi
    
    # Si llegamos aquí, docker-compose está disponible
    DOCKER_COMPOSE_CMD="docker-compose"
else
    # Usar el nuevo formato de comando
    DOCKER_COMPOSE_CMD="docker compose"
fi

# Verificar si Docker está en ejecución
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: El servicio Docker no está en ejecución.${NC}"
    echo "Por favor, inicie el servicio Docker antes de continuar."
    exit 1
fi

# Configuración por defecto
FORCE_TRAINING="false"
PROJECT_NAME="almabot_8001"

# Procesar argumentos
for arg in "$@"; do
    case $arg in
        --retrain)
            FORCE_TRAINING="true"
            shift
            ;;
        --help)
            echo -e "${BLUE}Uso: ./run_almabot.sh [--retrain]${NC}"
            echo "  --retrain: Fuerza el reentrenamiento del modelo aunque ya exista"
            exit 0
            ;;
        *)
            echo -e "${RED}Argumento desconocido: $arg${NC}"
            echo -e "Use ${BLUE}--help${NC} para ver las opciones disponibles."
            exit 1
            ;;
    esac
done

# Función para limpiar al salir
cleanup() {
    echo -e "\n\n${YELLOW}Deteniendo servicios...${NC}"
    FORCE_TRAINING=$FORCE_TRAINING $DOCKER_COMPOSE_CMD -f docker-compose.8001.yml -p $PROJECT_NAME down
    echo -e "${GREEN}Servicios detenidos.${NC}"
    exit 0
}

# Capturar señales para limpiar al salir
trap cleanup SIGINT SIGTERM

# Mostrar configuración
echo -e "${BLUE}=== Configuración de AlmaBot ===${NC}"
echo -e "Reentrenar modelo: ${FORCE_TRAINING}"

# Iniciar los servicios
echo -e "\n${BLUE}Iniciando servicios con $DOCKER_COMPOSE_CMD...${NC}"
FORCE_TRAINING=$FORCE_TRAINING $DOCKER_COMPOSE_CMD -f docker-compose.8001.yml -p $PROJECT_NAME up

# Esperar a que los servicios estén disponibles
echo -e "\n${BLUE}Esperando a que los servicios estén disponibles...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while ! curl -s http://localhost:8001/health &> /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}Error: No se pudo conectar a la API después de $MAX_RETRIES intentos.${NC}"
        echo "Verificando logs de los contenedores..."
        $DOCKER_COMPOSE_CMD -f docker-compose.8001.yml -p $PROJECT_NAME logs --tail=50 api_8001
        echo -e "${YELLOW}Deteniendo servicios...${NC}"
        FORCE_TRAINING=$FORCE_TRAINING $DOCKER_COMPOSE_CMD -f docker-compose.8001.yml -p $PROJECT_NAME down
        exit 1
    fi
    echo "Esperando a que la API esté disponible... Intento $RETRY_COUNT de $MAX_RETRIES"
    sleep 5
done

# Verificar el estado de salud de la API
echo -e "\n${BLUE}Verificando el estado de salud de la API...${NC}"
HEALTH_STATUS=$(curl -s http://localhost:8001/health)
echo "Estado de salud: $HEALTH_STATUS"

# Mostrar información sobre cómo acceder a la API
echo -e "\n${GREEN}=== AlmaBot está en ejecución ===${NC}"
echo -e "${BLUE}API disponible en:${NC} http://localhost:8001"
echo -e "${BLUE}Endpoint de salud:${NC} http://localhost:8001/health"
echo -e "${BLUE}Ollama disponible en:${NC} http://localhost:11434"
echo -e "\n${YELLOW}Presione Ctrl+C para detener los servicios.${NC}"

# Mantener el script en ejecución para que el usuario pueda ver los logs
FORCE_TRAINING=$FORCE_TRAINING $DOCKER_COMPOSE_CMD -f docker-compose.8001.yml -p $PROJECT_NAME logs -f

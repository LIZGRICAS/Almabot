#!/bin/bash

# Script para instalar y configurar Ollama con el modelo Mistral 7B
# para el chatbot AlmaBot

echo "=== Configurando Ollama para AlmaBot ==="
echo "Este script instalará Ollama y descargará el modelo Mistral 7B"

# Verificar si Ollama ya está instalado
if command -v ollama &> /dev/null; then
    echo "Ollama ya está instalado"
else
    echo "Instalando Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Verificar si la instalación fue exitosa
    if [ $? -ne 0 ]; then
        echo "Error: No se pudo instalar Ollama"
        exit 1
    fi
    
    echo "Ollama instalado correctamente"
fi

# Iniciar el servicio Ollama
echo "Iniciando el servicio Ollama..."
ollama serve &
sleep 5  # Esperar a que el servicio inicie

# Descargar el modelo Mistral 7B
echo "Descargando el modelo Mistral 7B (esto puede tardar varios minutos)..."
ollama pull mistral:7b-instruct

# Verificar si la descarga fue exitosa
if [ $? -ne 0 ]; then
    echo "Error: No se pudo descargar el modelo Mistral 7B"
    exit 1
fi

echo "=== Configuración completada ==="
echo "Modelo Mistral 7B instalado y listo para usar con AlmaBot"
echo "Para iniciar el servicio Ollama en el futuro, ejecuta: ollama serve"
echo "Para verificar los modelos instalados, ejecuta: ollama list"

# Instrucciones para actualizar requirements.txt
echo "=== Actualizando dependencias ==="
echo "Agregando dependencias necesarias a requirements.txt..."

# Verificar si requests ya está en requirements.txt
if grep -q "requests" /home/sublime-dev/dev/python/Almabot/modelo_pln/requirements.txt; then
    echo "La dependencia 'requests' ya está en requirements.txt"
else
    echo "requests>=2.28.0" >> /home/sublime-dev/dev/python/Almabot/modelo_pln/requirements.txt
    echo "Dependencia 'requests' agregada a requirements.txt"
fi

echo "=== Instalando dependencias ==="
pip install -r /home/sublime-dev/dev/python/Almabot/modelo_pln/requirements.txt

echo "=== Todo listo ==="
echo "Ahora puedes iniciar la API con: python -m src.api.api"

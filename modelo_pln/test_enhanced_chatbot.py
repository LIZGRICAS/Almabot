#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para probar el EnhancedChatbot con la integración de Ollama.
Este script permite verificar que la integración entre el modelo de detección de bullying
y el modelo de lenguaje Mistral 7B a través de Ollama funciona correctamente.
"""

import os
import sys
import json
import time
import random
import requests
from typing import Dict, Any, List, Optional

# Añadir el directorio src al path para poder importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importar los módulos necesarios
from bot.enhanced_chatbot import EnhancedChatbot
from bot.llm_processor import LLMProcessor
from bot.conversation_manager import ConversationManager

# Configuración para el test
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'localhost')
OLLAMA_PORT = os.environ.get('OLLAMA_PORT', '11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'mistral:7b-instruct')

# Mensajes de prueba
TEST_MESSAGES = [
    # Mensajes normales
    "Hola, ¿cómo estás?",
    "¿Puedes ayudarme con mi tarea de matemáticas?",
    "Me siento un poco triste hoy",
    "¿Qué puedo hacer si me siento solo?",
    
    # Mensajes que podrían indicar bullying
    "Mis compañeros me insultan todos los días",
    "Un niño de mi clase me empujó y me quitó mi mochila",
    "Me dicen que soy feo y que nadie me quiere",
    "Me amenazan con golpearme si no les doy mi dinero"
]

def check_ollama_availability() -> bool:
    """Verifica si el servicio de Ollama está disponible."""
    try:
        response = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/")
        return response.status_code == 200
    except requests.RequestException:
        return False

def wait_for_ollama(max_retries: int = 10, retry_interval: int = 5) -> bool:
    """Espera a que el servicio de Ollama esté disponible."""
    print(f"Verificando disponibilidad de Ollama en {OLLAMA_HOST}:{OLLAMA_PORT}...")
    
    for i in range(max_retries):
        if check_ollama_availability():
            print("Ollama está disponible.")
            return True
        
        print(f"Esperando a que Ollama esté disponible... Intento {i+1}/{max_retries}")
        time.sleep(retry_interval)
    
    print("Error: No se pudo conectar a Ollama después de varios intentos.")
    return False

def test_enhanced_chatbot():
    """Prueba el funcionamiento del EnhancedChatbot."""
    print("\n=== Iniciando prueba de EnhancedChatbot ===\n")
    
    # Crear instancias de los componentes necesarios
    conversation_manager = ConversationManager()
    llm_processor = LLMProcessor(
        ollama_host=OLLAMA_HOST,
        ollama_port=OLLAMA_PORT,
        model_name=OLLAMA_MODEL
    )
    
    # Crear instancia del chatbot mejorado
    chatbot = EnhancedChatbot(
        conversation_manager=conversation_manager,
        llm_processor=llm_processor
    )
    
    # Información de usuario de prueba
    user_id = "test_user_123"
    user_info = {
        "name": "Test User",
        "age": 12,
        "gender": "No especificado",
        "school_grade": "6to grado"
    }
    
    # Probar cada mensaje
    for i, message in enumerate(TEST_MESSAGES, 1):
        print(f"\n--- Prueba {i}/{len(TEST_MESSAGES)} ---")
        print(f"Usuario: {message}")
        
        # Procesar el mensaje con el chatbot
        try:
            start_time = time.time()
            response, metadata = chatbot.process_message(user_id, message, user_info)
            end_time = time.time()
            
            print(f"AlmaBot: {response}")
            print(f"Tiempo de respuesta: {end_time - start_time:.2f} segundos")
            print(f"Metadatos: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
            
            # Esperar un poco entre mensajes para no sobrecargar el servicio
            time.sleep(1)
        except Exception as e:
            print(f"Error al procesar el mensaje: {str(e)}")
    
    print("\n=== Prueba de EnhancedChatbot completada ===\n")

def main():
    """Función principal."""
    print("=== Test de integración de Ollama con AlmaBot ===")
    
    # Verificar disponibilidad de Ollama
    if not wait_for_ollama():
        print("No se puede continuar sin Ollama disponible.")
        return
    
    # Ejecutar la prueba del chatbot mejorado
    test_enhanced_chatbot()

if __name__ == "__main__":
    main()

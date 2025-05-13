#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para monitorear el estado de Ollama y verificar que esté funcionando correctamente.
Este script puede ejecutarse periódicamente para asegurar que el servicio de Ollama
está disponible y que el modelo Mistral 7B está cargado correctamente.
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configuración por defecto
DEFAULT_OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'localhost')
DEFAULT_OLLAMA_PORT = os.environ.get('OLLAMA_PORT', '11434')
DEFAULT_OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'mistral:7b-instruct')

def get_timestamp() -> str:
    """Obtiene una marca de tiempo formateada."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def check_ollama_health(host: str, port: str) -> Dict[str, Any]:
    """Verifica el estado de salud de Ollama."""
    result = {
        "timestamp": get_timestamp(),
        "status": "error",
        "message": "No se pudo conectar a Ollama",
        "details": {}
    }
    
    try:
        # Verificar que el servicio esté disponible
        response = requests.get(f"http://{host}:{port}/", timeout=5)
        if response.status_code != 200:
            result["message"] = f"Ollama respondió con código de estado {response.status_code}"
            return result
        
        # Verificar los modelos disponibles
        models_response = requests.get(f"http://{host}:{port}/api/tags", timeout=10)
        if models_response.status_code != 200:
            result["status"] = "warning"
            result["message"] = "Ollama está disponible pero no se pudieron obtener los modelos"
            return result
        
        # Analizar la respuesta
        models_data = models_response.json()
        result["details"]["models"] = models_data.get("models", [])
        
        # Verificar si hay modelos disponibles
        if not result["details"]["models"]:
            result["status"] = "warning"
            result["message"] = "Ollama está disponible pero no hay modelos cargados"
            return result
        
        # Todo está bien
        result["status"] = "ok"
        result["message"] = "Ollama está funcionando correctamente"
        return result
        
    except requests.RequestException as e:
        result["details"]["error"] = str(e)
        return result

def test_model_generation(host: str, port: str, model: str) -> Dict[str, Any]:
    """Prueba la generación de texto con el modelo especificado."""
    result = {
        "timestamp": get_timestamp(),
        "status": "error",
        "message": "No se pudo probar la generación de texto",
        "details": {
            "model": model
        }
    }
    
    try:
        # Datos para la solicitud
        data = {
            "model": model,
            "prompt": "Hola, ¿cómo estás?",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 50
            }
        }
        
        # Realizar la solicitud
        start_time = time.time()
        response = requests.post(
            f"http://{host}:{port}/api/generate",
            json=data,
            timeout=30
        )
        end_time = time.time()
        
        # Verificar la respuesta
        if response.status_code != 200:
            result["message"] = f"Error al generar texto: código {response.status_code}"
            return result
        
        # Analizar la respuesta
        response_data = response.json()
        result["details"]["response"] = response_data.get("response", "")
        result["details"]["time_taken"] = round(end_time - start_time, 2)
        
        # Todo está bien
        result["status"] = "ok"
        result["message"] = "Generación de texto exitosa"
        return result
        
    except requests.RequestException as e:
        result["details"]["error"] = str(e)
        return result

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Monitor de estado para Ollama")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Host de Ollama")
    parser.add_argument("--port", default=DEFAULT_OLLAMA_PORT, help="Puerto de Ollama")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Modelo a probar")
    parser.add_argument("--test-generation", action="store_true", help="Probar la generación de texto")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON")
    args = parser.parse_args()
    
    # Verificar el estado de salud
    health_result = check_ollama_health(args.host, args.port)
    
    # Probar la generación de texto si se solicita
    generation_result = None
    if args.test_generation and health_result["status"] in ["ok", "warning"]:
        generation_result = test_model_generation(args.host, args.port, args.model)
    
    # Preparar el resultado
    result = {
        "timestamp": get_timestamp(),
        "health": health_result,
        "generation": generation_result
    }
    
    # Mostrar el resultado
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"=== Monitor de Ollama ({result['timestamp']}) ===")
        print(f"Estado: {health_result['status'].upper()}")
        print(f"Mensaje: {health_result['message']}")
        
        if health_result["details"].get("models"):
            print("\nModelos disponibles:")
            for model in health_result["details"]["models"]:
                print(f"- {model.get('name', 'Desconocido')}")
        
        if generation_result:
            print(f"\nPrueba de generación ({args.model}):")
            print(f"Estado: {generation_result['status'].upper()}")
            print(f"Mensaje: {generation_result['message']}")
            print(f"Tiempo: {generation_result['details'].get('time_taken', 'N/A')} segundos")
            if generation_result["details"].get("response"):
                print(f"Respuesta: {generation_result['details']['response'][:100]}...")
    
    # Establecer el código de salida
    if health_result["status"] == "error" or (generation_result and generation_result["status"] == "error"):
        sys.exit(1)
    elif health_result["status"] == "warning" or (generation_result and generation_result["status"] == "warning"):
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

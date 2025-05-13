"""
LLM Processor Module

This module provides functionality for generating natural language responses
using Ollama with Mistral 7B or Llama 2 models.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    Processes natural language using Ollama with local LLM models.
    Generates contextual and empathetic responses for the chatbot.
    """
    
    def __init__(self, model_name: str = None, ollama_host: str = None, ollama_port: str = None):
        """
        Initialize the LLM processor.
        
        Args:
            model_name: The name of the Ollama model to use
            ollama_host: The hostname of the Ollama service
            ollama_port: The port of the Ollama service
        """
        # Use environment variables if parameters are not provided
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
        host = ollama_host or os.getenv("OLLAMA_HOST", "localhost")
        port = ollama_port or os.getenv("OLLAMA_PORT", "11434")
        
        self.ollama_url = f"http://{host}:{port}"
        logger.info(f"Initialized LLM Processor with model: {self.model_name} at {self.ollama_url}")
        
    def generate_response(self, conversation_history: List[Dict[str, str]], 
                         user_info: Optional[Dict[str, Any]] = None,
                         is_bullying: bool = False,
                         bullying_type: str = None) -> str:
        """
        Generate a response using the LLM based on conversation history.
        
        Args:
            conversation_history: List of previous messages in the conversation
            user_info: Optional information about the user
            is_bullying: Whether the current message is detected as bullying
            bullying_type: The type of bullying detected (if any)
            
        Returns:
            Generated response text
        """
        # Format conversation history for the LLM
        messages = []
        
        # Add system prompt with instructions
        system_prompt = self._create_system_prompt(is_bullying, bullying_type, user_info)
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for message in conversation_history:
            messages.append({
                "role": message.get("role", "user"),
                "content": message.get("content", "")
            })
            
        # Call Ollama API
        try:
            logger.info(f"Calling Ollama API with {len(messages)} messages")
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Received response from Ollama API")
            return result["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return self._get_fallback_response(is_bullying, bullying_type)
    
    def _create_system_prompt(self, is_bullying: bool, bullying_type: str, user_info: Optional[Dict[str, Any]]) -> str:
        """
        Create a system prompt based on the current context.
        
        Args:
            is_bullying: Whether the current message is detected as bullying
            bullying_type: The type of bullying detected (if any)
            user_info: Optional information about the user
            
        Returns:
            System prompt text
        """
        base_prompt = (
            "Eres AlmaBot, un asistente virtual amigable diseñado para niños de 6 a 14 años. "
            "Tu objetivo es proporcionar apoyo emocional, especialmente en situaciones de bullying. "
            "Usa un lenguaje sencillo y apropiado para niños. Sé empático, paciente y comprensivo. "
            "Nunca minimices sus problemas y ofrece consejos prácticos y seguros. "
            "Si detectas una situación de riesgo, anima al niño a hablar con un adulto de confianza. "
            "Mantén tus respuestas breves y claras, adecuadas para niños."
        )
        
        # Add bullying-specific instructions if bullying is detected
        if is_bullying:
            bullying_prompt = (
                f"\n\nEl mensaje actual ha sido detectado como posible bullying de tipo '{bullying_type}'. "
                "Responde con empatía y ofrece apoyo emocional. "
                "Sugiere estrategias apropiadas para manejar este tipo específico de bullying. "
                "Recuerda que la seguridad del niño es la prioridad. "
                "Anima al niño a hablar con un adulto de confianza sobre la situación."
            )
            base_prompt += bullying_prompt
        
        # Add user-specific instructions if available
        if user_info:
            age = user_info.get("age")
            if age:
                age_prompt = f"\n\nEl usuario tiene aproximadamente {age} años. Adapta tu lenguaje a esta edad."
                base_prompt += age_prompt
        
        return base_prompt
    
    def _get_fallback_response(self, is_bullying: bool, bullying_type: str) -> str:
        """
        Get a fallback response in case the LLM call fails.
        
        Args:
            is_bullying: Whether the current message is detected as bullying
            bullying_type: The type of bullying detected (if any)
            
        Returns:
            Fallback response text
        """
        if is_bullying:
            return (
                "Entiendo que estás pasando por una situación difícil. "
                "Es importante que hables con un adulto de confianza sobre esto. "
                "¿Hay alguien con quien te sientas cómodo hablando? "
                "Estoy aquí para escucharte y ayudarte."
            )
        else:
            return (
                "Gracias por compartir eso conmigo. "
                "¿Hay algo más de lo que te gustaría hablar? "
                "Estoy aquí para escucharte."
            )
    
    def check_ollama_status(self) -> bool:
        """
        Check if Ollama is running and the model is available.
        
        Returns:
            True if Ollama is running and the model is available, False otherwise
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            
            # Check if our model is in the list
            for model in models:
                if model.get("name", "").startswith(self.model_name.split(":")[0]):
                    return True
            
            logger.warning(f"Model {self.model_name} not found in Ollama")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking Ollama status: {e}")
            return False

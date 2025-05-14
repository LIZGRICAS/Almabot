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
        
        # Verify connection to Ollama at initialization
        self._verify_ollama_connection()
        
    def generate_response(self, conversation_history: List[Dict[str, str]], 
                         user_info: Optional[Dict[str, Any]] = None,
                         is_bullying: bool = False,
                         bullying_type: str = None,
                         current_message: str = None) -> str:
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
        system_prompt = self._create_system_prompt(is_bullying, bullying_type, user_info, current_message)
        messages.append({"role": "system", "content": system_prompt})
        
        # Log the system prompt for debugging
        logger.info(f"System prompt: {system_prompt[:200]}...")
        
        # Log the conversation history for debugging
        logger.info(f"Conversation history length: {len(conversation_history)}")
        if conversation_history:
            last_message = conversation_history[-1] if conversation_history else None
            logger.info(f"Last message in history: {last_message}")
        
        # Add conversation history
        for message in conversation_history:
            messages.append({
                "role": message.get("role", "user"),
                "content": message.get("content", "")
            })
            
        # Call Ollama API
        try:
            logger.info(f"Calling Ollama API with {len(messages)} messages")
            # Log the URL being called for debugging
            logger.info(f"Calling Ollama API at URL: {self.ollama_url}/api/chat")
            
            # Increase timeout to 180 seconds and add retries
            max_retries = 3
            current_retry = 0
            backoff_factor = 2  # Exponential backoff
            
            while current_retry < max_retries:
                try:
                    logger.info(f"Attempt {current_retry + 1}/{max_retries} to call Ollama API")
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
                        timeout=180  # Increased timeout to 3 minutes
                    )
                    response.raise_for_status()
                    break  # Exit the retry loop if successful
                except requests.exceptions.RequestException as e:
                    current_retry += 1
                    if current_retry >= max_retries:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    wait_time = backoff_factor ** current_retry
                    logger.warning(f"Retrying in {wait_time} seconds after error: {e}")
                    import time
                    time.sleep(wait_time)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Received response from Ollama API")
            return result["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return self._get_fallback_response(is_bullying, bullying_type, current_message)
    
    def _create_system_prompt(self, is_bullying: bool, bullying_type: str, user_info: Optional[Dict[str, Any]], current_message: str = None) -> str:
        """
        Create a system prompt based on the current context.
        
        Args:
            is_bullying: Whether the current message is detected as bullying
            bullying_type: The type of bullying detected (if any)
            user_info: Optional information about the user
            current_message: The current message from the user
            
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
        
        # Add information about the current message if available
        if current_message:
            message_context = f"\n\nEl mensaje actual del usuario es: '{current_message}'"
            base_prompt += message_context
        
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
        else:
            # Determine the type of message and add specific instructions
            if current_message:
                message_lower = current_message.lower()
                
                # Check if it's a greeting
                if any(greeting in message_lower for greeting in ["hola", "buenos días", "buenas tardes", "buenas noches", "saludos", "hey", "hi"]):
                    greeting_prompt = (
                        "\n\nEste es un saludo. Responde de manera amigable y entusiasta. "
                        "Haz una pregunta abierta sobre cómo se siente o qué ha hecho hoy. "
                        "Sé conversacional y natural, como un amigo que está feliz de hablar con ellos."
                    )
                    base_prompt += greeting_prompt
                    
                # Check if it's about school or achievements
                elif any(word in message_lower for word in ["escuela", "colegio", "clase", "examen", "nota", "matemáticas", "ciencias"]):
                    school_prompt = (
                        "\n\nEste mensaje es sobre la escuela o logros académicos. "
                        "Muestra entusiasmo genuino por sus logros. "
                        "Felícitalo de manera específica por lo que ha compartido. "
                        "Haz preguntas de seguimiento sobre cómo se sintió o qué fue lo que más le gustó. "
                        "Evita respuestas genéricas como 'gracias por compartir'."
                    )
                    base_prompt += school_prompt
                
                # Default instructions for other types of messages
                else:
                    normal_prompt = (
                        "\n\nEste parece ser un mensaje normal, no relacionado con bullying. "
                        "Responde de manera amigable y conversacional. "
                        "Muestra interés genuino en lo que el niño está compartiendo. "
                        "Haz preguntas de seguimiento relevantes para mantener la conversación. "
                        "Evita respuestas genéricas como 'gracias por compartir' cuando no sea apropiado."
                    )
                    base_prompt += normal_prompt
            else:
                # Default instructions if no current message is available
                normal_prompt = (
                    "\n\nEste parece ser un mensaje normal, no relacionado con bullying. "
                    "Responde de manera amigable y conversacional. "
                    "Si es un saludo o una pregunta simple, responde de forma natural y cercana. "
                    "Si el niño comparte información sobre su día o sus actividades, muestra interés genuino. "
                    "Evita respuestas genéricas como 'gracias por compartir' cuando no sea apropiado."
                )
                base_prompt += normal_prompt
        
        # Add user-specific instructions if available
        if user_info:
            age = user_info.get("age")
            if age:
                age_prompt = f"\n\nEl usuario tiene aproximadamente {age} años. Adapta tu lenguaje a esta edad."
                base_prompt += age_prompt
        
        # Add final instructions for all cases
        base_prompt += (
            "\n\nRecuerda que estás conversando con un niño. Sé natural, amigable y cercano. "
            "Evita respuestas que suenen como plantillas o respuestas genéricas. "
            "Adapta tu respuesta al contexto específico de la conversación."
        )
        
        return base_prompt
    
    def _get_fallback_response(self, is_bullying: bool, bullying_type: str, current_message: str = None) -> str:
        """
        Get a fallback response in case the LLM call fails.
        
        Args:
            is_bullying: Whether the current message is detected as bullying
            bullying_type: The type of bullying detected (if any)
            current_message: The current message from the user
            
        Returns:
            Fallback response text
        """
        # Check if we have a current message to analyze
        if current_message and not is_bullying:
            message_lower = current_message.lower()
            
            # Check if it's a greeting
            if any(greeting in message_lower for greeting in ["hola", "buenos días", "buenas tardes", "buenas noches", "saludos", "hey", "hi"]):
                return (
                    "¡Hola! ¿Cómo estás hoy? Me alegra mucho hablar contigo. "
                    "¿Hay algo especial que te gustaría contarme?"
                )
            
            # Check if it's about school or achievements
            elif any(word in message_lower for word in ["escuela", "colegio", "clase", "examen", "nota", "matemáticas", "ciencias"]):
                return (
                    "¡Qué bueno saber sobre tu experiencia en la escuela! "
                    "Me alegra mucho que compartas eso conmigo. "
                    "¿Cómo te sentiste al respecto? Me encantaría saber más."
                )
            
            # Check if it's about feelings or emotions
            elif any(word in message_lower for word in ["triste", "feliz", "enojado", "contento", "preocupado", "miedo", "nervioso"]):
                return (
                    "Gracias por compartir cómo te sientes. Es importante expresar nuestras emociones. "
                    "¿Hay algo que pueda hacer para ayudarte?"
                )
        
        # Bullying-specific responses
        if is_bullying:
            return (
                "Entiendo que estás pasando por una situación difícil. "
                "Es importante que hables con un adulto de confianza sobre esto. "
                "¿Hay alguien con quien te sientas cómodo hablando? "
                "Estoy aquí para escucharte y ayudarte."
            )
        
        # Default response
        return (
            "Gracias por compartir eso conmigo. "
            "¿Hay algo más de lo que te gustaría hablar? "
            "Estoy aquí para escucharte."
        )
    
    def _verify_ollama_connection(self) -> None:
        """
        Verify the connection to Ollama at initialization and log the result.
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=30)
            response.raise_for_status()
            models = response.json().get("models", [])
            
            if models:
                model_names = [model.get("name", "") for model in models]
                logger.info(f"Successfully connected to Ollama. Available models: {model_names}")
                
                # Check if our model is in the list
                model_prefix = self.model_name.split(":")[0]
                if any(model.startswith(model_prefix) for model in model_names):
                    logger.info(f"Model {self.model_name} is available in Ollama")
                else:
                    logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
            else:
                logger.warning("Connected to Ollama but no models are available")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama at {self.ollama_url}: {e}")
            logger.info("Will attempt to use fallback responses when generating responses")
    
    def check_ollama_status(self) -> bool:
        """
        Check if Ollama is running and the model is available.
        
        Returns:
            True if Ollama is running and the model is available, False otherwise
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=30)
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

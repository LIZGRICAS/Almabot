"""
LLM Processor Module

This module provides functionality for generating natural language responses
using external LLM APIs like OpenAI (ChatGPT) or similar services.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
import logging
import openai
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    Processes natural language using external LLM APIs.
    Generates contextual and empathetic responses for the chatbot.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None, api_type: str = None):
        """
        Initialize the LLM processor.
        
        Args:
            model_name: The name of the LLM model to use
            api_key: The API key for the LLM service
            api_type: The type of API to use (openai, azure, etc.)
        """
        # Use environment variables if parameters are not provided
        self.api_type = api_type or os.getenv("LLM_API_TYPE", "openai")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.api_base = os.getenv("LLM_API_BASE", "")
        
        # Set up the API client
        self._setup_api_client()
        
        logger.info(f"Initialized LLM Processor with model: {self.model_name} using {self.api_type} API")
        
    def _setup_api_client(self):
        """
        Set up the API client based on the API type.
        """
        if not self.api_key:
            logger.warning("No API key provided. LLM API calls will fail.")
            return
            
        if self.api_type.lower() == "openai":
            # Set up OpenAI client
            openai.api_key = self.api_key
            if self.api_base:
                openai.api_base = self.api_base
            logger.info("OpenAI API client configured")
        elif self.api_type.lower() == "azure":
            # Set up Azure OpenAI client
            openai.api_type = "azure"
            openai.api_key = self.api_key
            openai.api_base = self.api_base or "https://your-resource-name.openai.azure.com"
            openai.api_version = os.getenv("AZURE_API_VERSION", "2023-05-15")
            logger.info("Azure OpenAI API client configured")
        elif self.api_type.lower() == "together":
            # Set up Together.ai client using OpenAI's client
            openai.api_key = self.api_key
            openai.api_base = self.api_base or "https://api.together.xyz/v1"
            logger.info("Together.ai API client configured")
        else:
            # Default to OpenAI-compatible API
            openai.api_key = self.api_key
            if self.api_base:
                openai.api_base = self.api_base
            logger.info(f"Using OpenAI-compatible API for {self.api_type}")
            
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
            current_message: The current message from the user
            
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
            
        try:
            if not self.api_key:
                logger.error("No API key provided. Using fallback response.")
                return self._get_fallback_response(is_bullying, bullying_type, current_message)
                
            logger.info(f"Calling {self.api_type} API with {len(messages)} messages")
            
            if self.api_type.lower() == "azure":
                # Call Azure OpenAI API
                response = openai.ChatCompletion.create(
                    deployment_id=self.model_name,  # Azure uses deployment_id instead of model
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                    top_p=0.9,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                logger.info(f"Received response from Azure OpenAI API")
                return response.choices[0].message.content
            else:
                # Call OpenAI API or compatible API (Together.ai, etc.)
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=500,
                        top_p=0.9,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    logger.info(f"Received response from {self.api_type} API")
                    return response.choices[0].message.content
                except Exception as e:
                    # If there's an error with the model name, try with a different format
                    # Some APIs require different model name formats
                    if "model" in str(e).lower():
                        logger.warning(f"Error with model name format, trying alternative format: {e}")
                        try:
                            # Try with just the model name without path
                            model_name_simple = self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name
                            response = openai.ChatCompletion.create(
                                model=model_name_simple,
                                messages=messages,
                                temperature=0.7,
                                max_tokens=500,
                                top_p=0.9,
                                frequency_penalty=0,
                                presence_penalty=0
                            )
                            logger.info(f"Received response from {self.api_type} API using alternative model format")
                            return response.choices[0].message.content
                        except Exception as e2:
                            logger.error(f"Error with alternative model format: {e2}")
                            raise e2
                    else:
                        raise e
                
        except Exception as e:
            logger.error(f"Error calling {self.api_type} API: {e}")
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
    
    def check_api_status(self) -> bool:
        """
        Check if the API is available and configured correctly.
        
        Returns:
            True if the API is available, False otherwise
        """
        if not self.api_key:
            logger.warning("No API key provided. API calls will fail.")
            return False
            
        try:
            if self.api_type.lower() == "azure":
                # Simple test request to Azure OpenAI
                response = openai.ChatCompletion.create(
                    deployment_id=self.model_name,
                    messages=[{"role": "system", "content": "Hello"}],
                    max_tokens=5
                )
                return True
            else:
                # Test request to OpenAI or compatible API (Together.ai, etc.)
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[{"role": "system", "content": "Hello"}],
                        max_tokens=5
                    )
                    logger.info(f"Successfully connected to {self.api_type} API")
                    return True
                except Exception as e:
                    # If there's an error with the model name, try with a different format
                    if "model" in str(e).lower():
                        logger.warning(f"Error with model name format, trying alternative format: {e}")
                        try:
                            # Try with just the model name without path
                            model_name_simple = self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name
                            response = openai.ChatCompletion.create(
                                model=model_name_simple,
                                messages=[{"role": "system", "content": "Hello"}],
                                max_tokens=5
                            )
                            logger.info(f"Successfully connected to {self.api_type} API using alternative model format")
                            return True
                        except Exception as e2:
                            logger.error(f"Error with alternative model format: {e2}")
                            return False
                    else:
                        logger.error(f"Error connecting to {self.api_type} API: {e}")
                        return False
        except Exception as e:
            logger.error(f"Error checking {self.api_type} API status: {e}")
            return False

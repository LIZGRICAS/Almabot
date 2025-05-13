"""
Enhanced Chatbot Module

This module provides the enhanced chatbot implementation that combines
bullying detection with natural language generation using a local LLM.
"""

import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from nlp.nlp_processor import NLPProcessor
from models.logistic_regression_model import BullyingDetectionModel
from bot.conversation_manager import ConversationManager
from bot.llm_processor import LLMProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedChatbot:
    """
    Enhanced chatbot that combines bullying detection with natural language
    generation using a local LLM (Ollama with Mistral 7B or Llama 2).
    """
    
    def __init__(self, db_connection=None, model_name="mistral:7b-instruct"):
        """
        Initialize the enhanced chatbot.
        
        Args:
            db_connection: Database connection object
            model_name: Name of the Ollama model to use
        """
        self.nlp = NLPProcessor()
        self.bullying_model = BullyingDetectionModel()
        try:
            self.bullying_model.load_model()
            logger.info("Bullying detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading bullying detection model: {e}")
        
        self.conversation_manager = ConversationManager(db_connection)
        self.llm_processor = LLMProcessor(model_name=model_name)
        
        # Check if Ollama is available
        if not self.llm_processor.check_ollama_status():
            logger.warning("Ollama is not available. Using fallback responses.")
        
        self.log_file = 'enhanced_chat_logs.jsonl'
        self.logs = []
        self._load_logs()
    
    def _load_logs(self):
        """Load chat logs from file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.logs = [json.loads(line) for line in f if line.strip()]
            else:
                self.logs = []
        except Exception as e:
            logger.error(f"Error loading logs: {e}")
            self.logs = []
    
    def _save_log(self, user_id, message, response, analysis):
        """Save chat log to file"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'message': message,
                'response': response,
                'analysis': analysis
            }
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            self.logs.append(log_entry)
        except Exception as e:
            logger.error(f"Error saving log: {e}")
    
    def process_message(self, user_id: str, message: str, user_info: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Process a message from a user and generate a response.
        
        Args:
            user_id: The ID of the user
            message: The message content
            user_info: Optional information about the user
            
        Returns:
            A tuple containing the response text and analysis dictionary
        """
        try:
            # 1. Run bullying detection for analytics
            bullying_type, is_bullying, confidence = self._analyze_bullying(message)
            
            # 2. Extract keywords for analysis
            keywords = self.nlp.extract_keywords(message)
            
            # 3. Get conversation context (previous messages)
            conversation_history = self.conversation_manager.get_conversation_context(user_id)
            
            # 4. Create a direct message for the current user input to ensure it's included
            current_message = {"role": "user", "content": message}
            
            # 5. Combine previous history with current message
            full_context = conversation_history + [current_message]
            
            # Log the full context for debugging
            logger.info(f"Full conversation context for user {user_id}: {json.dumps(full_context, ensure_ascii=False)}")
            
            # 6. Generate response using LLM with the complete context
            llm_response = self.llm_processor.generate_response(
                full_context,
                user_info=user_info,
                is_bullying=is_bullying,
                bullying_type=bullying_type,
                current_message=message
            )
            
            # 7. Add both user message and response to the conversation history
            self.conversation_manager.add_message(user_id, "user", message)
            
            # 6. Add LLM response to context
            self.conversation_manager.add_message(user_id, "assistant", llm_response)
            
            # 7. Create analysis object for database and dashboard
            analysis = {
                'is_bullying': is_bullying,
                'confidence': confidence,
                'bullying_type': bullying_type,
                'keywords': keywords,
                'sentiment': 'negative' if is_bullying else 'neutral',
                'sentiment_score': 0.3 if is_bullying else 0.0,
                'emotion': {
                    'type': 'fear' if is_bullying else 'neutral',
                    'intensity': 0.7 if is_bullying else 0.0
                },
                'risk': {
                    'level': 'medium' if is_bullying else 'low',
                    'type': 'bullying' if is_bullying else 'none'
                }
            }
            
            # 8. Save log
            self._save_log(user_id, message, llm_response, analysis)
            
            return llm_response, analysis
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            return "Lo siento, ha ocurrido un error al procesar tu mensaje. Por favor, intenta de nuevo.", {}
    
    def _analyze_bullying(self, message: str) -> Tuple[str, bool, float]:
        """
        Analyze a message for bullying content.
        
        Args:
            message: The message to analyze
            
        Returns:
            A tuple containing the bullying type, whether it's bullying, and the confidence score
        """
        try:
            # Initialize variables
            message_lower = message.lower()
            prediction = 0
            confidence = 0.0
            bullying_type = "ninguno"
            
            # Check for common greetings and very short messages first
            common_greetings = ["hola", "buenos días", "buenas tardes", "buenas noches", "saludos", "hey", "hi"]
            common_questions = ["cómo estás", "qué tal", "cómo te va", "cómo va todo", "qué haces", "qué estás haciendo"]
            
            # If the message is a greeting or contains a common question, it's not bullying
            if (message_lower.strip() in common_greetings or 
                any(greeting in message_lower for greeting in common_greetings) or
                any(question in message_lower for question in common_questions) or
                len(message_lower.split()) <= 3):
                return "ninguno", False, 0.0
            
            # Check for positive sentiment words
            positive_words = ["feliz", "contento", "alegre", "bien", "genial", "excelente", "maravilloso", 
                             "fantástico", "divertido", "agradable", "bueno", "positivo", "encantado"]
            
            # If the message contains positive words and is very short, it's likely not bullying
            if any(word in message_lower for word in positive_words) and len(message_lower.split()) < 5:
                # Skip model prediction for clearly positive messages
                model_prediction = 0
                model_confidence = 0.9
                probability = [0.9, 0.1]
            else:
                # Run model prediction for other messages
                model_prediction, probability = self.bullying_model.predict(message)
                model_confidence = max(probability) if probability else 0.0
            
            # Check for physical bullying keywords
            if any(word in message_lower for word in ["empujar", "empujan", "golpear", "golpean", "pegar", "pegan"]):
                bullying_type = "físico"
                prediction = 1
                confidence = max(0.9, model_confidence)  # Use at least 0.9 confidence for explicit mentions
            # Check for cyberbullying keywords
            elif any(word in message_lower for word in ["mensaje", "whatsapp", "facebook", "instagram", "amenaza"]):
                bullying_type = "cibernético"
                prediction = 1
                confidence = max(0.85, model_confidence)
            # Check for verbal bullying keywords
            elif any(word in message_lower for word in ["insultar", "insulto", "gritar", "burlar", "burla"]):
                bullying_type = "verbal"
                prediction = 1
                confidence = max(0.85, model_confidence)
            # Check for social bullying keywords
            elif any(word in message_lower for word in ["excluir", "excluyen", "ignorar", "ignoran", "rechazar"]):
                bullying_type = "social"
                prediction = 1
                confidence = max(0.85, model_confidence)
            # If model detected bullying but we haven't categorized it yet
            elif model_prediction == 1:
                # Require a minimum confidence for uncategorized bullying detection
                if model_confidence > 0.7:
                    prediction = 1
                    confidence = model_confidence
                    
                    # Determine bullying type based on keywords if not already set
                    if any(word in message_lower for word in ["insultar", "insulto", "gritar", "burlar", "burla", "decir", "dice", "dicen", "llamar", "llaman"]):
                        bullying_type = "verbal"
                    elif any(word in message_lower for word in ["excluir", "excluyen", "ignorar", "ignoran", "rechazar", "rechazan", "no me invitan", "no me incluyen", "no me hablan"]):
                        bullying_type = "social"
                    elif any(word in message_lower for word in ["mensaje", "whatsapp", "facebook", "instagram", "tiktok", "internet", "online", "en línea", "foto", "video"]):
                        bullying_type = "cibernético"
                    elif any(word in message_lower for word in ["empujar", "empujan", "golpear", "golpean", "pegar", "pegan", "patear", "patean", "tirar", "tiran"]):
                        bullying_type = "físico"
                    else:
                        # Para clasificar como psicológico, requerimos un umbral de confianza más alto
                        if model_confidence > 0.85:
                            bullying_type = "psicológico"  # Default type if bullying detected but type unclear
                        else:
                            # Si la confianza no es suficiente para bullying psicológico, no lo clasificamos como bullying
                            prediction = 0
                            confidence = 1.0 - model_confidence  # Invertir la confianza para no-bullying
                else:
                    # If confidence is too low, don't classify as bullying
                    prediction = 0
                    confidence = 1.0 - model_confidence  # Invert confidence for non-bullying
            
            return bullying_type, bool(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error analyzing bullying: {e}")
            return "ninguno", False, 0.0
    
    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get the chat history for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            List of chat log entries for the user
        """
        try:
            return [log for log in self.logs if log.get('user_id') == user_id]
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []
    
    def get_conversation_summary(self, user_id: str) -> str:
        """
        Get a summary of the conversation for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            A string summary of the conversation
        """
        return self.conversation_manager.get_conversation_summary(user_id)

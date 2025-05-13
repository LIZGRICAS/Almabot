"""
Conversation Manager Module

This module provides functionality for managing conversation context
and history for the AlmaBot chatbot.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class ConversationManager:
    """
    Manages conversation context and history for the chatbot.
    Provides methods for retrieving, storing, and updating conversation context.
    """
    
    def __init__(self, db_connection=None):
        """
        Initialize the conversation manager.
        
        Args:
            db_connection: Database connection object for retrieving conversation history
        """
        self.db = db_connection
        self.active_conversations = {}  # user_id -> conversation context
        
    def get_conversation_context(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Retrieve recent conversation history for context.
        
        Args:
            user_id: The ID of the user
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        if user_id not in self.active_conversations:
            # Initialize empty conversation
            self.active_conversations[user_id] = []
            
            # Load from database if available
            if self.db:
                try:
                    messages = self.db.get_recent_messages(user_id, limit)
                    
                    # Convert database messages to the format expected by LLM
                    for msg in messages:
                        role = "assistant" if msg.get("sender") == "bot" else "user"
                        self.active_conversations[user_id].append({
                            "role": role,
                            "content": msg.get("content", "")
                        })
                except Exception as e:
                    print(f"Error loading conversation history: {e}")
        
        return self.active_conversations[user_id]
    
    def add_message(self, user_id: str, role: str, content: str) -> None:
        """
        Add a new message to the conversation context.
        
        Args:
            user_id: The ID of the user
            role: The role of the message sender ("user" or "assistant")
            content: The message content
        """
        if user_id not in self.active_conversations:
            self.active_conversations[user_id] = []
        
        # Add message to context
        self.active_conversations[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim context if it gets too long (keep last 20 messages)
        if len(self.active_conversations[user_id]) > 20:
            self.active_conversations[user_id] = self.active_conversations[user_id][-20:]
    
    def get_conversation_summary(self, user_id: str) -> str:
        """
        Generate a summary of the conversation for the given user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            A string summary of the conversation
        """
        if user_id not in self.active_conversations or not self.active_conversations[user_id]:
            return "No hay conversación activa."
        
        messages = self.active_conversations[user_id]
        summary = "Resumen de la conversación:\n\n"
        
        for i, msg in enumerate(messages):
            role = "Usuario" if msg["role"] == "user" else "AlmaBot"
            summary += f"{role}: {msg['content']}\n\n"
        
        return summary
    
    def clear_conversation(self, user_id: str) -> None:
        """
        Clear the conversation context for a user.
        
        Args:
            user_id: The ID of the user
        """
        if user_id in self.active_conversations:
            self.active_conversations[user_id] = []

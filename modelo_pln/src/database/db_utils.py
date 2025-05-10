from typing import Optional, Dict, Any
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import uuid

class Database:
    def __init__(self, host: str, user: str, password: str, database: str):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return True
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False

    def create_conversation(self, user_id: str, metadata: Dict[str, Any]) -> str:
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            conversation_id = str(uuid.uuid4())
            
            sql = """
                INSERT INTO conversations (id, user_id, metadata)
                VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (conversation_id, user_id, metadata))
            self.connection.commit()
            return conversation_id
        except Error as e:
            print(f"Error creating conversation: {e}")
            raise

    def create_message(self, conversation_id: str, sender: str, content: str, sentiment_score: float) -> str:
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            message_id = str(uuid.uuid4())
            
            sql = """
                INSERT INTO messages (id, conversation_id, sender, content, sentiment_score)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (message_id, conversation_id, sender, content, sentiment_score))
            self.connection.commit()
            return message_id
        except Error as e:
            print(f"Error creating message: {e}")
            raise

    def create_emotional_state(self, message_id: str, emotion_type: str, intensity: float):
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            state_id = str(uuid.uuid4())
            
            sql = """
                INSERT INTO emotional_states (id, message_id, emotion_type, intensity)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (state_id, message_id, emotion_type, intensity))
            self.connection.commit()
        except Error as e:
            print(f"Error creating emotional state: {e}")
            raise

    def create_risk_assessment(self, message_id: str, risk_level: str, risk_type: str):
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            assessment_id = str(uuid.uuid4())
            
            sql = """
                INSERT INTO risk_assessment (id, message_id, risk_level, risk_type)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (assessment_id, message_id, risk_level, risk_type))
            self.connection.commit()
        except Error as e:
            print(f"Error creating risk assessment: {e}")
            raise

    def get_user_history(self, user_id: str) -> list:
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor(dictionary=True)
            
            sql = """
                SELECT 
                    m.id as message_id,
                    m.content,
                    m.sender,
                    m.sentiment_score,
                    m.created_at,
                    e.emotion_type,
                    e.intensity,
                    r.risk_level,
                    r.risk_type
                FROM messages m
                LEFT JOIN emotional_states e ON m.id = e.message_id
                LEFT JOIN risk_assessment r ON m.id = r.message_id
                WHERE m.conversation_id IN (
                    SELECT id FROM conversations WHERE user_id = %s
                )
                ORDER BY m.created_at DESC
            """
            cursor.execute(sql, (user_id,))
            return cursor.fetchall()
        except Error as e:
            print(f"Error getting user history: {e}")
            raise

    def close(self):
        if self.connection:
            self.connection.close()

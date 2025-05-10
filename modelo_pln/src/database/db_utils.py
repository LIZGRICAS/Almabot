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
        self.system_user_id = str(uuid.uuid4())  # System user ID for automated operations

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

    def log_audit(self, table_name: str, record_id: str, action: str, old_data: Dict = None, new_data: Dict = None, user_id: str = None):
        """Log an audit entry for a database operation"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            audit_id = str(uuid.uuid4())
            
            sql = """
                INSERT INTO audit_log (id, table_name, record_id, action, old_data, new_data, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                audit_id,
                table_name,
                record_id,
                action,
                old_data,
                new_data,
                user_id or self.system_user_id
            ))
            self.connection.commit()
        except Error as e:
            print(f"Error logging audit: {e}")
            raise

    def create_conversation(self, user_id: str, metadata: Dict[str, Any], created_by: str = None) -> str:
        """Create a new conversation with audit logging"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            conversation_id = str(uuid.uuid4())
            
            # Insert conversation
            sql = """
                INSERT INTO conversations (id, user_id, metadata, created_by)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (conversation_id, user_id, metadata, created_by or self.system_user_id))
            
            # Log audit
            self.log_audit(
                table_name='conversations',
                record_id=conversation_id,
                action='INSERT',
                new_data={'user_id': user_id, 'metadata': metadata},
                user_id=created_by
            )
            
            self.connection.commit()
            return conversation_id
        except Error as e:
            print(f"Error creating conversation: {e}")
            raise

    def create_message(self, conversation_id: str, sender: str, content: str, sentiment_score: float, created_by: str = None) -> str:
        """Create a new message with audit logging"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            message_id = str(uuid.uuid4())
            
            # Insert message
            sql = """
                INSERT INTO messages (id, conversation_id, sender, content, sentiment_score, created_by)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                message_id,
                conversation_id,
                sender,
                content,
                sentiment_score,
                created_by or self.system_user_id
            ))
            
            # Log audit
            self.log_audit(
                table_name='messages',
                record_id=message_id,
                action='INSERT',
                new_data={'conversation_id': conversation_id, 'sender': sender, 'content': content},
                user_id=created_by
            )
            
            self.connection.commit()
            return message_id
        except Error as e:
            print(f"Error creating message: {e}")
            raise

    def create_emotional_state(self, message_id: str, emotion_type: str, intensity: float, created_by: str = None):
        """Create a new emotional state with audit logging"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            state_id = str(uuid.uuid4())
            
            # Insert emotional state
            sql = """
                INSERT INTO emotional_states (id, message_id, emotion_type, intensity, created_by)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                state_id,
                message_id,
                emotion_type,
                intensity,
                created_by or self.system_user_id
            ))
            
            # Log audit
            self.log_audit(
                table_name='emotional_states',
                record_id=state_id,
                action='INSERT',
                new_data={'message_id': message_id, 'emotion_type': emotion_type, 'intensity': intensity},
                user_id=created_by
            )
            
            self.connection.commit()
        except Error as e:
            print(f"Error creating emotional state: {e}")
            raise

    def create_anonymous_user(self, age: Optional[int] = None, neighborhood: Optional[str] = None, school: Optional[str] = None, created_by: str = None) -> str:
        """Create a new anonymous user with audit logging"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            user_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())
            
            # Insert anonymous user
            sql = """
                INSERT INTO anonymous_users (id, session_id, age, neighborhood, school, created_by)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                user_id,
                session_id,
                age,
                neighborhood,
                school,
                created_by or self.system_user_id
            ))
            
            # Log audit
            self.log_audit(
                table_name='anonymous_users',
                record_id=user_id,
                action='INSERT',
                new_data={'session_id': session_id, 'age': age, 'neighborhood': neighborhood, 'school': school},
                user_id=created_by
            )
            
            self.connection.commit()
            return user_id
        except Error as e:
            print(f"Error creating anonymous user: {e}")
            raise

    def create_risk_assessment(self, message_id: str, risk_level: str, risk_type: str, created_by: str = None):
        """Create a new risk assessment with audit logging"""
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

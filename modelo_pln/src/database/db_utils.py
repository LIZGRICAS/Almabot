from typing import Optional, Dict, Any
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import uuid
import json

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
            
            # Convertir diccionarios a cadenas JSON
            old_data_json = json.dumps(old_data) if old_data is not None else None
            new_data_json = json.dumps(new_data) if new_data is not None else None
            
            # Insert audit log
            sql = """
                INSERT INTO audit_log (id, table_name, record_id, action, old_data, new_data, created_by)
                VALUES (UUID(), %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                table_name,
                record_id,
                action,
                old_data_json,
                new_data_json,
                user_id or self.system_user_id
            ))
            self.connection.commit()
        except Error as e:
            print(f"Error logging audit: {e}")
            print(f"SQL: {sql}")
            print(f"Params: {(table_name, record_id, action, old_data, new_data, user_id or self.system_user_id)}")
            raise

    def create_conversation(self, user_id: str, metadata: Dict[str, Any], created_by: str = None) -> str:
        """Create a new conversation with audit logging"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            conversation_id = str(uuid.uuid4())
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Insert conversation
            sql = """
                INSERT INTO conversations (id, user_id, metadata, created_by)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (conversation_id, user_id, metadata_json, created_by or self.system_user_id))
            
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
            
            try:
                # Try with created_by first
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
            except Error as e:
                if "Unknown column 'created_by'" in str(e):
                    # Fallback to schema without created_by
                    sql = """
                        INSERT INTO emotional_states (id, message_id, emotion_type, intensity)
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        state_id,
                        message_id,
                        emotion_type,
                        intensity
                    ))
                else:
                    raise
            
            # Log audit
            try:
                self.log_audit(
                    table_name='emotional_states',
                    record_id=state_id,
                    action='INSERT',
                    new_data={'message_id': message_id, 'emotion_type': emotion_type, 'intensity': intensity},
                    user_id=created_by
                )
            except Exception as audit_error:
                print(f"Warning: Failed to create audit log: {audit_error}")
                # Continue even if audit log fails
            
            self.connection.commit()
        except Error as e:
            print(f"Error creating emotional state: {e}")
            raise

    def create_anonymous_user(self, age: Optional[int] = None, neighborhood: Optional[str] = None, school: Optional[str] = None, created_by: str = None) -> str:
        """Create a new anonymous user with audit logging"""
        try:
            print("Creating anonymous user with data:", {
                'age': age,
                'neighborhood': neighborhood,
                'school': school,
                'created_by': created_by or self.system_user_id
            })
            
            if not self.connection:
                print("No active connection, attempting to connect...")
                if not self.connect():
                    raise Error("Failed to connect to database")
            
            cursor = self.connection.cursor()
            user_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())
            
            # Insert anonymous user
            sql = """
                INSERT INTO anonymous_users (id, session_id, age, neighborhood, school, created_by)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (
                user_id,
                session_id,
                age,
                neighborhood,
                school,
                created_by or self.system_user_id
            )
            
            print("Executing SQL:", sql)
            print("With params:", params)
            
            cursor.execute(sql, params)
            print("User inserted successfully")
            
            try:
                # Log audit
                self.log_audit(
                    table_name='anonymous_users',
                    record_id=user_id,
                    action='INSERT',
                    new_data={'session_id': session_id, 'age': age, 'neighborhood': neighborhood, 'school': school},
                    user_id=created_by
                )
                print("Audit log created successfully")
            except Exception as audit_error:
                print(f"Warning: Failed to create audit log: {audit_error}")
                # Continue even if audit log fails
            
            self.connection.commit()
            print(f"Transaction committed. Created user with ID: {user_id}")
            return user_id
        except Error as e:
            error_msg = f"Error creating anonymous user: {str(e)}"
            print(error_msg)
            if hasattr(e, 'errno'):
                print(f"MySQL Error {e.errno}: {e.sqlstate} - {e.msg}")
            if hasattr(e, '_full_msg'):
                print("Full error:", e._full_msg)
            raise Error(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in create_anonymous_user: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise Error(error_msg) from e

    def create_risk_assessment(self, message_id: str, risk_level: str, risk_type: str, created_by: str = None):
        """Create a new risk assessment with audit logging"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            assessment_id = str(uuid.uuid4())
            
            # First, check if the created_by column exists
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = 'risk_assessment' 
                AND COLUMN_NAME = 'created_by'
            """, (self.database,))
            
            has_created_by = cursor.fetchone()[0] > 0
            
            if has_created_by:
                sql = """
                    INSERT INTO risk_assessment (id, message_id, risk_level, risk_type, created_by)
                    VALUES (%s, %s, %s, %s, %s)
                """
                params = (
                    assessment_id, 
                    message_id, 
                    risk_level, 
                    risk_type,
                    created_by or self.system_user_id
                )
            else:
                sql = """
                    INSERT INTO risk_assessment (id, message_id, risk_level, risk_type)
                    VALUES (%s, %s, %s, %s)
                """
                params = (
                    assessment_id, 
                    message_id, 
                    risk_level, 
                    risk_type
                )
            
            cursor.execute(sql, params)
            
            # Log audit
            try:
                self.log_audit(
                    table_name='risk_assessment',
                    record_id=assessment_id,
                    action='INSERT',
                    new_data={
                        'message_id': message_id, 
                        'risk_level': risk_level, 
                        'risk_type': risk_type
                    },
                    user_id=created_by
                )
            except Exception as audit_error:
                print(f"Warning: Failed to create audit log for risk assessment: {audit_error}")
                # Continue even if audit log fails
            
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
            
    def end_conversation(self, conversation_id: str, created_by: str = None) -> bool:
        """Mark a conversation as ended with audit logging"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor(dictionary=True)
            
            # Get current conversation data for audit
            cursor.execute("SELECT * FROM conversations WHERE id = %s", (conversation_id,))
            old_data = cursor.fetchone()
            
            if not old_data:
                print(f"Conversation not found: {conversation_id}")
                return False
            
            # Update conversation
            sql = """
                UPDATE conversations 
                SET end_time = NOW(), is_active = FALSE, updated_by = %s, updated_at = NOW()
                WHERE id = %s
            """
            cursor.execute(sql, (created_by or self.system_user_id, conversation_id))
            
            # Log audit
            self.log_audit(
                table_name='conversations',
                record_id=conversation_id,
                action='UPDATE',
                old_data={'is_active': True},
                new_data={'is_active': False, 'end_time': datetime.now().isoformat()},
                user_id=created_by
            )
            
            self.connection.commit()
            return True
        except Error as e:
            print(f"Error ending conversation: {e}")
            raise
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Generate a summary of a conversation"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor(dictionary=True)
            
            # Get conversation details
            cursor.execute("""
                SELECT 
                    id, user_id, start_time, end_time,
                    TIMESTAMPDIFF(MINUTE, start_time, COALESCE(end_time, NOW())) as duration_minutes
                FROM conversations 
                WHERE id = %s
            """, (conversation_id,))
            
            conversation = cursor.fetchone()
            if not conversation:
                return {
                    "duration_minutes": 0,
                    "message_count": 0,
                    "risk_level": "unknown"
                }
            
            # Count messages
            cursor.execute("""
                SELECT COUNT(*) as message_count
                FROM messages
                WHERE conversation_id = %s
            """, (conversation_id,))
            
            message_count = cursor.fetchone()["message_count"]
            
            # Get highest risk level
            cursor.execute("""
                SELECT risk_level, COUNT(*) as count
                FROM risk_assessment ra
                JOIN messages m ON ra.message_id = m.id
                WHERE m.conversation_id = %s
                GROUP BY risk_level
                ORDER BY FIELD(risk_level, 'high', 'medium', 'low') ASC
                LIMIT 1
            """, (conversation_id,))
            
            risk_result = cursor.fetchone()
            risk_level = risk_result["risk_level"] if risk_result else "low"
            
            return {
                "duration_minutes": conversation["duration_minutes"],
                "message_count": message_count,
                "risk_level": risk_level
            }
        except Error as e:
            print(f"Error getting conversation summary: {e}")
            return {
                "duration_minutes": 0,
                "message_count": 0,
                "risk_level": "error"
            }

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
from bot.chatbot import Chatbot
from database.db_utils import Database
from .logging_config import logger
import os

# Initialize database connection
DB_HOST = os.getenv('DB_HOST', 'db')
DB_USER = os.getenv('DB_USER', 'almabot')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'almabot123')
DB_NAME = os.getenv('DB_NAME', 'almabot')

db = Database(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

app = FastAPI(
    title="Chatbot Anti-Bullying API",
    description="API para detección y respuesta a casos de bullying",
    version="1.0.0"
)

# Inicializar el chatbot
chatbot = Chatbot()

logger.info("API initialized")
logger.info(f"Database connection: {DB_HOST}/{DB_NAME}")

class MessageRequest(BaseModel):
    user_id: str
    message: str
    metadata: Optional[Dict] = None

class MessageResponse(BaseModel):
    response: str
    analysis: Dict
    timestamp: str
    conversation_id: str
    message_id: str

class UserHistoryResponse(BaseModel):
    history: List[Dict]
    user_id: str

@app.post("/chat", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    """
    Process a chat message and return a response
    """
    try:
        print("Processing message...")
        logger.info(f"Processing message from user: {request.user_id}")
        logger.debug(f"Message content: {request.message}")
        
        # Connect to database
        try:
            if not db.connect():
                logger.error("Database connection failed")
                raise HTTPException(status_code=500, detail="Database connection failed")
            logger.info("Database connection successful")
            
            # Create conversation if it doesn't exist
            conversation_id = db.create_conversation(request.user_id, request.metadata or {})
            logger.info(f"Conversation created with ID: {conversation_id}")
            
            # Process the message with the chatbot
            response, analysis = chatbot.process_message(request.message)
            logger.info("Message processed successfully")
            logger.debug(f"Analysis result: {analysis}")
            
            # Create message record
            message_id = db.create_message(
                conversation_id=conversation_id,
                sender='user',
                content=request.message,
                sentiment_score=analysis.get('sentiment_score', 0.0)
            )
            logger.info(f"Message record created with ID: {message_id}")
            
            # Create emotional state if available
            if 'emotion' in analysis:
                db.create_emotional_state(
                    message_id=message_id,
                    emotion_type=analysis['emotion']['type'],
                    intensity=analysis['emotion']['intensity']
                )
                logger.info(f"Emotional state created: {analysis['emotion']['type']}")
            
            # Create risk assessment if available
            if 'risk' in analysis:
                db.create_risk_assessment(
                    message_id=message_id,
                    risk_level=analysis['risk']['level'],
                    risk_type=analysis['risk']['type']
                )
                logger.warning(f"Risk assessment created: {analysis['risk']['level']} - {analysis['risk']['type']}")
        except Exception as e:
            logger.error(f"Error during database operations: {str(e)}", exc_info=True)
            raise

        return MessageResponse(
            response=response,
            analysis=analysis,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id,
            message_id=message_id
        )
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        logger.info("Database connection closed")

@app.get("/history/{user_id}", response_model=UserHistoryResponse)
async def get_user_history(user_id: str):
    """
    Get user's chat history
    """
    try:
        logger.info(f"Getting history for user: {user_id}")
        
        # Connect to database
        if not db.connect():
            logger.error("Database connection failed")
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Get history from database
        history = db.get_user_history(user_id)
        logger.info(f"Retrieved {len(history)} records for user {user_id}")
        
        return UserHistoryResponse(
            history=history,
            user_id=user_id
        )
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        logger.info("Database connection closed")

@app.get("/health")
async def health_check():
    """
    Endpoint para verificar el estado de la API
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
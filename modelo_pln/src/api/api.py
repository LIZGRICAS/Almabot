from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
from bot.chatbot import Chatbot
from database.db_utils import Database
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
        # Connect to database
        if not db.connect():
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Create conversation if it doesn't exist
        conversation_id = db.create_conversation(request.user_id, request.metadata or {})

        # Process the message with the chatbot
        response, analysis = chatbot.process_message(request.message)
        
        # Create message record
        message_id = db.create_message(
            conversation_id=conversation_id,
            sender='user',
            content=request.message,
            sentiment_score=analysis.get('sentiment_score', 0.0)
        )

        # Create emotional state if available
        if 'emotion' in analysis:
            db.create_emotional_state(
                message_id=message_id,
                emotion_type=analysis['emotion']['type'],
                intensity=analysis['emotion']['intensity']
            )

        # Create risk assessment if available
        if 'risk' in analysis:
            db.create_risk_assessment(
                message_id=message_id,
                risk_level=analysis['risk']['level'],
                risk_type=analysis['risk']['type']
            )

        return MessageResponse(
            response=response,
            analysis=analysis,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id,
            message_id=message_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/history/{user_id}", response_model=UserHistoryResponse)
async def get_user_history(user_id: str):
    """
    Get user's chat history
    """
    try:
        # Connect to database
        if not db.connect():
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Get history from database
        history = db.get_user_history(user_id)
        
        return UserHistoryResponse(
            history=history,
            user_id=user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/health")
async def health_check():
    """
    Endpoint para verificar el estado de la API
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
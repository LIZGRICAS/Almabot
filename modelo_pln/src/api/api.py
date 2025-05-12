from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from bot.chatbot import Chatbot
from database.db_utils import Database
import os
import logging

# Create API Router
api_router = APIRouter(prefix="/api")

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Configuración de CORS
allowed_origins = [
    "https://mc72r2h5-8000.use2.devtunnels.ms",  # VS Code tunnel URL
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:3000",
    "http://localhost:5173",  # Vite dev server
    "https://translate.google.com"  # Google Translate for testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600  # Cache preflight requests for 10 minutes
)

# Inicializar el chatbot
chatbot = Chatbot()


# Modelos Pydantic
class UserRequest(BaseModel):
    age: Optional[int] = None
    neighborhood: Optional[str] = None
    school: Optional[str] = None
    created_by: Optional[str] = None

class MessageRequest(BaseModel):
    user_id: str
    message: str
    metadata: Optional[Dict] = None
    created_by: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    analysis: Dict
    timestamp: str
    conversation_id: str
    message_id: str

class UserHistoryResponse(BaseModel):
    history: List[Dict]
    user_id: str

class UserResponse(BaseModel):
    user_id: str
    created_at: datetime = None

class ConversationRequest(BaseModel):
    user_id: str
    metadata: Optional[Dict] = None
    created_by: Optional[str] = None

# Middleware para log de solicitudes
# Keep the middleware at the app level
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Endpoint de health check
# Keep the root health check at the app level

@app.get("/")
async def health_check():
    """
    Endpoint para verificar el estado de la API
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """
    Endpoint para verificar el estado de la API
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@api_router.post("/user", response_model=UserResponse, tags=["users"])
async def create_user(request: UserRequest):
    """
    Create a new anonymous user and return the user_id
    """
    try:
        print("Creating new anonymous user")
        print(f"Request data: {request.dict()}")
        
        # Connect to database
        print("Attempting to connect to database...")
        if not db.connect():
            error_msg = "Database connection failed"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        print("Database connection successful")
        
        # Create user
        print("Attempting to create user...")
        try:
            user_id = db.create_anonymous_user(
                age=request.age,
                neighborhood=request.neighborhood,
                school=request.school,
                created_by=request.created_by
            )
            print(f"User created with ID: {user_id}")
            
            return UserResponse(user_id=user_id, created_at=datetime.now())
        except Exception as e:
            print(f"Error in create_anonymous_user: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in create_user: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

# @api_router.post("/conversation", response_model=Dict, tags=["conversations"])
# async def create_conversation(request: ConversationRequest):
#     """
#     Create a new conversation and return the conversation_id
#     """
#     try:
#         print(f"Creating conversation for user: {request.user_id}")
        
#         # Connect to database
#         if not db.connect():
#             print("Database connection failed")
#             raise HTTPException(status_code=500, detail="Database connection failed")
        
#         # Create conversation
#         conversation_id = db.create_conversation(
#             request.user_id,
#             request.metadata or {},
#             created_by=request.created_by
#         )
#         print(f"Conversation created with ID: {conversation_id}")
        
#         return {"conversation_id": conversation_id}
#     except Exception as e:
#         print(f"Error creating conversation: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/chat-message", response_model=MessageResponse, tags=["chat"])
async def process_message(request: MessageRequest):
    """
    Process a chat message and return a response
    """
    try:
        print("Processing message...")
        print(f"Processing message from user: {request.user_id}")
        logger.debug(f"Message content: {request.message}")
        
        # Connect to database
        try:
            if not db.connect():
                print("Database connection failed")
                raise HTTPException(status_code=500, detail="Database connection failed")
            print("Database connection successful")
            
            # Check if user exists, if not create a new anonymous user
            try:
                # Try to get user from database
                cursor = db.connection.cursor(dictionary=True)
                cursor.execute("SELECT id FROM anonymous_users WHERE id = %s", (request.user_id,))
                user = cursor.fetchone()
                
                if not user:
                    # Create new anonymous user if not exists
                    print(f"Creating new anonymous user with ID: {request.user_id}")
                    cursor.execute(
                        """
                        INSERT INTO anonymous_users (id, created_by, is_active, last_interaction)
                        VALUES (%s, %s, TRUE, NOW())
                        """,
                        (request.user_id, request.created_by or 'system')
                    )
                    db.connection.commit()
                
                # Create conversation
                conversation_id = db.create_conversation(
                    request.user_id,
                    request.metadata or {},
                    created_by=request.created_by
                )
                print(f"Conversation created with ID: {conversation_id}")
                
            except Exception as e:
                print(f"Error creating user or conversation: {str(e)}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error creating user or conversation: {str(e)}")
            
            # Process the message with the chatbot and get the full analysis
            response = chatbot.process_message(request.user_id, request.message)
            print("Message processed successfully")
            logger.debug(f"Response: {response}")
            
            # Get the analysis from the chatbot's logs
            last_log = chatbot.logs[-1] if hasattr(chatbot, 'logs') and chatbot.logs else None
            
            # Create the analysis structure with sentiment and other metrics
            analysis = {
                'is_bullying': last_log.get('is_bullying', False) if last_log else False,
                'confidence': last_log.get('confidence', 0.0) if last_log else 0.0,
                'bullying_type': last_log.get('bullying_type', 'ninguno') if last_log else 'ninguno',
                'keywords': last_log.get('keywords', []) if last_log else [],
                'sentiment': last_log.get('sentiment', 'neutral') if last_log else 'neutral',
                'sentiment_score': last_log.get('sentiment_score', 0.0) if last_log else 0.0,
                'emotion': last_log.get('emotion', {'type': 'neutral', 'intensity': 0.0}) if last_log else {'type': 'neutral', 'intensity': 0.0},
                'risk': last_log.get('risk', {'level': 'low', 'type': 'none'}) if last_log else {'level': 'low', 'type': 'none'}
            }
            
            # Create message record
            message_id = db.create_message(
                conversation_id=conversation_id,
                sender='user',
                content=request.message,
                sentiment_score=analysis.get('sentiment_score', 0.0),
                created_by=request.created_by
            )
            print(f"Message record created with ID: {message_id}")
            
            # Create emotional state if available
            if 'emotion' in analysis:
                db.create_emotional_state(
                    message_id=message_id,
                    emotion_type=analysis['emotion']['type'],
                    intensity=analysis['emotion']['intensity'],
                    created_by=request.created_by
                )
                print(f"Emotional state created: {analysis['emotion']['type']}")
            
            # Create risk assessment if available
            if 'risk' in analysis:
                # Ensure risk_type is one of the allowed values: 'bullying', 'suicide', 'abuse', 'other'
                risk_type = analysis['risk']['type'].lower()
                if risk_type not in ['bullying', 'suicide', 'abuse', 'other']:
                    risk_type = 'other'  # Default to 'other' if not a valid type
                
                # Ensure risk_level is one of the allowed values: 'low', 'medium', 'high'
                risk_level = analysis['risk']['level'].lower()
                if risk_level not in ['low', 'medium', 'high']:
                    risk_level = 'low'  # Default to 'low' if not a valid level
                
                db.create_risk_assessment(
                    message_id=message_id,
                    risk_level=risk_level,
                    risk_type=risk_type,
                    created_by=request.created_by
                )
                logger.warning(f"Risk assessment created: {risk_level} - {risk_type}")
        except Exception as e:
            print(f"Error during database operations: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        return MessageResponse(
            response=response,
            analysis=analysis,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id,
            message_id=message_id
        )
    except Exception as e:
        import traceback
        print(f"Error processing message: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        print("Database connection closed")

@api_router.get("/history/{user_id}", response_model=UserHistoryResponse, tags=["history"])
async def get_user_history(user_id: str):
    """
    Get user's chat history
    """
    try:
        print(f"Getting history for user: {user_id}")
        
        # Connect to database
        if not db.connect():
            print("Database connection failed")
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Get history from database
        history = db.get_user_history(user_id)
        print(f"Retrieved {len(history)} records for user {user_id}")
        
        return UserHistoryResponse(
            history=history,
            user_id=user_id
        )
    except Exception as e:
        print(f"Error getting user history: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        print("Database connection closed")

@api_router.get("/health")
async def health_check():
    """
    Endpoint para verificar el estado de la API
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Include the API router after all routes are defined
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
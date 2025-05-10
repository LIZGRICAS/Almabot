from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
from chatbot import Chatbot

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

class MessageResponse(BaseModel):
    response: str
    analysis: Dict
    timestamp: str

class UserHistoryResponse(BaseModel):
    history: List[Dict]

@app.post("/chat", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    """
    Procesa un mensaje y retorna la respuesta del chatbot junto con el análisis
    """
    try:
        response = chatbot.process_message(request.user_id, request.message)
        
        # Obtener el último log guardado
        last_log = chatbot.logs[-1] if chatbot.logs else None
        
        return MessageResponse(
            response=response,
            analysis=last_log["analysis"] if last_log else {},
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}", response_model=UserHistoryResponse)
async def get_user_history(user_id: str):
    """
    Obtiene el historial de interacciones de un usuario específico
    """
    try:
        history = chatbot.get_user_history(user_id)
        return UserHistoryResponse(history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Endpoint para verificar el estado de la API
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
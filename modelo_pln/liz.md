# Pasos de ejecución de la app

1. Te clonas el repositorio
2. Entras al directorio

```bash
cd modelo_pln
```

3. Ejecutas el docker compose

```bash
docker-compose up --build -d
```

# Documentación de la API de AlmaBot

## Pasos de ejecución de las APIs


1. Crear usuario anónimo
2. Procesar mensaje


## Endpoints Disponibles

### 1. Health Check
Verifica el estado de la API.

- **Método**: `GET`
- **Ruta**: `/health`
- **Respuesta Exitosa**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2025-05-11T22:30:45.123456"
  }
  ```

### 2. Crear Usuario Anónimo
Crea un nuevo usuario anónimo.

- **Método**: `POST`
- **Ruta**: `/api/user`
- **Cuerpo de la Solicitud**:
  ```json
  {
    "age": 12,
    "neighborhood": "Centro",
    "school": "Escuela Primaria 123",
    "created_by": "system"
  }
  ```
- **Respuesta Exitosa**:
  ```json
  {
    "user_id": "abc123",
    "created_at": "2025-05-11T22:30:45.123456"
  }
  ```

### 3. Procesar Mensaje
Envía un mensaje al chatbot y recibe una respuesta.

- **Método**: `POST`
- **Ruta**: `/api/chat-message`
- **Cuerpo de la Solicitud**:
  ```json
  {
    "user_id": "abc123",
    "message": "Hola, me siento triste",
    "metadata": {
      "platform": "telegram"
    },
    "created_by": "user123"
  }
  ```
- **Respuesta Exitosa**:
  ```json
  {
    "response": "Lamento escuchar que te sientes así. ¿Quieres contarme más sobre lo que te está pasando?",
    "analysis": {
      "is_bullying": false,
      "confidence": 0.1,
      "bullying_type": "ninguno",
      "keywords": ["triste"],
      "sentiment": "negative",
      "sentiment_score": -0.8,
      "emotion": {
        "type": "sadness",
        "intensity": 0.75
      },
      "risk": {
        "level": "low",
        "type": "none"
      }
    },
    "timestamp": "2025-05-11T22:30:45.123456",
    "conversation_id": "conv_123",
    "message_id": "msg_456"
  }
  ```

### 4. Obtener Historial de Usuario
Obtiene el historial de conversaciones de un usuario.

- **Método**: `GET`
- **Ruta**: `/api/history/{user_id}`
- **Parámetros de Ruta**:
  - `user_id` (string, requerido): ID del usuario
- **Respuesta Exitosa**:
  ```json
  {
    "history": [
      {
        "message_id": "msg_123",
        "conversation_id": "conv_123",
        "sender": "user",
        "content": "Hola, me siento triste",
        "sentiment_score": -0.8,
        "created_at": "2025-05-11T22:30:45.123456",
        "emotional_state": {
          "type": "sadness",
          "intensity": 0.75
        },
        "risk_assessment": {
          "level": "low",
          "type": "none"
        }
      },
      {
        "message_id": "msg_456",
        "conversation_id": "conv_123",
        "sender": "bot",
        "content": "Lamento escuchar que te sientes así. ¿Quieres contarme más?",
        "created_at": "2025-05-11T22:30:46.123456"
      }
    ],
    "user_id": "abc123"
  }
  ```

## Ejemplo de Uso con cURL

```bash
# Health Check
curl -X GET "http://localhost:8000/health"

# Crear usuario
curl -X POST "http://localhost:8000/api/user" \
  -H "Content-Type: application/json" \
  -d '{"age": 12, "neighborhood": "Centro", "school": "Escuela Primaria 123"}'

# Enviar mensaje
curl -X POST "http://localhost:8000/api/chat-message" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "abc123", "message": "Hola, me siento triste"}'

# Obtener historial
curl -X GET "http://localhost:8000/api/history/abc123"
```

## Documentación Interactiva

Puedes acceder a la documentación interactiva de la API en:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
# 
# Instrucciones para AlmaBot con Ollama y Mistral 7B

## Descripción General

Hemos mejorado AlmaBot integrando un modelo de lenguaje natural (LLM) local usando Ollama con Mistral 7B. Esta mejora permite que el chatbot:

1. Mantenga el contexto de la conversación
2. Genere respuestas más naturales y empáticas
3. Proporcione apoyo emocional más efectivo
4. Conserve la funcionalidad de detección de bullying para análisis y estadísticas

## Requisitos del Sistema

- Linux o macOS (para Ollama)
- Mínimo 8GB de RAM (16GB recomendado)
- Al menos 5GB de espacio libre en disco para el modelo
- Python 3.8+
- Conexión a Internet para la instalación inicial

## Instalación y Configuración

### Opción 1: Instalación con Docker (Recomendada)

Hemos integrado Ollama en el archivo `docker-compose.8001.yml`, lo que facilita el despliegue completo del sistema:

```bash
cd /home/sublime-dev/dev/python/Almabot/modelo_pln
docker-compose -f docker-compose.8001.yml up -d
```

Este comando iniciará:
- Base de datos MySQL
- Servicio de entrenamiento del modelo de bullying
- Servicio Ollama con el modelo Mistral 7B
- API con integración completa

La primera vez que se ejecute, descargará automáticamente el modelo Mistral 7B (aproximadamente 4.1GB).

### Opción 2: Instalación Local

Si prefieres una instalación local sin Docker, hemos creado un script de instalación:

```bash
cd /home/sublime-dev/dev/python/Almabot/modelo_pln
chmod +x setup_ollama.sh
./setup_ollama.sh
```

Este script:
- Instala Ollama si no está instalado
- Descarga el modelo Mistral 7B
- Actualiza las dependencias en requirements.txt
- Instala las dependencias necesarias

### Verificar la Instalación

Para verificar que Ollama está funcionando correctamente:

```bash
# Con Docker
docker exec almabot-ollama-8001 ollama list

# Instalación local
ollama list
```

Deberías ver `mistral:7b-instruct` en la lista de modelos.

## Estructura de Archivos Nuevos

Hemos agregado los siguientes archivos al proyecto:

- `src/bot/conversation_manager.py`: Gestiona el contexto de las conversaciones
- `src/bot/llm_processor.py`: Procesa las solicitudes al modelo de lenguaje
- `src/bot/enhanced_chatbot.py`: Combina la detección de bullying con la generación de lenguaje natural
- `src/tests/test_enhanced_chatbot.py`: Prueba el chatbot mejorado
- `setup_ollama.sh`: Script para instalar y configurar Ollama

## Cómo Funciona

El flujo de procesamiento de mensajes ahora es:

1. El usuario envía un mensaje a través de la API
2. El sistema detecta si es un caso de bullying usando el modelo existente
3. Se recupera el contexto de la conversación anterior
4. Se envía el mensaje, junto con el contexto, al modelo Mistral 7B
5. El modelo genera una respuesta natural y contextualmente relevante
6. La respuesta se almacena en el contexto de la conversación para futuras interacciones
7. Los datos de análisis de bullying se siguen almacenando en la base de datos para estadísticas

## Pruebas

Para probar el chatbot mejorado:

```bash
# Asegúrate de que Ollama esté ejecutándose
ollama serve &

# Ejecuta el script de prueba
cd /home/sublime-dev/dev/python/Almabot
source venv/bin/activate
python -m modelo_pln.src.tests.test_enhanced_chatbot
```

## Ejecución de la API

Para ejecutar la API con el chatbot mejorado:

```bash
# Asegúrate de que Ollama esté ejecutándose
ollama serve &

# Inicia la API
cd /home/sublime-dev/dev/python/Almabot
source venv/bin/activate
python -m modelo_pln.src.api.api
```

## Ejemplos de Uso

### Ejemplo de solicitud a la API:

```bash
curl --location 'http://localhost:8000/api/chat-message' \
--header 'Content-Type: application/json' \
--data '{   
    "user_id": "8a1f9e23-e52c-42d3-ba92-2a29b32c818a", 
    "message": "Me empujan en la escuela", 
    "created_by": "system" 
}'
```

## Solución de Problemas

### Ollama no inicia

Si Ollama no inicia correctamente, verifica:

```bash
# Reinicia el servicio
pkill ollama
ollama serve
```

### Modelo no disponible

Si el modelo no está disponible:

```bash
# Vuelve a descargar el modelo
ollama pull mistral:7b-instruct
```

### Respuestas lentas

Las primeras respuestas pueden ser lentas mientras el modelo se carga en memoria. Las respuestas posteriores deberían ser más rápidas.

## Personalización

### Cambiar el Modelo

Si deseas usar un modelo diferente (como Llama 2), modifica `enhanced_chatbot.py`:

```python
chatbot = EnhancedChatbot(db_connection=db, model_name="llama2:7b")
```

Y luego descarga el nuevo modelo:

```bash
ollama pull llama2:7b
```

## Notas Adicionales

- El modelo se ejecuta localmente, por lo que no hay costos de API ni problemas de privacidad
- La calidad de las respuestas depende del modelo utilizado
- Mistral 7B ofrece un buen equilibrio entre calidad y requisitos de recursos

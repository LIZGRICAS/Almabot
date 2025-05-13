# AlmaBot - Modelo de Detección de Bullying

## Descripción del Proyecto

AlmaBot es un asistente virtual diseñado para proporcionar apoyo emocional a niños de 6 a 14 años que son víctimas de bullying. El sistema utiliza técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático para detectar indicadores de bullying en los mensajes de los usuarios y responder de manera empática y apropiada.

## Estructura del Proyecto

```
modelo_pln/
├── data/                      # Datos para entrenamiento y evaluación
│   ├── models/                # Modelos entrenados
│   ├── plots/                 # Gráficos de evaluación
│   └── raw/                   # Datos sin procesar
├── src/                       # Código fuente
│   ├── api/                   # API REST para el chatbot
│   ├── bot/                   # Lógica del chatbot
│   ├── database/              # Utilidades de base de datos
│   ├── models/                # Modelos de aprendizaje automático
│   ├── nlp/                   # Procesamiento de lenguaje natural
│   └── tests/                 # Pruebas unitarias
├── models/                    # Modelos entrenados (local)
├── docker-compose.8001.yml    # Configuración de Docker para el servicio
└── requirements.txt           # Dependencias del proyecto
```

## Características Principales

- **Detección de Bullying**: Identifica diferentes tipos de bullying (físico, verbal, social, cibernético) en los mensajes de los usuarios.
- **Evaluación de Riesgo**: Evalúa el nivel de riesgo basado en el contenido del mensaje.
- **Respuestas Empáticas**: Genera respuestas apropiadas según el tipo de bullying y nivel de riesgo detectado.
- **API REST**: Proporciona endpoints para interactuar con el chatbot desde diferentes interfaces.
- **Almacenamiento de Conversaciones**: Guarda las conversaciones para análisis y seguimiento.

## Instalación y Configuración

### Requisitos Previos

- Python 3.8+
- pip
- Virtualenv (opcional)

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd Almabot/modelo_pln
   ```

2. Crear y activar un entorno virtual (opcional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Configuración

1. Configurar variables de entorno:
   ```bash
   # Variables de base de datos
   export DB_HOST=localhost
   export DB_USER=almabot
   export DB_PASSWORD=almabot123
   export DB_NAME=almabot
   ```

## Uso

### Entrenamiento del Modelo

Para entrenar el modelo de detección de bullying:

```bash
python -m src.models.train_model
```

### Ejecución de la API

Para iniciar el servidor API:

```bash
python -m src.api.api
```

### Pruebas

Para ejecutar las pruebas del modelo:

```bash
python -m src.tests.test_model
```

## API Endpoints

- `POST /api/chat-message`: Envía un mensaje al chatbot y recibe una respuesta.
- `POST /api/register`: Registra un nuevo usuario anónimo.
- `POST /api/conversation/start`: Inicia una nueva conversación.
- `POST /api/conversation/end`: Finaliza una conversación activa y genera un resumen.

## Ejemplo de Uso

```python
import requests
import json

# Enviar un mensaje al chatbot
response = requests.post(
    "http://localhost:8000/api/chat-message",
    headers={"Content-Type": "application/json"},
    data=json.dumps({
        "user_id": "8a1f9e23-e52c-42d3-ba92-2a29b32c818a",
        "message": "Me empujan en la escuela",
        "created_by": "system"
    })
)

# Mostrar la respuesta
print(response.json())
```

## Despliegue con Docker

Para desplegar la aplicación usando Docker:

```bash
docker-compose -f docker-compose.8001.yml up -d
```

## Mejoras Recientes

- **Validación Cruzada Mejorada**: Implementación de métricas más completas incluyendo balanced accuracy y ROC AUC.
- **Optimización de Hiperparámetros**: Proceso de optimización de dos etapas para encontrar los mejores parámetros del modelo.
- **Detección de Overfitting**: Comparación de métricas de entrenamiento y validación para detectar overfitting.
- **Evaluación Detallada**: Informes completos de evaluación con matrices de confusión y métricas por categoría.

## Contribución

Para contribuir al proyecto:

1. Haz un fork del repositorio
2. Crea una rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

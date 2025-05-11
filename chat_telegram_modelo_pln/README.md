# AlmaBot - Chat de Telegram con Modelo de Detección de Bullying

Este proyecto implementa un chatbot de Telegram con las siguientes funcionalidades:

1. Menú principal con opciones:
   - Entrenar Modelo
   - Chat con AlmaBot
   - Ver Estadísticas

2. Sistema de detección de bullying usando regresión logística
3. Interfaz de usuario amigable y responsive

## Requisitos

- Python 3.8+
- Telegram Bot Token
- Librerías listadas en requirements.txt

## Instalación

1. Clonar el repositorio
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configurar el token de Telegram en el archivo .env

## Estructura del Proyecto

```
chat_telegram_modelo_pln/
├── telegram_bot.py      # Código principal del bot
├── requirements.txt     # Dependencias del proyecto
├── .env                # Configuración del bot
└── README.md           # Documentación del proyecto
```

## Uso

1. Ejecutar el bot:
   ```bash
   python telegram_bot.py
   ```

2. Buscar el bot en Telegram y comenzar a interactuar

## Funcionalidades

- Entrenamiento automático del modelo de detección de bullying
- Chat interactivo con respuestas empáticas
- Visualización de estadísticas del modelo
- Sistema de detección de bullying en tiempo real

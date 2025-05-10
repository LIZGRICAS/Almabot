# AlmaBot - Asistente Virtual Anti-Bullying

AlmaBot es un asistente virtual basado en inteligencia artificial diseñado para brindar apoyo emocional a niños y niñas entre 6 y 14 años que enfrentan situaciones de acoso escolar.

## Características

- Interfaz de chat amigable y anónima
- Detección de bullying y emociones
- Recomendaciones de recursos
- Sistema de alertas para casos de riesgo
- Integración con Telegram para pruebas

## Requisitos

- Docker y Docker Compose
- Python 3.9+
- MySQL 8.0

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/almabot.git
cd almabot
```

2. Copiar el archivo de configuración:
```bash
cp .env.example .env
```

3. Editar el archivo `.env` con tus credenciales:
- Configurar las credenciales de la base de datos
- Agregar el token de tu bot de Telegram

4. Construir y levantar los servicios:
```bash
docker-compose up --build
```

## Servicios

El proyecto incluye los siguientes servicios Docker:

- `db`: Base de datos MySQL
- `api`: API REST para el chatbot
- `telegram_bot`: Bot de Telegram para pruebas
- `training`: Servicio para entrenar el modelo

## Estructura del Proyecto

```
.
├── api.py                 # API principal
├── chatbot.py            # Lógica del chatbot
├── database/             # Esquema de la base de datos
├── models/              # Modelos entrenados
├── telegram_bot.py      # Bot de Telegram
└── training_phrases.csv # Datos de entrenamiento
```

## Uso

1. Acceder al chatbot a través de Telegram:
   - Buscar el bot en Telegram
   - Iniciar una conversación

2. Acceder a la API:
   - Endpoint base: `http://localhost:8000`
   - Documentación: `http://localhost:8000/docs`

## Seguridad

- Todas las interacciones son anónimas
- Los datos sensibles se almacenan de forma segura
- Sistema de alertas para casos de riesgo

## Contribución

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la licencia MIT.

# AlmaBot - Asistente Virtual Anti-Bullying

AlmaBot es un asistente virtual basado en inteligencia artificial diseñado para brindar apoyo emocional a niños y niñas entre 6 y 14 años que enfrentan situaciones de acoso escolar.

## Características

- Interfaz de chat amigable y anónima
- Detección de bullying y emociones
- Recomendaciones de recursos
- Sistema de alertas para casos de riesgo
- Integración con Telegram para pruebas

## Requisitos Previos

- Docker y Docker Compose instalados
- Cuenta de Telegram para probar el bot (opcional)
- Al menos 4GB de RAM disponible
- Conexión a Internet para descargar dependencias

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/almabot.git
cd almabot
```

### 2. Configuración del entorno

1. Copiar el archivo de configuración:
   ```bash
   cp .env.example .env
   ```

2. Editar el archivo `.env` con tus credenciales:
   - Configurar las credenciales de la base de datos
   - Agregar el token de tu bot de Telegram (opcional)
   - Ajustar otros parámetros según sea necesario

## Ejecución del Proyecto

### 1. Iniciar los servicios básicos

```bash
# Construir e iniciar la base de datos y la API
docker compose up -d db api
```

### 2. Entrenar el modelo

Es necesario entrenar el modelo antes de usar el chatbot:

```bash
# Iniciar el servicio de entrenamiento
docker compose --profile train up training
```

**Nota:** El entrenamiento puede tardar varios minutos dependiendo de tu hardware.

### 3. Verificar el estado de los servicios

```bash
docker compose ps
```

Deberías ver los servicios `db` y `api` en estado "running".

### 4. Acceder a la API

- URL de la API: `http://localhost:8000`
- Documentación interactiva: `http://localhost:8000/docs`

### 5. (Opcional) Iniciar el bot de Telegram

Si configuraste el token de Telegram:

```bash
docker compose up -d telegram_bot
```

## Estructura del Proyecto

```
.
├── src/
│   ├── api/                 # Código de la API
│   ├── chatbot/             # Lógica del chatbot
│   ├── database/            # Esquema y migraciones
│   ├── models/              # Modelos entrenados
│   └── utils/               # Utilidades
├── data/                   # Datos de entrenamiento
├── docker/                 # Configuraciones de Docker
├── .env.example           # Plantilla de variables de entorno
└── docker-compose.yml     # Configuración de servicios
```

## Solución de Problemas

### Error: "No se encontró el modelo entrenado"

Asegúrate de haber ejecutado el entrenamiento:

```bash
docker compose --profile train up training
```

### Error de permisos de Docker

Si ves errores de permisos, ejecuta:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Reiniciar el sistema

Para reiniciar completamente los servicios:

```bash
docker compose down
docker compose up -d
```

## Base de Datos

### Acceso a la consola de MySQL

```bash
docker compose exec db mysql -ualmabot -palmabot123 almabot
```

### Consultas útiles

```sql
-- Ver tablas disponibles
SHOW TABLES;

-- Ver estructura de las tablas principales
DESCRIBE anonymous_users;
DESCRIBE therapy_sessions;
DESCRIBE messages;

-- Ver registros recientes
SELECT * FROM messages ORDER BY created_at DESC LIMIT 10;
```

## Contribución

1. Haz fork del repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Soporte

Para reportar problemas o solicitar ayuda, por favor abre un issue en el repositorio.
#!/bin/bash

# Cargar variables de entorno
DB_USER="almabot"
DB_PASSWORD="almabot123"
DB_NAME="almabot"

# Verificar conexión a la base de datos
sudo docker exec -i almabot-db mysql -u${DB_USER} -p${DB_PASSWORD} -e "SHOW DATABASES;"

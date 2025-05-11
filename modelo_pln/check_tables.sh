#!/bin/bash

# Cargar variables de entorno
DB_USER="almabot"
DB_PASSWORD="almabot123"
DB_NAME="almabot"

# Verificar tablas en la base de datos
sudo docker exec -i almabot-db mysql -u${DB_USER} -p${DB_PASSWORD} ${DB_NAME} -e "SHOW TABLES;"

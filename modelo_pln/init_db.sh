#!/bin/bash

# Wait for MySQL to be ready
until mysql -h db -u${DB_USER} -p${DB_PASSWORD} -e "SELECT 1"; do
    echo "Waiting for MySQL..."
    sleep 1
    if [ $((++count)) -gt 30 ]; then
        echo "MySQL took too long to start"
        exit 1
    fi
done

echo "Initializing database..."
mysql -h db -u${DB_USER} -p${DB_PASSWORD} ${DB_NAME} < /schema.sql

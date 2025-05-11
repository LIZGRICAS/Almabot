import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        try:
            self.conn = mysql.connector.connect(
                host=os.getenv('DB_HOST', 'db'),
                user=os.getenv('DB_USER', 'almabot'),
                password=os.getenv('DB_PASSWORD', 'almabot123'),
                database=os.getenv('DB_NAME', 'almabot'),
                port=int(os.getenv('DB_PORT', '3306')),
                connection_timeout=30
            )
            # No es necesario crear las tablas ya que se crean con el script de inicialización
            # self.crear_tablas()
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            print(f"Connection details: host={os.getenv('DB_HOST', 'db')}, user={os.getenv('DB_USER', 'almabot')}, database={os.getenv('DB_NAME', 'almabot')}")
            raise
    
    def crear_tablas(self):
        try:
            with open('src/database/schema.sql', 'r') as f:
                schema = f.read()
            cursor = self.conn.cursor()
            for statement in schema.split(';'):
                if statement.strip():
                    cursor.execute(statement)
            self.conn.commit()
        except Error as e:
            print(f"Error creating tables: {e}")
            raise 
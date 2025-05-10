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
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD', 'password'),
                database=os.getenv('DB_NAME', 'almabot')
            )
            self.crear_tablas()
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
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
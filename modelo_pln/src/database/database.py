import sqlite3
from datetime import datetime

class Database:
    def __init__(self, db_name="almabot.db"):
        self.conn = sqlite3.connect(db_name)
        self.crear_tablas()
        
    def crear_tablas(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sesiones (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                fecha_inicio TIMESTAMP,
                fecha_fin TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mensajes (
                id INTEGER PRIMARY KEY,
                sesion_id INTEGER,
                contenido TEXT,
                emisor TEXT,
                timestamp TIMESTAMP,
                bullying_detectado BOOLEAN,
                emocion TEXT,
                FOREIGN KEY (sesion_id) REFERENCES sesiones (id)
            )
        ''')
        self.conn.commit() 
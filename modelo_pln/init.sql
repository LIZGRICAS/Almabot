-- Crear base de datos
CREATE DATABASE IF NOT EXISTS almabot;
USE almabot;

-- Tabla de usuarios anónimos
CREATE TABLE IF NOT EXISTS anonymous_users (
    id VARCHAR(36) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    age_range VARCHAR(20),
    school_type VARCHAR(50),
    neighborhood VARCHAR(100)
);

-- Tabla de sesiones
CREATE TABLE IF NOT EXISTS therapy_sessions (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    status ENUM('active', 'closed') DEFAULT 'active',
    FOREIGN KEY (user_id) REFERENCES anonymous_users(id)
);

-- Tabla de mensajes
CREATE TABLE IF NOT EXISTS messages (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36),
    content TEXT NOT NULL,
    sender ENUM('user', 'bot') NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    emotion_detected VARCHAR(50),
    bullying_detected BOOLEAN DEFAULT FALSE,
    bullying_type VARCHAR(50),
    confidence_score FLOAT,
    FOREIGN KEY (session_id) REFERENCES therapy_sessions(id)
);

-- Tabla de recursos recomendados
CREATE TABLE IF NOT EXISTS recommended_resources (
    id VARCHAR(36) PRIMARY KEY,
    message_id VARCHAR(36),
    resource_type VARCHAR(50),
    resource_content TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

-- Tabla de análisis de sesión
CREATE TABLE IF NOT EXISTS session_analysis (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36),
    dominant_emotion VARCHAR(50),
    bullying_incidents INT DEFAULT 0,
    risk_level ENUM('low', 'medium', 'high') DEFAULT 'low',
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES therapy_sessions(id)
);

-- Índices
CREATE INDEX idx_user_sessions ON therapy_sessions(user_id);
CREATE INDEX idx_session_messages ON messages(session_id);
CREATE INDEX idx_message_resources ON recommended_resources(message_id);
CREATE INDEX idx_session_analysis ON session_analysis(session_id); 
-- Create database
CREATE DATABASE IF NOT EXISTS almabot;
USE almabot;

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id VARCHAR(36) PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    record_id VARCHAR(36) NOT NULL,
    action ENUM('INSERT', 'UPDATE', 'DELETE') NOT NULL,
    old_data JSON,
    new_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(36)
);

-- Users table (for admin purposes)
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin', 'moderator') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(36),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    updated_by VARCHAR(36),
    deleted_at TIMESTAMP NULL,
    deleted_by VARCHAR(36)
);

-- Anonymous users table
CREATE TABLE IF NOT EXISTS anonymous_users (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) UNIQUE,
    age INT,
    neighborhood VARCHAR(100),
    school VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(36),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    updated_by VARCHAR(36),
    deleted_at TIMESTAMP NULL,
    deleted_by VARCHAR(36),
    is_active BOOLEAN DEFAULT TRUE,
    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(36),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    updated_by VARCHAR(36),
    deleted_at TIMESTAMP NULL,
    deleted_by VARCHAR(36),
    FOREIGN KEY (user_id) REFERENCES anonymous_users(id)
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id VARCHAR(36) PRIMARY KEY,
    conversation_id VARCHAR(36),
    sender ENUM('user', 'bot') NOT NULL,
    content TEXT NOT NULL,
    sentiment_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(36),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    updated_by VARCHAR(36),
    deleted_at TIMESTAMP NULL,
    deleted_by VARCHAR(36),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Emotional States
CREATE TABLE IF NOT EXISTS emotional_states (
    id VARCHAR(36) PRIMARY KEY,
    message_id VARCHAR(36),
    emotion_type ENUM('happy', 'sad', 'angry', 'fear', 'neutral') NOT NULL,
    intensity FLOAT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

-- Risk Assessment
CREATE TABLE IF NOT EXISTS risk_assessment (
    id VARCHAR(36) PRIMARY KEY,
    message_id VARCHAR(36),
    risk_level ENUM('low', 'medium', 'high') NOT NULL,
    risk_type ENUM('bullying', 'suicide', 'abuse', 'other') NOT NULL,
    requires_attention BOOLEAN DEFAULT FALSE,
    reviewed_by VARCHAR(50),
    reviewed_at TIMESTAMP NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id),
    FOREIGN KEY (reviewed_by) REFERENCES users(username)
);

-- Resources and Recommendations
CREATE TABLE IF NOT EXISTS resources (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    description TEXT,
    type ENUM('article', 'video', 'activity', 'emergency_contact') NOT NULL,
    url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Recommendations per message
CREATE TABLE IF NOT EXISTS message_resources (
    id VARCHAR(36) PRIMARY KEY,
    message_id VARCHAR(36),
    resource_id VARCHAR(36),
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES messages(id),
    FOREIGN KEY (resource_id) REFERENCES resources(id)
);

-- Session Analysis
CREATE TABLE IF NOT EXISTS session_analysis (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36),
    dominant_emotion VARCHAR(50),
    bullying_incidents INT DEFAULT 0,
    risk_level ENUM('low', 'medium', 'high') DEFAULT 'low',
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES conversations(id)
);

-- Create indexes for better performance
CREATE INDEX idx_conversations_active ON conversations(is_active);
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_user_sessions_last_active ON anonymous_users(last_interaction);
CREATE INDEX idx_risk_assessment_level ON risk_assessment(risk_level);
CREATE INDEX idx_session_analysis ON session_analysis(session_id);

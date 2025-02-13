"""
SQLite database access layer for conversation management.
"""
import sqlite3
from contextlib import contextmanager
import logging
from typing import Any, List, Dict, Optional, Union
from datetime import datetime
import json
import os
from pathlib import Path
import uuid
from app.models.conversation import Message, MessageRole

logger = logging.getLogger(__name__)

# Get database path from environment or use default
DB_PATH = os.getenv('SQLITE_DB_PATH', 'conversations.db')

# Ensure database directory exists
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# SQL Statements for creating tables
CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_conversations_id ON conversations(id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_active ON conversations(active);
"""

def init_db():
    """Initialize database schema."""
    try:
        # Ensure we have a fresh connection for initialization
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript(CREATE_TABLES)
            conn.commit()
            logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def get_db():
    """Get a database connection."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        raise

@contextmanager
def get_db_connection():
    """Context manager for database connections with proper error handling."""
    conn = None
    try:
        conn = get_db()
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

class ConversationDB:
    """Handles conversation database operations."""
    
    @staticmethod
    def save_conversation(conversation_id: str, messages: List[Union[Message, Dict[str, Any]]], 
                         created_at: datetime, updated_at: datetime, active: bool = True) -> None:
        """Save or update a conversation with its messages."""
        try:
            with get_db_connection() as conn:
                # Start transaction
                conn.execute("BEGIN")
                try:
                    # Insert or update conversation
                    conn.execute("""
                        INSERT OR REPLACE INTO conversations (id, created_at, updated_at, active)
                        VALUES (?, ?, ?, ?)
                    """, (conversation_id, created_at.isoformat(), updated_at.isoformat(), int(active)))
                    
                    # Delete existing messages for this conversation
                    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                    
                    # Insert new messages
                    for msg in messages:
                        if isinstance(msg, Message):
                            # If it's a Message object, use its attributes directly
                            message_data = {
                                "message_id": msg.message_id,
                                "role": msg.role.value if isinstance(msg.role, MessageRole) else msg.role,
                                "content": msg.content,
                                "timestamp": msg.timestamp.isoformat() if isinstance(msg.timestamp, datetime) else msg.timestamp
                            }
                        else:
                            # If it's a dictionary, use it as is
                            message_data = msg
                            
                        conn.execute("""
                            INSERT INTO messages (message_id, conversation_id, role, content, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            message_data.get("message_id", str(uuid.uuid4())),
                            conversation_id,
                            message_data.get("role"),
                            message_data.get("content"),
                            message_data.get("timestamp", datetime.utcnow().isoformat())
                        ))
                    
                    conn.commit()
                    logger.info(f"Saved conversation {conversation_id} with {len(messages)} messages")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to save conversation {conversation_id}: {e}")
                    raise
        except Exception as e:
            logger.error(f"Database error while saving conversation: {e}")
            raise

    @staticmethod
    def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a conversation by ID with all its messages."""
        try:
            with get_db_connection() as conn:
                # Get conversation details
                conv_row = conn.execute(
                    "SELECT * FROM conversations WHERE id = ?", 
                    (conversation_id,)
                ).fetchone()
                
                if not conv_row:
                    return None
                
                # Get all messages for this conversation
                messages = conn.execute("""
                    SELECT message_id, role, content, timestamp 
                    FROM messages 
                    WHERE conversation_id = ?
                    ORDER BY timestamp
                """, (conversation_id,)).fetchall()
                
                return {
                    "id": conv_row["id"],
                    "created_at": conv_row["created_at"],
                    "updated_at": conv_row["updated_at"],
                    "active": bool(conv_row["active"]),
                    "messages": [dict(msg) for msg in messages]
                }
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            raise

    @staticmethod
    def delete_conversation(conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        try:
            with get_db_connection() as conn:
                conn.execute("BEGIN")
                try:
                    # Messages will be deleted automatically due to CASCADE
                    result = conn.execute(
                        "DELETE FROM conversations WHERE id = ?",
                        (conversation_id,)
                    )
                    deleted = result.rowcount > 0
                    conn.commit()
                    if deleted:
                        logger.info(f"Deleted conversation {conversation_id}")
                    return deleted
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to delete conversation {conversation_id}: {e}")
                    raise
        except Exception as e:
            logger.error(f"Database error while deleting conversation: {e}")
            raise

    @staticmethod
    def deactivate_conversation(conversation_id: str) -> bool:
        """Mark a conversation as inactive."""
        try:
            with get_db_connection() as conn:
                result = conn.execute("""
                    UPDATE conversations 
                    SET active = 0, updated_at = ? 
                    WHERE id = ?
                """, (datetime.utcnow().isoformat(), conversation_id))
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error deactivating conversation {conversation_id}: {e}")
            raise

# Initialize database on module import
init_db()

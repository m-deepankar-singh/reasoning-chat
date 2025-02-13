"""
Database dependencies and connection management.
"""
from fastapi import Depends
from app.db.sqlite_db import get_db_connection, init_db

# Initialize database
init_db()

def get_db():
    """FastAPI dependency for database connection."""
    with get_db_connection() as conn:
        yield conn

# Re-export the database dependency
__all__ = ['get_db']

"""
Base model utilities.
"""
from typing import Any, Dict

class BaseModel:
    """Base class for all models."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        return {
            key: getattr(self, key)
            for key in self.__dict__
            if not key.startswith('_')
        }

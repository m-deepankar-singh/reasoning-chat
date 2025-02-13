"""
Conversation models for managing chat interactions.
"""
from typing import Dict, List, Optional, Union
import uuid
from datetime import datetime
from enum import Enum

from fastapi import logger

class MessageRole(str, Enum):
    """Enum for message roles in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    @classmethod
    def from_str(cls, value: str) -> 'MessageRole':
        """Create MessageRole from string, with proper error handling."""
        try:
            return cls(value.lower())
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid message role: {value}")

class Message:
    """Represents a single message in a conversation."""
    def __init__(self, role: Union[MessageRole, str], content: str, 
                 message_id: Optional[str] = None, timestamp: Optional[Union[datetime, str]] = None):
        self.role = MessageRole.from_str(role) if isinstance(role, str) else role
        # Ensure content is converted to string
        if isinstance(content, (list, tuple, set, dict)):
            self.content = str(content)
        elif hasattr(content, '__iter__') and not isinstance(content, str):
            # Handle generator case
            self.content = "".join(str(x) for x in content)
        else:
            self.content = str(content)
        self.message_id = message_id or str(uuid.uuid4())
        if isinstance(timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                self.timestamp = datetime.utcnow()
        else:
            self.timestamp = timestamp or datetime.utcnow()

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Message':
        """Create a Message instance from a dictionary."""
        try:
            return cls(
                role=data["role"],
                content=data["content"],
                message_id=data.get("message_id"),
                timestamp=data.get("timestamp")
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in message data: {e}")

    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary format."""
        return {
            "message_id": self.message_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

class Conversation:
    """Manages a conversation session with message history."""
    def __init__(self, conversation_id: Optional[str] = None):
        self.id = conversation_id or str(uuid.uuid4())
        self.messages: List[Message] = []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self._active = True

    def add_message(self, role: Union[MessageRole, str], content: str) -> Message:
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender (user/assistant/system)
            content: The content of the message
            
        Returns:
            Message: The created message object
        """
        if not self._active:
            raise ValueError("Cannot add message to an inactive conversation")
            
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the conversation in dictionary format."""
        return [msg.to_dict() for msg in self.messages]

    def get_message_history(self) -> List[Dict[str, str]]:
        """Get simplified message history for model context."""
        return [{"role": msg.role.value, "content": msg.content} 
                for msg in self.messages]

    def deactivate(self) -> None:
        """Mark conversation as inactive/archived."""
        self._active = False
        self.updated_at = datetime.utcnow()

    def is_active(self) -> bool:
        """Check if conversation is active."""
        return self._active

    @classmethod
    def from_dict(cls, data: Dict) -> 'Conversation':
        """Create a Conversation instance from a dictionary."""
        try:
            conv = cls(conversation_id=data["id"])
            conv.created_at = datetime.fromisoformat(data["created_at"])
            conv.updated_at = datetime.fromisoformat(data["updated_at"])
            conv._active = data.get("active", True)
            
            # Convert message dictionaries to Message objects
            for msg_data in data.get("messages", []):
                try:
                    msg = Message.from_dict(msg_data)
                    conv.messages.append(msg)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid message in conversation {conv.id}: {e}")
                    continue
                    
            return conv
        except KeyError as e:
            raise ValueError(f"Missing required field in conversation data: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid conversation data: {e}")

    def to_dict(self) -> Dict:
        """Convert conversation to dictionary format."""
        return {
            "id": self.id,
            "messages": self.get_messages(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "active": self._active
        }

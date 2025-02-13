# Database Revision Plan for Conversation Logs

## Overview
This document outlines the database revision plan for the conversation logging system. The main goal is to create a more robust and focused database structure that exclusively handles conversation-related data.

## Schema Design

### Conversations Table
```sql
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    active INTEGER DEFAULT 1
);
```

### Messages Table
```sql
CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
```

### Indexes
- `idx_conversations_id`: Index on conversations(id)
- `idx_messages_conversation`: Index on messages(conversation_id)
- `idx_conversations_active`: Index on conversations(active)

## Key Changes

1. **Schema Separation**
   - Split the conversation data into two tables: `conversations` and `messages`
   - Each message now has its own record with a unique ID
   - Added foreign key constraints for data integrity

2. **New Features**
   - Added conversation activity tracking
   - Improved message timestamps and ordering
   - Better support for message roles (user/assistant/system)

3. **Performance Improvements**
   - Added indexes for faster querying
   - Optimized table structure for common operations
   - Implemented proper transaction handling

## Data Model

### Conversation
```python
{
    "id": str,                    # Unique conversation identifier
    "created_at": datetime,       # Creation timestamp
    "updated_at": datetime,       # Last update timestamp
    "active": bool,               # Conversation status
    "messages": List[Message]     # List of messages
}
```

### Message
```python
{
    "message_id": str,            # Unique message identifier
    "role": MessageRole,          # Message sender role (user/assistant/system)
    "content": str,               # Message content
    "timestamp": datetime         # Message timestamp
}
```

## Database Operations

### Key Functions
1. `save_conversation`: Save or update a conversation with its messages
2. `get_conversation`: Retrieve a conversation by ID with all its messages
3. `delete_conversation`: Delete a conversation and all its messages
4. `deactivate_conversation`: Mark a conversation as inactive

### Error Handling
- Improved error logging and handling
- Transaction management for data consistency
- Proper connection cleanup

## Migration Notes
1. The new schema is automatically created when the application starts
2. Existing data will need to be migrated manually if needed
3. The system uses SQLite with foreign key support enabled

## Best Practices
1. Always use the provided database connection context manager
2. Handle all database operations within transactions
3. Log all significant database operations and errors
4. Use proper parameter binding to prevent SQL injection

## Security Considerations
1. Input validation at the model level
2. Proper error handling without exposing internal details
3. Use of prepared statements for all database operations

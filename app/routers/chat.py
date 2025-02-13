"""
Chat and text generation endpoints.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.model_manager import model_manager
from app.services.reasoning_engine import reasoning_engine
from app.services.document_context import document_context_service
from app.models.conversation import Conversation
from app.db.sqlite_db import ConversationDB
from app.core.config import get_settings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Global state
conversations: Dict[str, Conversation] = {}
settings = get_settings()

@router.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Handle chat completion requests."""
    try:
        # Get or create conversation
        conversation = conversations.get(request.conversation_id)
        if not conversation:
            conversation = Conversation()
            conversations[conversation.id] = conversation
        
        # Add user message
        conversation.add_message("user", request.prompt)
        logger.info(f"Processing chat request with prompt: {request.prompt[:100]}...")
        
        # Initialize response
        response = None
        
        # Process document context if enabled
        if request.enable_doc_context:
            logger.info("Document context enabled, attempting to process markdown documents")
            try:
                # Try processing markdown documents first
                response = await document_context_service.process_markdown_documents(request.prompt)
                if response:
                    logger.info("Successfully processed markdown documents")
                else:
                    logger.info("No markdown response, trying raw documents")
                    response = await document_context_service.process_raw_documents(request.prompt)
                
                if response:
                    logger.info("Successfully processed document context")
                    conversation.add_message("assistant", response)
                    return ChatResponse(
                        content=response,
                        conversation_id=conversation.id
                    )
                else:
                    logger.warning("No document context available, falling back to regular chat")
            except Exception as e:
                logger.error(f"Error processing document context: {e}")
                # Continue without document context if there's an error
        
        # If no document context or it failed, use regular chat
        logger.info("Using regular chat processing")
        
        content = None
        reasoning = None
        
        # Try deepseek-reasoner if enabled
        if request.enable_reasoning:
            try:
                client = model_manager.get_client("deepseek")
                if client:
                    logger.debug("Attempting to use deepseek-reasoner")
                    messages = [
                        {"role": "system", "content": """You are an expert at reasoning and analysis. 
                        When provided with document context and a question, carefully analyze the context to provide accurate and relevant responses.
                        Break down complex problems step by step and ensure your response directly addresses the query using the provided context."""},
                        {"role": "user", "content": request.prompt}
                    ]
                    
                    logger.debug(f"Sending request to deepseek-reasoner with messages: {messages}")
                    response = client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=messages,
                        temperature=float(request.temperature),
                        max_tokens=500
                    )
                    logger.debug(f"Received raw response from deepseek: {response}")
                    
                    # Extract reasoning and generate final output
                    try:
                        if not response or not response.choices:
                            logger.error("Empty response or no choices from deepseek")
                            raise ValueError("Invalid response from deepseek")
                            
                        message = response.choices[0].message
                        if not message:
                            logger.error("No message in response")
                            raise ValueError("No message in response")
                            
                        reasoning = message.reasoning_content
                        if not reasoning:
                            logger.warning("Empty reasoning received")
                            raise ValueError("Empty reasoning")
                        
                        logger.debug(f"Successfully extracted reasoning: {reasoning[:100]}...")
                        
                        # Use local LLM to generate final output from reasoning
                        logger.debug("Generating final output using local LLM")
                        content = model_manager.llm(reasoning)
                        if not content:
                            logger.warning("Empty content generated by local LLM")
                            raise ValueError("Empty content generated")
                            
                        logger.debug(f"Successfully generated content: {content[:100]}...")
                            
                    except (AttributeError, IndexError, ValueError) as e:
                        logger.error(f"Error processing deepseek response: {str(e)}")
                        reasoning = None
                        content = None
            except Exception as e:
                logger.error(f"Error using deepseek-reasoner: {str(e)}")
                if hasattr(e, 'response'):
                    logger.error(f"API Response: {e.response}")
                reasoning = None
                content = None
        
        # If reasoning failed or was disabled, fall back to regular processing
        if not content:
            content = await reasoning_engine.get_reasoning(
                prompt=request.prompt,
                enable_reasoning=False,  # Force regular processing
                max_tokens=int(request.max_tokens) if request.max_tokens else 500,
                temperature=float(request.temperature) if request.temperature else 0.3
            )
        
        # Add response to conversation
        conversation.add_message("assistant", content)
        logger.info("Chat request processed successfully")
        
        # Save conversation to database
        ConversationDB.save_conversation(
            conversation_id=conversation.id,
            messages=conversation.messages,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return ChatResponse(
            content=content,
            conversation_id=conversation.id,
            reasoning=reasoning
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )

@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    # Try memory first
    conversation = conversations.get(conversation_id)
    if conversation:
        return {"messages": conversation.messages}
    
    # Try database
    db_conversation = ConversationDB.get_conversation(conversation_id)
    if db_conversation:
        return {"messages": db_conversation["messages"]}
    
    raise HTTPException(status_code=404, detail="Conversation not found")

@router.get("/test-llm")
async def test_local_llm():
    """Test if the local LLM is working."""
    try:
        test_prompt = "Say 'Hello! I am working correctly!' if you can read this message."
        response = await reasoning_engine.get_reasoning(
            prompt=test_prompt,
            enable_reasoning=False,
            max_tokens=50,
            temperature=0.7
        )
        return {"status": "success", "response": response}
    except Exception as e:
        logger.error(f"Error testing local LLM: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Local LLM test failed: {str(e)}"
        )

@router.get("/test-deepseek")
async def test_deepseek():
    """Test if the Deepseek API is working."""
    try:
        if not model_manager.clients["deepseek"]:
            raise HTTPException(
                status_code=500,
                detail="Deepseek client not initialized. Make sure DEEPSEEK_API_KEY is set in environment variables."
            )
            
        response = model_manager.clients["deepseek"].chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "Say 'Hello! Deepseek API is working correctly!' if you can read this message."}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        return {
            "status": "success", 
            "response": response.choices[0].message.content if response.choices else "No response generated"
        }
    except Exception as e:
        logger.error(f"Error testing Deepseek API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Deepseek API test failed: {str(e)}"
        )
        
# @router.get("/test-deepseek-reasoner")
# async def test_deepseek_reasoner():
#    """Test if the Deepseek Reasoner API is working."""
#    try:
#        if not model_manager.clients["deepseek"]:
#            raise HTTPException(
#                status_code=500,
#                detail="Deepseek client not initialized. Make sure DEEPSEEK_API_KEY is set in environment variables."
#            )
       
#        response = model_manager.clients["deepseek"].chat.completions.create(
#            model="deepseek-reasoner",
#            messages=[
#                {"role": "user", "content": "Say 'Hello! Deepseek Reasoner API is working correctly!' if you can read this message."}
#            ],
#            max_tokens=50,
#            temperature=0.7
#        )
#        message_content = response.choices[0].message.content if response.choices else "No response generated"
#        reasoning_content = response.choices[0].message.reasoning_content if response.choices and hasattr(response.choices[0].message, 'reasoning_content') else "No reasoning provided"
#        return {
#            "status": "success",
#            "response": message_content,
#            "reasoning": reasoning_content
#        }
#    except Exception as e:
#        logger.error(f"Error testing Deepseek Reasoner API: {str(e)}")
#        raise HTTPException(
#            status_code=500,
#            detail=f"Deepseek Reasoner API test failed: {str(e)}"
#        )

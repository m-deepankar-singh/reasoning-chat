"""
Reasoning engine for processing prompts.
"""
from typing import List, Optional
import logging
from app.services.model_manager import model_manager
from app.core.config import get_settings
import google.generativeai as genai

settings = get_settings()
logger = logging.getLogger(__name__)

class ReasoningEngine:
    """Handles reasoning and processing of prompts."""
    
    @staticmethod
    async def get_reasoning(
        prompt: str, 
        enable_reasoning: bool = False,
        images: Optional[List[str]] = None,
        max_tokens: int = 150,
        temperature: float = 0.2,
        context: Optional[str] = None
    ) -> str:
        """Get reasoning for a given prompt."""
        try:
            deepseek_client = model_manager.get_client("deepseek")
            if deepseek_client:
                logger.debug("Using _process_with_deepseek_chat branch")
                return await ReasoningEngine._process_with_deepseek_chat(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            elif model_manager.get_client("gemini") and images:
                logger.debug("Using _process_with_gemini_thinking branch")
                return await ReasoningEngine._process_with_gemini_thinking(prompt, images)
            else:
                logger.debug("No suitable client available; falling back to _process_with_fallback")
                return await ReasoningEngine._process_with_fallback(prompt)
        except Exception as e:
            logger.error(f"Error in reasoning engine: {str(e)}")
            return await ReasoningEngine._process_with_fallback(prompt)

    @staticmethod
    async def _process_with_deepseek_chat(prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
        """Process prompt with DeepSeek Chat model."""
        try:
            client = model_manager.get_client("deepseek")
            if not client:
                raise ValueError("DeepSeek client not available")
            
            logger.debug("Entering _process_with_deepseek_chat with prompt: " + prompt[:50])
                 
            logger.debug("Sending request to DeepSeek Chat API")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.debug("Received response from DeepSeek Chat API")
            content = response.choices[0].message.content
            return content
        except Exception as e:
            logger.error(f"DeepSeek Chat processing error: {str(e)}")
            raise

    @staticmethod
    async def _process_with_gemini_thinking(prompt: str, images: List[str]) -> str:
        """Process prompt with Gemini model."""
        try:
            client = model_manager.get_client("gemini")
            if not client:
                raise ValueError("Gemini client not available")
            
            logger.debug("Entering _process_with_gemini_thinking with prompt: " + prompt[:50] + " and images: " + str(images))
            if not images:
                response = client.generate_content(prompt)
            else:
                response = client.generate_content(prompt, images)
            
            # Handle different response types
            if hasattr(response, 'text'):
                text = response.text
                if isinstance(text, (list, tuple, set)):
                    final_text = "".join(str(x) for x in text)
                elif hasattr(text, '__iter__') and not isinstance(text, str):
                    # Handle generator case
                    final_text = "".join(str(x) for x in text)
                else:
                    final_text = str(text)
            else:
                final_text = str(response)
                
            logger.debug("Exiting _process_with_gemini_thinking with response: " + final_text[:50])
            return final_text
        except Exception as e:
            logger.error(f"Gemini processing error: {str(e)}")
            raise

    @staticmethod
    async def _process_with_fallback(prompt: str) -> str:
        """Fallback processing using Gemini model."""
        logger.debug("Entering _process_with_fallback with prompt: " + prompt[:50])
        try:
            model_name = settings.GEMINI_MODELS["reasoning"] if settings.ENABLE_REASONING else settings.GEMINI_MODELS["default"]
            model = genai.GenerativeModel(model_name)
            if not model:
                raise ValueError("Gemini client not available for fallback")
            response = model.generate_content(prompt)
            logger.debug("Received response in fallback: " + response.text[:50])
            return response.text
        except Exception as e:
            logger.error(f"Fallback processing error: {str(e)}")
            return "I apologize, but I'm unable to process your request at the moment."

# Global instance
reasoning_engine = ReasoningEngine()

"""
Chat interface for LORE.

This module provides functionality for chatting with a repository using the Llama API.
"""
import logging
import json
from typing import List, Dict, Any, Optional

from ..llm.llama_client import LlamaClient
from ..analysis.analyzer import RepositoryAnalyzer

logger = logging.getLogger(__name__)

class ChatHistory:
    """Manage chat history for the repository conversation."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize chat history.
        
        Args:
            max_history: Maximum number of messages to store in history
        """
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the chat history.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content
        })
        
        # Trim history if it exceeds max_history
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the chat history.
        
        Returns:
            List of message dictionaries
        """
        return self.messages.copy()
    
    def clear(self):
        """Clear the chat history."""
        self.messages = []


class RepoChat:
    """Chat interface for interacting with repository analysis."""
    
    def __init__(self, llm_client: LlamaClient, analyzer: RepositoryAnalyzer, repo_context: str):
        """
        Initialize the repository chat interface.
        
        Args:
            llm_client: Llama API client
            analyzer: Repository analyzer
            repo_context: Repository context for analysis
        """
        self.llm_client = llm_client
        self.analyzer = analyzer
        self.repo_context = repo_context
        self.chat_history = ChatHistory()
        
        # Initialize with system prompt
        self.system_prompt = (
            "You are LORE (Long-context Organizational Repository Explorer), an expert AI assistant "
            "that helps developers understand codebases. You have been provided with the repository context. "
            "Provide clear, concise, and accurate answers. Keep responses focused and relevant to the "
            "questions asked. If you don't know something, say so directly. End your response when you've "
            "fully answered the question."
        )
    
    def chat(self, user_message: str, stream: bool = False) -> str:
        """
        Send a message to the chat interface and get a response.
        
        Args:
            user_message: Message from the user
            stream: Whether to stream the response
            
        Returns:
            Response from the assistant
        """
        try:
            # Construct messages array with proper schema
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            
            # Add context message if we have repository context
            if self.repo_context:
                messages.insert(1, {
                    "role": "system",
                    "content": f"Repository context:\n{self.repo_context}"
                })

            logger.debug(f"Sending chat request with messages: {json.dumps(messages, indent=2)}")
            
            response = self.llm_client.analyze_repository(
                content="",  # Not used when providing messages
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                temperature=0.2,  # Lower temperature for more focused responses
                max_tokens=1000,  # Limit response length
                task="chat",  # Custom task for chat
                messages=messages  # Pass the messages
            )
            
            logger.debug(f"Response received: {json.dumps(response, indent=2)}")
            
            # Handle Llama API response format
            if 'completion_message' in response:
                completion = response['completion_message']
                if 'content' in completion:
                    content = completion['content']
                    if isinstance(content, dict) and 'text' in content:
                        assistant_message = content['text']
                        logger.debug(f"Found message in Llama format: {assistant_message}")
                    else:
                        assistant_message = str(content)
                        logger.debug(f"Found raw content: {assistant_message}")
                else:
                    logger.error(f"No content in completion message: {completion}")
                    assistant_message = "I'm sorry, I couldn't generate a response. The API response was missing content."
            # Fallback to OpenAI format
            elif 'choices' in response and len(response['choices']) > 0:
                choice = response['choices'][0]
                logger.debug(f"First choice: {json.dumps(choice, indent=2)}")
                
                if 'message' in choice and 'content' in choice['message']:
                    assistant_message = choice['message']['content']
                    logger.debug(f"Found message in OpenAI format: {assistant_message}")
                elif 'text' in choice:
                    assistant_message = choice['text']
                    logger.debug(f"Found message in text format: {assistant_message}")
                elif 'content' in choice:
                    assistant_message = choice['content']
                    logger.debug(f"Found message in content format: {assistant_message}")
                else:
                    logger.error(f"Unexpected choice format. Available keys: {list(choice.keys())}")
                    assistant_message = "I'm sorry, I couldn't generate a response. The API response format was unexpected."
            else:
                logger.error(f"No completion_message or choices in response. Full response: {response}")
                assistant_message = "I'm sorry, I couldn't generate a response. The API response was missing the expected data."
            
            # Add assistant message to history
            self.chat_history.add_message("assistant", assistant_message)
            
            return assistant_message
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_message = f"I encountered an error: {str(e)}"
            self.chat_history.add_message("assistant", error_message)
            return error_message

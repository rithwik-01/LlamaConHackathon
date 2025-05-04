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
            "\n\nWhen analyzing commits, PRs, or code changes:\n"
            "1. ALWAYS list ALL files that were modified in the commit/PR - MAKE SURE YOU LIST EVERY SINGLE FILE\n"
            "2. Group files by their directory structure\n"
            "3. For each file show:\n"
            "   - Full file path\n"
            "   - Number of lines added and removed\n"
            "   - A brief description of what changed\n"
            "4. Include the commit message and any PR description\n"
            "5. If multiple commits are involved, show the changes from each commit\n"
            "6. IMPORTANT: Double-check that you've actually listed ALL files mentioned in the context\n"
            "7. DO NOT assume files are unchanged or skip listing them - explicitly list every file that was changed\n"
            "\nProvide clear, concise, and accurate answers. If you don't know something, say so directly."
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
                }
            ]
            
            # Add repository context with better structure
            if self.repo_context:
                messages.append({
                    "role": "system",
                    "content": (
                        "Repository Analysis Context:\n"
                        "Here is the detailed repository context including all commits, "
                        "file changes, and documentation. Pay special attention to the "
                        "RECENT SIGNIFICANT COMMITS section which shows all file changes:\n\n"
                        f"{self.repo_context}"
                    )
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            logger.debug(f"Sending chat request with messages: {json.dumps(messages, indent=2)}")
            
            # Use the best model with maximum context and tokens
            response = self.llm_client.analyze_repository(
                content="",  # Not used when providing messages
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",  # Best model for code analysis
                temperature=0.2,  # Lower temperature for more focused responses
                max_tokens=10000,  # Maximum token limit
                task="chat",
                messages=messages
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

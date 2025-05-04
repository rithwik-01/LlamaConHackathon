"""
Chat interface for LORE.

This module provides functionality for chatting with a repository using the Llama API.
"""
import logging
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
            "that helps developers understand codebases. You have been provided with the full context "
            "of a repository including its code, Git history, documentation, and potentially issues and PRs. "
            "Answer questions about the codebase based on this context. If you don't know the answer, say so."
        )
    
    def chat(self, user_message: str, stream: bool = False) -> str:
        """
        Send a message to the chat interface and get a response.
        
        Args:
            user_message: User message
            stream: Whether to stream the response
            
        Returns:
            Assistant response
        """
        # Add user message to history
        self.chat_history.add_message("user", user_message)
        
        # Prepare messages for the API call
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add repository context as a system message
        # We'll use a truncated version to avoid exceeding token limits
        truncated_context = self.repo_context[:500000]  # 500K chars, adjust based on model
        messages.append({
            "role": "system", 
            "content": f"Repository context:\n\n{truncated_context}"
        })
        
        # Add chat history
        messages.extend(self.chat_history.get_history()[:-1])  # Exclude the most recent user message
        
        # Add the latest user message with explicit instruction
        messages.append({
            "role": "user",
            "content": f"Based on the repository information provided, {user_message}"
        })
        
        try:
            if stream:
                response = self.llm_client.stream_analysis(
                    content="",  # Not used when providing messages
                    model="llama-4-10m",
                    temperature=0.7,
                    max_tokens=2000,
                    task="chat"  # Custom task for chat
                )
                assistant_message = response.get("content", "")
            else:
                response = self.llm_client.analyze_repository(
                    content="",  # Not used when providing messages
                    model="llama-4-10m",
                    temperature=0.7,
                    max_tokens=2000,
                    task="chat"  # Custom task for chat
                )
                
                if 'choices' in response and len(response['choices']) > 0:
                    assistant_message = response['choices'][0]['message']['content']
                else:
                    assistant_message = "I'm sorry, I couldn't generate a response."
            
            # Add assistant message to history
            self.chat_history.add_message("assistant", assistant_message)
            
            return assistant_message
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_message = f"I encountered an error: {str(e)}"
            self.chat_history.add_message("assistant", error_message)
            return error_message

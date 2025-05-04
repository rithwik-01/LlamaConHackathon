"""
Chat interface for LORE.

This module provides functionality for chatting with a repository using the Llama API.
"""
import logging
import json
from typing import List, Dict, Any, Optional
import copy
from pathlib import Path

from ..llm.llama_client import LlamaClient
from ..analysis.analyzer import RepositoryAnalyzer
from ..utils.image_utils import resize_image_if_needed, encode_image_to_base64

logger = logging.getLogger(__name__)

class ChatHistory:
    """Manage chat history for the repository conversation."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize chat history.
        
        Args:
            max_history: Maximum number of messages to store in history
        """
        self.messages: List[Dict[str, Any]] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: Any):
        """
        Add a message to the chat history.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content (string, dict, or list)
        """
        # Ensure content is properly formatted for the API
        if isinstance(content, dict) and "type" in content and content["type"] == "text" and "text" in content:
            # Handle content that's already in the structured format
            # For API compatibility, if content is structured, convert to plain text
            # as structured content shouldn't be saved in history directly
            content = content["text"]
        
        self.messages.append({
            "role": role,
            "content": content
        })
        
        # Trim history if it exceeds max_history
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get the chat history.
        
        Returns:
            List of message dictionaries
        """
        # Make a deep copy to prevent modification of the original messages
        return copy.deepcopy(self.messages)
    
    def clear(self):
        """Clear the chat history."""
        self.messages = []


class RepoChat:
    """Chat interface for interacting with repository analysis."""
    
    def __init__(self, llm_client: Any, repository_context: str, history: Optional[ChatHistory] = None):
        """
        Initialize the repository chat interface.
        
        Args:
            llm_client: LLM client to use
            repository_context: Context of the repository
            history: Chat history, if available
        """
        self.llm_client = llm_client
        self.repository_context = repository_context
        self.history = history or ChatHistory()
        self.first_message = True
        self.has_design_diagram = False
        # Store multiple diagrams
        self.design_diagrams = []
        
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
            "\n"
            "For ALL responses:\n"
            "1. ALWAYS cite your sources of information at the end of your response\n"
            "2. ONLY mention sources that were actually provided to you:\n"
            "   - Repository context (only if used)\n"
            "   - Design diagrams or images (only if provided AND used)\n"
            "   - Specific files, file paths, or code snippets referenced (list them precisely)\n"
            "3. DO NOT mention hypothetical sources - if you don't have access to something, don't list it as a source\n"
            "4. DO NOT use phrases like 'not provided' or 'not available' in source citations\n"
            "5. Format your responses clearly with Markdown for headings, code blocks, and lists\n"
            "6. Provide clear, concise answers focused on high-level insights\n"
            "7. Only show code when specifically asked to do so\n"
        )
        
        # Add repository context to the history
        if repository_context:
            self.history.add_message("system", self.system_prompt)
            self.history.add_message("system", repository_context)
    
    def set_design_diagram(self, exists: bool = False):
        """Set whether design diagrams are available."""
        self.has_design_diagram = exists
    
    def add_design_diagram(self, diagram_path: str):
        """Add a design diagram to the chat context.
        
        Args:
            diagram_path: Path to the design diagram
        """
        if diagram_path and Path(diagram_path).exists():
            self.design_diagrams.append(diagram_path)
            self.has_design_diagram = True
    
    def chat(self, user_message: str, design_diagram_path: Optional[str] = None, stream: bool = False):
        """
        Send a message to the chat interface and get a response.
        
        Args:
            user_message: Message from the user
            design_diagram_path: Path to design diagram, if any
            stream: Whether to stream the response
            
        Returns:
            Response from the LLM
        """
        try:
            # Add user message to history
            self.history.add_message("user", user_message)
            
            # Prepare messages for the API
            messages = self.history.get_messages()
            
            # Add design diagram if available and this is the first message or explicitly requested
            if (design_diagram_path or self.design_diagrams) and self.has_design_diagram and ("diagram" in user_message.lower() or "image" in user_message.lower() or self.first_message):
                # If a new diagram path is provided, add it to our collection
                if design_diagram_path and design_diagram_path not in self.design_diagrams:
                    self.design_diagrams.append(design_diagram_path)
                
                # Use the first diagram by default or a specific one if path is provided
                current_diagram = design_diagram_path if design_diagram_path else self.design_diagrams[0]
                
                try:
                    # Process and encode the diagram
                    if current_diagram and Path(current_diagram).exists():
                        # Resize image if needed and encode to base64
                        encoded_image = encode_image_to_base64(current_diagram)
                        
                        # Create an image message in the Llama format
                        image_url = {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                            "detail": "high"
                        }
                        
                        # Insert image message before user's latest message
                        image_message = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": image_url
                                }
                            ]
                        }
                        
                        # Find the index of the last user message
                        last_user_idx = None
                        for i, msg in enumerate(messages):
                            if msg["role"] == "user":
                                last_user_idx = i
                        
                        if last_user_idx is not None:
                            # Insert the image message before the last user message
                            messages.insert(last_user_idx, image_message)
                            
                            # If this is the first message, also add a reminder
                            if self.first_message:
                                instruction_msg = {
                                    "role": "user",
                                    "content": "This is a design diagram for the repository. Please refer to it when answering questions about the architecture."
                                }
                                messages.insert(last_user_idx + 1, instruction_msg)
                except Exception as e:
                    logger.error(f"Error processing design diagram: {e}")
                    # If there's an error, continue without the image
                    pass
            
            # Get response from LLM
            assistant_message = ""
            
            if stream:
                # Streaming response
                for chunk in self.llm_client.stream_analysis(None, messages=messages):
                    if isinstance(chunk, str):
                        assistant_message += chunk
                    elif isinstance(chunk, dict):
                        if "content" in chunk:
                            assistant_message += chunk["content"]
                        elif "text" in chunk:
                            assistant_message += chunk["text"]
            else:
                # Normal (non-streaming) response
                response = self.llm_client.chat(messages)
                
                # Extract assistant message
                if isinstance(response, dict) and "choices" in response:
                    try:
                        # Handle different response formats
                        if "message" in response["choices"][0]:
                            assistant_message = response["choices"][0]["message"]["content"]
                        elif "text" in response["choices"][0]:
                            assistant_message = response["choices"][0]["text"]
                        elif "delta" in response["choices"][0] and "content" in response["choices"][0]["delta"]:
                            assistant_message = response["choices"][0]["delta"]["content"]
                        else:
                            logger.error(f"Unexpected response format: {response}")
                            assistant_message = "I'm sorry, I couldn't generate a response. The API response was in an unexpected format."
                    except (KeyError, IndexError) as e:
                        logger.error(f"Error extracting message from response: {e}")
                        assistant_message = "I'm sorry, I couldn't generate a response. The API response was missing the expected data."
                elif isinstance(response, dict) and "text" in response:
                    assistant_message = response["text"]
                elif isinstance(response, dict) and "content" in response:
                    content = response["content"]
                    if isinstance(content, dict):
                        if "text" in content:
                            # Handle the {type: 'text', text: '...'} format
                            assistant_message = content["text"]
                        elif "type" in content and content["type"] == "text" and "text" in content:
                            # Handle the nested {type: 'text', text: '...'} format
                            assistant_message = content["text"]
                        else:
                            logger.error(f"Unexpected content format: {content}")
                            assistant_message = str(content)
                    else:
                        assistant_message = str(content)
                elif isinstance(response, dict) and "completion_message" in response:
                    completion = response["completion_message"]
                    if isinstance(completion, dict) and "content" in completion:
                        content = completion["content"]
                        if isinstance(content, dict) and "type" in content and content["type"] == "text" and "text" in content:
                            # Handle the {type: 'text', text: '...'} format
                            assistant_message = content["text"]
                        elif isinstance(content, str):
                            assistant_message = content
                        else:
                            assistant_message = str(content)
                    else:
                        assistant_message = str(completion)
                else:
                    logger.error(f"Unexpected response format: {response}")
                    assistant_message = "I'm sorry, I couldn't generate a response. The API response was in an unexpected format."
            
            # Add assistant message to history
            self.history.add_message("assistant", assistant_message)
            
            # Set first_message to False after the first interaction
            if self.first_message:
                self.first_message = False
                
            return assistant_message
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_message = f"I encountered an error: {str(e)}"
            self.history.add_message("assistant", error_message)
            return error_message

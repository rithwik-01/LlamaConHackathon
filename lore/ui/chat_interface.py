"""
Chat interface for LORE.

This module provides functionality for chatting with a repository using the Llama API.
"""
import logging
import json
from typing import List, Dict, Any, Optional
import copy

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
            "2. Clearly indicate which parts of your response came from:\n"
            "   - Code in specific files\n"
            "   - Commit history\n"
            "   - Issues or PRs\n"
            "   - Documentation\n"
            "   - Product requirements\n"
            "   - Meeting notes\n"
            "   - Additional context\n"
            "   - Design diagrams\n"
            "3. Use the format: 'Sources: [type of source] - [specific file/commit/issue]'\n"
            "4. If information is derived from multiple sources, list ALL of them\n"
            "\nProvide clear, concise, and accurate answers. If you don't know something, say so directly."
        )
        
        # Add design diagram handling to the system prompt
        self.design_diagram_exists = False
    
    def set_design_diagram(self, exists: bool = False):
        """Set whether a design diagram is available."""
        self.design_diagram_exists = exists
    
    def chat(self, user_message: str, design_diagram_path: Optional[str] = None, stream: bool = False) -> str:
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
            
            # Create messages list with system prompt
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Llama API has strict requirements:
            # 1. All messages must have role and content
            # 2. 'content' must be a string for system, user, and assistant messages
            # 3. Only 'user' messages can have array content for multimodal
            
            # Add repository context with better structure - as a system message
            if self.repository_context:
                context_content = (
                    "# REPOSITORY ANALYSIS CONTEXT\n\n"
                    "Here is the detailed repository context including all commits, "
                    "file changes, and documentation. Pay special attention to the "
                    "RECENT SIGNIFICANT COMMITS section which shows all file changes:\n\n"
                    f"{self.repository_context}"
                )
                messages.append({
                    "role": "system",
                    "content": context_content
                })
            
            # Image message handling
            if design_diagram_path and self.design_diagram_exists:
                # First add a regular text message about the diagram
                messages.append({
                    "role": "user", 
                    "content": "I'd like you to analyze this design diagram for the codebase."
                })
                
                # Add image as a separate message with proper multimodal format
                try:
                    resized_path = resize_image_if_needed(design_diagram_path)
                    if resized_path:
                        base64_image = encode_image_to_base64(resized_path)
                        if base64_image:
                            # Create a properly structured multimodal message
                            image_msg = {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                            messages.append(image_msg)
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
            
            # Add chat history - only include simple text messages
            history_messages = []
            for msg in self.history.get_messages():
                # Skip the current message as we'll add it separately
                if msg["role"] == "user" and msg["content"] == user_message:
                    continue
                    
                # Ensure content is a string (Llama API requirement)
                if not isinstance(msg["content"], str):
                    logger.warning(f"Skipping non-string message in history: {type(msg['content'])}")
                    continue
                    
                # Add message with validly formatted content
                history_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]  # This is guaranteed to be a string now
                })
            
            # Limit history messages to avoid context limits
            if len(history_messages) > 0:
                messages.extend(history_messages[-10:])  # Only include last 10 messages
                
            # Add current user message (always a string)
            messages.append({"role": "user", "content": user_message})
            
            # Log the final message structure for debugging
            logger.debug("Final messages structure for Llama API:")
            for i, msg in enumerate(messages):
                logger.debug(f"Message {i}:")
                logger.debug(f"  Role: {msg.get('role')}")
                if isinstance(msg.get('content'), list):
                    logger.debug("  Content: [multimodal content]")
                else:
                    content_preview = str(msg.get('content', ''))[:100] + "..." if len(str(msg.get('content', ''))) > 100 else str(msg.get('content', ''))
                    logger.debug(f"  Content: {content_preview}")
            
            # Get response from LLM
            if stream:
                # We don't want to return a generator - collect all chunks
                assistant_message = ""
                for chunk in self.llm_client.chat_stream(messages):
                    if isinstance(chunk, str):
                        assistant_message += chunk
                    elif isinstance(chunk, dict) and "content" in chunk:
                        assistant_message += chunk["content"]
                    elif isinstance(chunk, dict) and "text" in chunk:
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
            
            return assistant_message
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_message = f"I encountered an error: {str(e)}"
            self.history.add_message("assistant", error_message)
            return error_message

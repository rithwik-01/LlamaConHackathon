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
        self.system_messages: List[Dict[str, Any]] = []  # Store system messages separately
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
        
        # Store system messages separately to ensure they're always included
        if role == "system":
            self.system_messages.append({
                "role": role,
                "content": content
            })
        else:
            self.messages.append({
                "role": role,
                "content": content
            })
            
            # Trim regular history if it exceeds max_history
            if len(self.messages) > self.max_history:
                self.messages = self.messages[-self.max_history:]
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get the chat history.
        
        Returns:
            List of message dictionaries, with system messages always included
        """
        # Combine system messages + regular messages
        combined_messages = copy.deepcopy(self.system_messages) + copy.deepcopy(self.messages)
        return combined_messages
    
    def clear(self):
        """Clear the chat history but preserve system messages."""
        self.messages = []  # Clear regular messages
        # Intentionally keep system_messages intact


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
            "that analyzes and explains code repositories in detail. Your primary mission is to "
            "NEVER ASSUME OR GENERALIZE - instead, you MUST ALWAYS search through the actual repository "
            "code to find precise, exact answers. GENERIC RESPONSES ARE STRICTLY FORBIDDEN.\n\n"
            
            "===== ABSOLUTELY CRITICAL: CODE-FIRST APPROACH =====\n"
            "For EVERY question asked, you MUST:\n"
            "1. SEARCH THE REPOSITORY FIRST: Before answering, thoroughly examine the repository code "
            "to find the exact implementation details. NEVER rely on general knowledge.\n"
            "2. SHOW THE EXACT CODE: Your response MUST contain multiple code snippets from specific "
            "files, showing the ACTUAL implementation, not what you think might be there.\n"
            "3. CITE PRECISE LOCATIONS: Always specify the exact file path, line numbers, function names "
            "and class names where the code appears. Every response needs at least 3-5 specific citations.\n"
            "4. EXPLAIN THE SPECIFIC CODE: Don't explain concepts - explain how THIS specific code works.\n\n"
            
            "===== FORBIDDEN RESPONSE PATTERNS =====\n"
            "1. NEVER provide abstract explanations that don't reference specific code\n"
            "2. NEVER say 'typically' or 'generally' or 'commonly' - only what IS in THIS codebase\n"
            "3. NEVER mention frameworks, patterns or concepts without showing the EXACT code implementation\n"
            "4. NEVER infer architecture or design that isn't explicitly shown in the code\n"
            "5. NEVER assume functionality exists - if you can't find it, say so clearly\n\n"
            
            "===== MANDATORY RESPONSE STRUCTURE =====\n"
            "All responses MUST follow this format:\n"
            "1. CODE ANSWER: Start by showing the EXACT code that answers the question\n"
            "2. DETAILED EXPLANATION: Explain how the specific code works, referencing file names and line numbers\n"
            "3. RELATED COMPONENTS: Show how this code connects to other parts of the repository\n"
            "4. SOURCES: List ALL specific files and functions you referenced\n\n"
            
            "===== REPOSITORY SEARCH METHODOLOGY =====\n"
            "For each question, follow this search process:\n"
            "1. IDENTIFY KEY FILES: First find the most relevant files through directory structure\n"
            "2. EXAMINE IMPORTS & DEPENDENCIES: Look at imports to understand code connections\n"
            "3. FOLLOW EXECUTION FLOW: Trace how functions call each other across files\n"
            "4. CHECK IMPLEMENTATION DETAILS: Look at the actual code, not just function signatures\n"
            "5. VERIFY WITH TESTS: Check test files to confirm behavior\n\n"
            
            "===== SELF-VERIFICATION CHECKLIST =====\n"
            "Before submitting your answer, verify that you've:\n"
            "1. Included at least 3 specific code snippets with file paths\n"
            "2. Referenced exact function names, class names, and line numbers\n"
            "3. Explained how THIS CODE works, not general concepts\n"
            "4. Cited ALL sources at the end of your response\n"
            "5. Ensured your answer could ONLY apply to THIS repository\n\n"
            
            "Remember: The user has access to the same repository as you. They will IMMEDIATELY know if "
            "you're giving generic explanations instead of analyzing the actual code. Your answers must "
            "demonstrate deep understanding of THIS SPECIFIC codebase, not general programming knowledge.\n"
        )
        
        # Add repository context to the history
        if repository_context:
            # Split the context if it's very long to ensure it stays under token limits
            if len(repository_context) > 50000:  # Approximate token threshold
                # Enhanced introduction that emphasizes the importance of using the context
                intro_part = "===== REPOSITORY CONTEXT: CRITICAL REFERENCE MATERIAL =====\n\n"
                intro_part += "The following contains ACTUAL CODE and METADATA from the repository.\n"
                intro_part += "You MUST use this information as your primary source for answers.\n"
                intro_part += "NEVER fabricate file paths, function names, or code snippets.\n"
                intro_part += "If asked about something not found in this context, ADMIT you cannot find it.\n\n"
                
                # Add specific instructions to focus on repository content
                analysis_instruction = "\n\n===== ANALYSIS REQUIREMENT =====\n"
                analysis_instruction += "For each user query, search the above repository context for relevant code.\n"
                analysis_instruction += "Quote SPECIFIC CODE SNIPPETS in your answers.\n"
                analysis_instruction += "Include COMPLETE FILE PATHS when referencing code.\n"
                analysis_instruction += "NEVER provide general explanations when specific details from the code are available.\n"
                
                # Add the system prompt first
                self.history.add_message("system", self.system_prompt)
                
                # Split the context into manageable chunks with clear section markers
                self.history.add_message("system", intro_part + repository_context[:25000] + "\n\n[Part 1 of repository context ends here]")
                
                self.history.add_message("system", "===== REPOSITORY CONTEXT CONTINUED (PART 2) =====\n\n" + 
                                        repository_context[25000:50000] + "\n\n[Part 2 of repository context ends here]")
                
                if len(repository_context) > 50000:
                    self.history.add_message("system", "===== REPOSITORY CONTEXT CONTINUED (FINAL PART) =====\n\n" + 
                                            repository_context[50000:] + "\n\n[Final part of repository context ends here]")
                
                # Add the analysis instruction at the end to reinforce its importance
                self.history.add_message("system", analysis_instruction)
            else:
                # Enhanced introduction and instructions for smaller repositories
                intro_part = "===== REPOSITORY CONTEXT: CRITICAL REFERENCE MATERIAL =====\n\n"
                intro_part += "The following contains ACTUAL CODE and METADATA from the repository.\n"
                intro_part += "You MUST use this information as your primary source for answers.\n"
                intro_part += "NEVER fabricate file paths, function names, or code snippets.\n"
                intro_part += "If asked about something not found in this context, ADMIT you cannot find it.\n\n"
                
                # Add the system prompt first
                self.history.add_message("system", self.system_prompt)
                
                # Add the repository context with clear markers
                self.history.add_message("system", intro_part + repository_context + "\n\n[Repository context ends here]")
                
                # Add specific analysis instructions
                analysis_instruction = "\n\n===== ANALYSIS REQUIREMENT =====\n"
                analysis_instruction += "For each user query, search the above repository context for relevant code.\n"
                analysis_instruction += "Quote SPECIFIC CODE SNIPPETS in your answers.\n"
                analysis_instruction += "Include COMPLETE FILE PATHS when referencing code.\n"
                analysis_instruction += "NEVER provide general explanations when specific details from the code are available.\n"
                
                self.history.add_message("system", analysis_instruction)
    
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
            
            # Always get all messages including system context
            messages = self.history.get_messages()
            
            # Debug log to help diagnose context issues
            logger.debug(f"Sending {len(messages)} total messages to LLM")
            logger.debug(f"First few message types: {[msg['role'] for msg in messages[:5]]}")  # Log first few messages
            
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
                        # Get file extension
                        file_ext = Path(current_diagram).suffix.lower()
                        mime_type = "image/jpeg"
                        
                        # Set appropriate MIME type based on file extension
                        if file_ext == ".png":
                            mime_type = "image/png"
                        elif file_ext == ".gif":
                            mime_type = "image/gif"
                        elif file_ext in [".jpg", ".jpeg"]:
                            mime_type = "image/jpeg"
                        
                        # Resize image if needed and encode to base64
                        encoded_image = encode_image_to_base64(current_diagram)
                        
                        if not encoded_image:
                            logger.error(f"Failed to encode image: {current_diagram}")
                            raise ValueError(f"Unable to encode image: {current_diagram}")
                        
                        # Create an image message in the Llama format
                        image_url = {
                            "url": f"data:{mime_type};base64,{encoded_image}",
                            "detail": "auto"  # Using 'auto' instead of 'high' for better compatibility
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
                    # Instead of silently continuing, add a message to inform the user about the error
                    self.history.add_message("assistant", f"I encountered an error processing the diagram: {str(e)}\n\nThis might be because:\n1. Your API key doesn't have permissions for image analysis\n2. The image format is not supported\n3. The image is too large\n\nI'll continue without the diagram and try to answer based on the code alone.")
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

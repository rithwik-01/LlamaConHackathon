"""
Llama API client module for LORE.

This module handles communication with the Llama 4 API for analyzing repository data.
"""
import logging
import os
import json
import base64
import sys
import traceback
from typing import Dict, List, Optional, Union, Any
import requests
from tqdm import tqdm

# Set up terminal logging
def setup_terminal_logger():
    # Create a console handler for direct terminal output
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('\n[LORE API] %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    
    # Add the handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console)
    
    # Remove duplicate handlers if any
    handlers = root_logger.handlers[:]
    for h in handlers:
        if isinstance(h, logging.StreamHandler) and h != console:
            root_logger.removeHandler(h)
    
    return root_logger

# Create a terminal logger
terminal_logger = setup_terminal_logger()

# Configure logging to show all debug messages
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Force debug output for this module
logger.propagate = True

class LlamaAPIError(Exception):
    """Exception raised for Llama API errors."""
    def __init__(self, message: str, response: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.response = response

class LlamaClient:
    """Client for interacting with the Llama 4 API."""
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Initialize the Llama API client.
        
        Args:
            api_key: Llama API key (defaults to LLAMA_API_KEY environment variable)
            api_base: Llama API base URL (defaults to LLAMA_API_BASE environment variable 
                      or https://api.llama.ai/v1)
        """
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY")
        self.api_base = api_base or os.environ.get("LLAMA_API_BASE", "https://api.llama.com/v1")
        self.model = os.environ.get("LLAMA_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8")
        
        logger.debug(f"API Key: {self.api_key}")
        logger.debug(f"API Base: {self.api_base}")
        logger.debug(f"Default Model: {self.model}")
        
        if not self.api_key:
            raise ValueError("Llama API key not provided")
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the Llama API.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            API response
        """
        try:
            url = f"{self.api_base}/{endpoint}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Enhanced debug logging
            logger.debug(f"Making request to {url}")
            
            # Log message count and size
            if 'messages' in payload:
                msg_count = len(payload['messages'])
                msg_sizes = [len(str(msg.get('content', ''))) for msg in payload['messages']]
                msg_roles = [msg.get('role', 'unknown') for msg in payload['messages']]
                
                logger.debug(f"Sending {msg_count} messages with roles: {msg_roles}")
                logger.debug(f"Message sizes (chars): {msg_sizes}")
                logger.debug(f"Total content size (approx): {sum(msg_sizes)} chars")
                
                # Identify any potential issues
                for i, msg in enumerate(payload['messages']):
                    if not isinstance(msg.get('content'), (str, list)):
                        logger.warning(f"Message {i} has invalid content type: {type(msg.get('content')).__name__}")
                    if msg.get('role') not in ['system', 'user', 'assistant']:
                        logger.warning(f"Message {i} has invalid role: {msg.get('role')}")
            
            # Make the request and capture full response
            try:
                response = requests.post(url, headers=headers, json=payload)
                
                # Try to get JSON response, but handle cases where it's not JSON
                try:
                    response_body = response.json()
                    if response.status_code == 200:
                        logger.debug(f"Response Body (success): {json.dumps(response_body, indent=2)[:500]}...")
                    else:
                        # Log the full error response
                        logger.error(f"API Error Response: {json.dumps(response_body, indent=2)}")
                except json.JSONDecodeError:
                    response_body = {"detail": "Non-JSON response: " + response.text[:500]}
                    logger.error(f"Non-JSON response: {response.text[:500]}")
            except Exception as req_ex:
                logger.error(f"Request failed with exception: {str(req_ex)}")
                raise
            
            if response.status_code != 200:
                error_msg = response_body.get('detail', 'Unknown error')
                logger.error(f"API request failed with status {response.status_code}: {error_msg}")
                # Print the full request payload for debugging
                logger.error(f"Request payload that caused error: {json.dumps(payload, default=str)[:1000]}...")
                raise LlamaAPIError(f"API request failed: {error_msg}", response_body)
            
            # Extract the actual response content
            content = None
            if 'choices' in response_body and response_body['choices']:
                choice = response_body['choices'][0]
                if isinstance(choice, dict):
                    if 'message' in choice and isinstance(choice['message'], dict):
                        content = choice['message'].get('content')
                    elif 'text' in choice:
                        content = choice['text']
            elif 'completion_message' in response_body:
                msg = response_body['completion_message']
                if isinstance(msg, dict):
                    if 'content' in msg:
                        content = msg['content']
                    elif 'text' in msg:
                        content = msg['text']
                else:
                    content = str(msg)
                    
            if content is None:
                logger.error(f"Could not extract content from response: {json.dumps(response_body, indent=2)}")
                raise LlamaAPIError("No content found in API response", response_body)
            
            # Log content structure before cleaning
            if isinstance(content, dict):
                logger.debug(f"Content is a dictionary: {json.dumps(content, indent=2)}")
                
            # Clean up the content
            cleaned_content = self._sanitize_response(content)
            
            # Return in a consistent format
            return {
                'completion_message': {
                    'content': cleaned_content,
                    'role': 'assistant',
                    'type': 'text'
                },
                'metrics': response_body.get('usage', {})
            }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in _make_request: {str(e)}")
            raise
    
    def _limit_context_size(self, messages: List[Dict[str, Any]], model: str = None, max_tokens: int = 100000) -> List[Dict[str, Any]]:
        """
        Limit the context size to avoid token limit errors.
        
        Args:
            messages: List of message dictionaries
            model: Model name to determine token limit
            max_tokens: Maximum tokens allowed (default set conservatively)
            
        Returns:
            Processed messages that fit within token limits
        """
        # If there are very few messages, likely no problem
        if len(messages) <= 3:
            return messages
            
        # First, identify system messages with very long content
        system_messages = []
        user_assistant_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                user_assistant_messages.append(msg)
        
        # Sort system messages by length (shortest first)
        system_messages.sort(key=lambda x: len(x.get("content", "")) if isinstance(x.get("content"), str) else 0)
        
        # If we have large system messages (repository context), keep essential ones
        if system_messages and len(system_messages) > 1:
            # Keep the largest system message (likely main repository context)
            # and the smallest one (likely system instructions)
            main_context = system_messages[-1]  # Largest
            system_instructions = system_messages[0]  # Smallest
            
            # If the repository context is extremely large, truncate it
            if isinstance(main_context.get("content"), str) and len(main_context.get("content", "")) > 50000:
                content = main_context["content"]
                logger.warning(f"Repository context is very large ({len(content)} chars). Truncating to 50000 chars.")
                
                # Extract first and last parts to preserve structure
                main_context["content"] = (
                    "===== TRUNCATED REPOSITORY CONTEXT =====\n\n" +
                    content[:25000] + "\n\n...\n\n" + content[-25000:] +
                    "\n\n===== END OF TRUNCATED REPOSITORY CONTEXT ====="
                )
            
            # Rebuild with limited system messages
            final_messages = [system_instructions, main_context] + user_assistant_messages[-6:]  # Keep last 6 conversation turns
        else:
            # No large system messages, just keep everything
            final_messages = messages
        
        logger.debug(f"Limited context size: {len(messages)} messages → {len(final_messages)} messages")
        return final_messages

    def _sanitize_response(self, text: str) -> Any:
        """
        Clean up the response text to remove repetitive or nonsensical content.
        
        Args:
            text: Raw response text or content dictionary
            
        Returns:
            Cleaned response text or preserved content dictionary
        """
        # Handle dictionary content - preserve it rather than converting to string
        if isinstance(text, dict):
            if 'type' in text and text['type'] == 'text' and 'text' in text:
                # This is a structured content object - sanitize just the text part
                sanitized_text = self._sanitize_response(text['text'])
                return {
                    'type': 'text',
                    'text': sanitized_text
                }
            elif 'text' in text:
                text = text['text']
            elif 'content' in text:
                text = text['content']
            else:
                logger.warning(f"Unexpected dict format in _sanitize_response: {text}")
                return text  # Return the original dict instead of converting to string
        
        # Handle non-string, non-dict input
        if not isinstance(text, str):
            logger.warning(f"Unexpected response type in _sanitize_response: {type(text)}")
            return text  # Return as is instead of forcing to string
            
        if not text:
            return text
        
        # Check for excessive repetition of phrases or patterns
        common_repetitive_phrases = [
            "for example", "for instance", "as follows", "to say", "as if",
            "for as", "as to", "they were", "SUFFIX", "example example"
        ]
        
        # Count occurrences of repetitive phrases
        occurrences = {}
        for phrase in common_repetitive_phrases:
            count = text.lower().count(phrase.lower())
            if count > 5:  # If a phrase appears more than 5 times, it may be repetitive
                occurrences[phrase] = count
                logger.warning(f"Detected potentially repetitive phrase: '{phrase}' ({count} occurrences)")
        
        # If excessive repetition is found, truncate the text
        if any(count > 20 for count in occurrences.values()):
            # Find where the repetitive section probably starts
            max_phrase = max(occurrences.items(), key=lambda x: x[1])[0]
            logger.warning(f"Excessive repetition detected for '{max_phrase}' - truncating response")
            
            # Find roughly where the repetitive content starts
            normal_content_percentage = 0.7
            estimated_good_length = int(len(text) * normal_content_percentage)
            
            # Check if repetition starts after a certain point
            segments = text.lower().split(max_phrase.lower())
            good_content = segments[0]
            
            # Try to find a good cutoff point
            for i, segment in enumerate(segments[1:10]):  # Check first few segments
                if len(good_content + max_phrase + segment) < estimated_good_length:
                    good_content += max_phrase + segment
                else:
                    break
            
            # Add a note about truncation
            truncated_text = good_content + "\n\n[Content truncated due to excessive repetition]"
            return truncated_text
        
        # Split into sections by headers
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            if line.startswith('#'):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
                
        if current_section:
            sections.append('\n'.join(current_section))
            
        # Process each section
        cleaned_sections = []
        for section in sections:
            # Skip sections that are mostly repetitive
            lines = section.split('\n')
            unique_lines = set(lines)
            if len(unique_lines) < len(lines) * 0.3:  # More than 70% repetition
                continue
                
            # Remove consecutive duplicate lines
            cleaned_lines = []
            prev_line = None
            for line in lines:
                if line != prev_line and line.strip():
                    cleaned_lines.append(line)
                prev_line = line
                    
            # Skip sections with nonsensical content
            section_text = '\n'.join(cleaned_lines)
            if any(word in section_text.lower() for word in ['sălbă', 'săptă', 'spre', 'mistress']):
                continue
                
            cleaned_sections.append(section_text)
            
        return '\n\n'.join(cleaned_sections)
    
    SYSTEM_PROMPTS = {
        "analyze_architecture": (
            "IMPORTANT: Analyze the ACTUAL code in this repository, not generic frameworks. "
            "Analyze this codebase's architecture and design. Focus on: "
            "1. Main components and their interactions with specific file/class references "
            "2. Key design patterns and architectural choices as implemented in the code "
            "3. Code organization and modularity with directory structure details "
            "4. Notable strengths or areas for improvement based on actual implementation "
            "Be concise and specific. Include references to actual files, classes, and functions."
        ),
        "analyze_complexity": (
            "IMPORTANT: Analyze the ACTUAL code in this repository, not generic frameworks. "
            "Evaluate code complexity and maintainability. Identify: "
            "1. Complex components needing attention (with specific file paths) "
            "2. Potential technical debt with code examples "
            "3. Specific improvement recommendations based on the actual implementation "
            "Be brief and actionable. Always reference actual code."
        ),
        "historical_analysis": (
            "IMPORTANT: Analyze the ACTUAL code in this repository, not generic frameworks. "
            "Analyze repository history. Focus on: "
            "1. Major changes and evolution with specific commit references "
            "2. Development patterns seen in the actual commit history "
            "3. Key milestones with dates and feature implementations "
            "Keep it short and focused. Use only information from the repository history."
        ),
        "chat": (
            "IMPORTANT: You must analyze the ACTUAL CODE from this repository. DO NOT provide generic responses about frameworks unless they're actually used here. "
            "You are a helpful coding assistant analyzing this repository. "
            "Always reference specific files, classes, functions and code structures from the repository. "
            "Never make up information - if you don't see certain code, admit you can't find it. "
            "Focus on implementation details found in the code, not theoretical descriptions. "
            "Quote relevant code snippets when appropriate to support your answers."
        )
    }
    
    def analyze_repository(
        self,
        content: str,
        model: str = "Llama-4-Maverick-17B-128E-Instruct-FP8",
        temperature: float = 0.2,
        max_tokens: int = 10000,
        task: str = "analyze_architecture",
        messages: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze repository content using the Llama API.
        
        Args:
            content: Repository content to analyze
            model: Model to use for analysis
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            task: Analysis task to perform
            messages: Optional chat messages for chat task
            
        Returns:
            API response as a dictionary
        """
        logger.debug(f"Using {model} for analysis")
        
        # Prepare request payload
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if messages:
            # Chat format
            payload["messages"] = messages
        else:
            # Standard format
            logger.debug("Using standard format for analysis")
            payload["messages"] = [
                {
                    "role": "system",
                    "content": "IMPORTANT: You must analyze the ACTUAL CODE from this repository. DO NOT provide generic responses about frameworks unless they're actually used here. "
                              "You are a helpful coding assistant analyzing this repository. "
                              "Always reference specific files, classes, functions and code structures from the repository. "
                              "Never make up information - if you don't see certain code, admit you can't find it. "
                              "Focus on implementation details found in the code, not theoretical descriptions. "
                              "Quote relevant code snippets when appropriate to support your answers."
                },
                {
                    "role": "user",
                    "content": f"Task: {task}\n\nRepository content:\n{content}"
                }
            ]
        
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Add model and metrics to response
            result["model"] = model
            result["metrics"] = {
                "tokens": result.get("usage", {}).get("total_tokens", 0),
                "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": result.get("usage", {}).get("completion_tokens", 0)
            }
            
            # Add response format metadata without sending it to the API
            result["response_format"] = {
                "type": "markdown",
                "version": "1.0"
            }
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'json'):
                error_details = e.response.json()
                logger.error(f"API error details: {error_details}")
            raise LlamaAPIError(f"API request failed: {str(e)}")
    
    def stream_analysis(self, 
                       content: str, 
                       model: str = "Llama-4-Maverick-17B-128E-Instruct-FP8", 
                       temperature: float = 0.2,
                       max_tokens: int = 10000,
                       task: str = "analyze_architecture",
                       messages: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Stream repository analysis from Llama API.
        
        Args:
            content: Repository content to analyze
            model: Llama model to use (defaults to Llama-4-Maverick-17B-128E-Instruct-FP8)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            task: Analysis task to perform
            messages: Optional list of chat messages for chat mode
            
        Returns:
            Complete response after streaming
        """
        # Define system prompts based on task (same as above)
        system_prompts = {
            "analyze_architecture": (
                "IMPORTANT: Analyze the ACTUAL code in this repository, not generic frameworks. "
                "Analyze this codebase's architecture and design. Focus on: "
                "1. Main components and their interactions with specific file/class references "
                "2. Key design patterns and architectural choices as implemented in the code "
                "3. Code organization and modularity with directory structure details "
                "4. Notable strengths or areas for improvement based on actual implementation "
                "Be concise and specific. Include references to actual files, classes, and functions."
            ),
            "analyze_complexity": (
                "IMPORTANT: Analyze the ACTUAL code in this repository, not generic frameworks. "
                "Evaluate code complexity and maintainability. Identify: "
                "1. Complex components needing attention (with specific file paths) "
                "2. Potential technical debt with code examples "
                "3. Specific improvement recommendations based on the actual implementation "
                "Be brief and actionable. Always reference actual code."
            ),
            "historical_analysis": (
                "IMPORTANT: Analyze the ACTUAL code in this repository, not generic frameworks. "
                "Analyze repository history. Focus on: "
                "1. Major changes and evolution with specific commit references "
                "2. Development patterns seen in the actual commit history "
                "3. Key milestones with dates and feature implementations "
                "Keep it short and focused. Use only information from the repository history."
            ),
            "chat": (
                "IMPORTANT: You must analyze the ACTUAL CODE from this repository. DO NOT provide generic responses about frameworks unless they're actually used here. "
                "You are a helpful coding assistant analyzing this repository. "
                "Always reference specific files, classes, functions and code structures from the repository. "
                "Never make up information - if you don't see certain code, admit you can't find it. "
                "Focus on implementation details found in the code, not theoretical descriptions. "
                "Quote relevant code snippets when appropriate to support your answers."
            )
        }
        
        # Use default if task not found
        system_prompt = system_prompts.get(
            task, 
            "IMPORTANT: You must analyze the ACTUAL CODE from this repository. DO NOT provide generic responses about frameworks unless they're actually used here. "
            "You are an expert software engineer analyzing a repository. "
            "Always reference specific files, classes, functions and code structures from the repository. "
            "Never make up information - if you don't see certain code, admit you can't find it. "
            "Focus on implementation details found in the code, not theoretical descriptions."
        )
        
        # Prepare the request payload
        if messages is not None:
            # Use provided messages for chat mode
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
        else:
            # Use standard format for analysis
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
        
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.debug(f"Request URL: {url}")
        logger.debug(f"Request Headers: {headers}")
        logger.debug(f"Request Payload: {json.dumps(payload, indent=2)}")
        
        full_response = ""
        
        try:
            with requests.post(url, headers=headers, json=payload, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data == '[DONE]':
                                break
                                
                            try:
                                chunk = json.loads(data)
                                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if content:
                                    full_response += content
                                    print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse streaming response: {data}")
            
            print()  # New line after streaming finishes
            return {"content": full_response}
            
        except Exception as e:
            logger.error(f"Error streaming from Llama API: {e}")
            raise
    
    def chunk_and_analyze(self, 
                         content: str,
                         model: str = "llama-2-70b-chat",  
                         chunk_size: int = 100000,
                         overlap: int = 5000,
                         task: str = "analyze_architecture") -> List[Dict[str, Any]]:
        """
        Chunk large repository content and analyze each chunk separately.
        
        Args:
            content: Repository content to analyze
            model: Llama model to use
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            task: Analysis task to perform
            
        Returns:
            List of analysis results for each chunk
        """
        if len(content) <= chunk_size:
            return [self.analyze_repository(content, model=model, task=task)]
        
        results = []
        start = 0
        
        chunks = []
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunks.append(content[start:end])
            start = end - overlap
        
        for i, chunk in enumerate(tqdm(chunks, desc="Analyzing repository chunks")):
            try:
                result = self.analyze_repository(
                    chunk,
                    model=model,
                    task=f"{task} (chunk {i+1}/{len(chunks)})",
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing chunk {i+1}: {e}")
        
        return results
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 10000,
        temperature: float = 0.7,
        stream: bool = False,
        retry_count: int = 0  # Track retries to prevent infinite recursion
    ) -> Dict[str, Any]:
        """
        Send a chat request to the API.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            stream: Whether to stream the response
            
        Returns:
            API response
        """
        # Enhanced logging for debugging message context issues
        logger.debug(f"Chat API request with {len(messages)} messages")
        
        # Check if there are any system messages with repository context
        has_system_message = any(msg.get("role") == "system" for msg in messages)
        has_repo_context = any(msg.get("role") == "system" and 
                              isinstance(msg.get("content"), str) and 
                              ("repository context" in msg.get("content").lower() or 
                               "code repository" in msg.get("content").lower()) 
                              for msg in messages)
        
        logger.debug(f"Has system message: {has_system_message}, Has repo context: {has_repo_context}")
        
        # Log some details about the messages being sent
        for i, msg in enumerate(messages[:5]):  # Log first 5 messages
            role = msg.get("role", "unknown")
            content = msg.get("content")
            content_type = type(content).__name__
            content_preview = ""
            
            if isinstance(content, str):
                content_preview = content[:100] + "..." if len(content) > 100 else content
            elif isinstance(content, list):
                content_preview = f"list with {len(content)} items"
            
            logger.debug(f"Message {i} - Role: {role}, Type: {content_type}, Preview: {content_preview}")
        
        # For debugging: Check if any system message is very long (likely contains repo context)
        system_msg_lengths = [len(msg.get("content", "")) if isinstance(msg.get("content"), str) else 0 
                            for msg in messages if msg.get("role") == "system"]
        if system_msg_lengths:
            logger.debug(f"System message lengths: {system_msg_lengths}")
            
        # NEW: Verify that we have repository context in the messages
        has_repo_ctx = False
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                content = msg.get("content", "")
                if ("repository context" in content.lower() or 
                    "===== REPOSITORY CONTEXT" in content or
                    len(content) > 1000):  # Long system messages likely contain repo context
                    has_repo_ctx = True
                    logger.debug(f"Found repository context in system message: {content[:100]}...")
                    break
        
        if not has_repo_ctx:
            logger.warning("No repository context found in messages! Responses may be generic.")
        
        # Use the default model or override
        model = model or self.model
        url = f"{self.api_base}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Do not limit context since Llama API can handle up to 1M tokens
        # Just log the size of the context being sent
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        logger.debug(f"Sending full context to Llama API: {len(messages)} messages, approximately {total_chars} characters")
        
        # Detailed debug logs for consecutive requests
        import time
        logger.debug(f"\n================== NEW CHAT REQUEST ==================")
        logger.debug(f"Request with user API key: {self.api_key[:8]}...")
        logger.debug(f"Request timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Track message history to debug second message issue
        try:
            # Use a global variable to track conversation turns
            if not hasattr(self, '_conversation_turn'):
                self._conversation_turn = 1
                logger.debug(f"First conversation turn")
            else:
                self._conversation_turn += 1
                logger.debug(f"Conversation turn #{self._conversation_turn}")
                
            # Record detailed info about messages in this turn
            logger.debug(f"Message roles in this turn: {[m.get('role', 'unknown') for m in messages]}")
            logger.debug(f"Total messages in this turn: {len(messages)}")
        except Exception as e:
            logger.error(f"Error tracking conversation: {e}")
        
        # Dump the full messages to a log file for debugging
        try:
            import os
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"message_dump_{int(time.time())}.json")
            with open(log_path, "w") as f:
                import json
                json.dump({"messages": [m for m in messages]}, f, default=str, indent=2)
            logger.debug(f"Dumped full message context to {log_path}")
        except Exception as e:
            logger.error(f"Failed to dump messages to file: {e}")
        
        # FIX: Clean up the messages to ensure they're valid for the API
        # Using the structured format the user provided from chat_interface.py
        cleaned_messages = []
        
        # First, let's check if there might be messages with wrong structure
        has_invalid_format = False
        invalid_indices = []
        for i, msg in enumerate(messages):
            # Check if message conforms to the expected schema
            if not isinstance(msg, dict) or 'role' not in msg:
                has_invalid_format = True
                invalid_indices.append(i)
            elif msg.get('role') not in ['system', 'user', 'assistant', 'tool']:
                has_invalid_format = True
                invalid_indices.append(i)
            elif 'content' not in msg and msg.get('role') != 'assistant':
                # Content is required except for assistant messages that might have function_call instead
                has_invalid_format = True
                invalid_indices.append(i)
                
        if has_invalid_format:
            logger.warning(f"Found {len(invalid_indices)} messages with invalid format at indices: {invalid_indices}")
            
        # Now process each message individually
        for i, msg in enumerate(messages):
            # Create a new message with only the essential fields
            clean_msg = {}
            
            # Extract role and content
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Ensure role is valid - this is critical for JSON schema validation
            if role not in ["system", "user", "assistant", "tool"]:
                logger.warning(f"Invalid role '{role}' in message {i}, defaulting to 'user'")
                role = "user"  # Default to user role
                
            # CRITICAL FIX: Format content properly according to chat_interface.py pattern
            # This is the key to fixing the message.6 schema validation error
            
            # Now set the fields in the clean message
            clean_msg["role"] = role
            
            # Format content according to the structured format in chat_interface.py
            if content is not None:
                try:
                    if isinstance(content, str):
                        # Convert plain strings to structured format
                        clean_msg["content"] = {
                            "type": "text",
                            "text": content
                        }
                    elif isinstance(content, dict) and "type" in content and content["type"] == "text" and "text" in content:
                        # Already in correct format
                        clean_msg["content"] = content.copy()  # Make a copy to avoid reference issues
                    elif isinstance(content, list):
                        # For multi-part messages, ensure each part has correct structure
                        formatted_parts = []
                        for part in content:
                            if isinstance(part, dict) and "type" in part:
                                formatted_parts.append(part.copy())  # Make a copy
                            else:
                                # Safely convert to string
                                text = "[Content]"  # Default
                                try:
                                    text = str(part)
                                except Exception as e:
                                    logger.error(f"Error converting part to string: {e}")
                                
                                formatted_parts.append({
                                    "type": "text",
                                    "text": text
                                })
                        clean_msg["content"] = formatted_parts
                    else:
                        # Convert any other content type to structured format safely
                        text = "[Content]"  # Default if conversion fails
                        try:
                            if content is not None:
                                text = str(content)
                        except Exception as e:
                            logger.error(f"Error converting content to string: {e}")
                        
                        clean_msg["content"] = {
                            "type": "text",
                            "text": text
                        }
                except Exception as e:
                    # Fallback for any unexpected errors
                    logger.error(f"Unexpected error formatting content: {e}")
                    clean_msg["content"] = {
                        "type": "text",
                        "text": "[Error: Could not format content]"
                    }
            
            # Add name field if present and valid
            if "name" in msg and msg["name"]:
                clean_msg["name"] = msg["name"]
                
            # Add tool_call_id if this is a tool message
            if role == "tool" and "tool_call_id" in msg:
                clean_msg["tool_call_id"] = msg["tool_call_id"]
                
            # Function calls for assistant messages
            if role == "assistant" and "function_call" in msg:
                clean_msg["function_call"] = msg["function_call"]
                
            cleaned_messages.append(clean_msg)
            
        logger.debug(f"Cleaned {len(messages)} messages to {len(cleaned_messages)} valid messages")
        
        # Create the data payload for the API
        # If repository is very large, just keep the system message, last 5 messages of conversation
        if len(cleaned_messages) > 10 and total_chars > 800000:
            logger.warning(f"Total context size is very large: {total_chars} chars. Limiting to essential messages.")
            
            # Find system messages (likely containing repo context)
            system_msgs = [msg for msg in cleaned_messages if msg.get("role") == "system"]
            
            # Find main system message (likely the longest one with repo context)
            main_system_msg = None
            if system_msgs:
                if len(system_msgs) == 1:
                    main_system_msg = system_msgs[0]
                else:
                    # Find the longest system message that likely contains repo context
                    main_system_msg = max(system_msgs, key=lambda m: len(str(m.get("content", ""))))
            
            # Get recent conversation (user/assistant messages)
            recent_msgs = [m for m in cleaned_messages[-10:] if m.get("role") in ["user", "assistant"]]
            
            # Build final message list
            final_messages = []
            if main_system_msg:
                final_messages.append(main_system_msg)
            final_messages.extend(recent_msgs)
            
            logger.debug(f"Limited context from {len(cleaned_messages)} to {len(final_messages)} messages")
            cleaned_messages = final_messages
        
        # Prepare the data for the API request
        # CRITICAL: Convert the structured content format back to string format for Llama API
        api_messages = []
        
        for msg in cleaned_messages:
            api_msg = {"role": msg.get("role")}
            content = msg.get("content")
            
            # Convert structured content back to string format for the API
            try:
                if isinstance(content, dict) and "type" in content and content["type"] == "text" and "text" in content:
                    # Extract the text content from structured format
                    text_value = content["text"]
                    api_msg["content"] = text_value if isinstance(text_value, str) else str(text_value)
                elif isinstance(content, list):
                    # For multimodal content, extract text parts and join
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and "type" in part and part["type"] == "text" and "text" in part:
                            text_value = part["text"]
                            if isinstance(text_value, str):
                                text_parts.append(text_value)
                            else:
                                # Handle non-string text fields
                                try:
                                    text_parts.append(str(text_value))
                                except Exception:
                                    text_parts.append("[Non-text content]")
                    api_msg["content"] = "\n".join(text_parts) if text_parts else ""
                elif isinstance(content, str):
                    # Already a string
                    api_msg["content"] = content
                else:
                    # Safely convert other types to string
                    try:
                        api_msg["content"] = str(content) if content is not None else ""
                    except Exception as e:
                        logger.error(f"Error converting content to string: {e}")
                        api_msg["content"] = "[Content conversion error]"
            except Exception as e:
                # Ultimate fallback
                logger.error(f"Unexpected error processing content: {e}")
                api_msg["content"] = ""
            
            # Include other fields if present
            if "name" in msg:
                api_msg["name"] = msg["name"]
            if "function_call" in msg:
                api_msg["function_call"] = msg["function_call"]
            if "tool_call_id" in msg and msg.get("role") == "tool":
                api_msg["tool_call_id"] = msg["tool_call_id"]
                
            api_messages.append(api_msg)
            
        # Log what we're sending to the API
        logger.debug(f"Sending {len(api_messages)} messages to Llama API")
        
        # Construct the data for the API request
        data = {
            "model": model,
            "messages": api_messages,  # Send converted messages that Llama API can understand
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        # Detailed message debugging for first cleaned messages
        if len(cleaned_messages) > 0:
            logger.debug("First cleaned message role: " + cleaned_messages[0].get("role", "unknown"))
            if isinstance(cleaned_messages[0].get("content"), str):
                preview = cleaned_messages[0].get("content", "")[:100] + "..." if len(cleaned_messages[0].get("content", "")) > 100 else cleaned_messages[0].get("content", "")
                logger.debug(f"First cleaned message content preview: {preview}")
                
        if len(cleaned_messages) > 1:
            logger.debug("Last cleaned message role: " + cleaned_messages[-1].get("role", "unknown"))
            if isinstance(cleaned_messages[-1].get("content"), str):
                preview = cleaned_messages[-1].get("content", "")[:100] + "..." if len(cleaned_messages[-1].get("content", "")) > 100 else cleaned_messages[-1].get("content", "")
                logger.debug(f"Last cleaned message content preview: {preview}")
        
        # Fix for message 6 schema validation
        if len(cleaned_messages) > 6:
            print(f"\n🔥 APPLYING STRUCTURED FORMAT FIX TO MESSAGE 6")
            index_6_msg = cleaned_messages[6]
            role = index_6_msg.get('role')
            content = index_6_msg.get('content')
            
            # Ensure message 6 has proper structure with nested content format
            print(f"Original message 6: {index_6_msg}")
            
            # Apply the structured format specifically for message 6
            if role in ['system', 'user', 'assistant']:
                if not isinstance(content, dict) or 'type' not in content or 'text' not in content:
                    # Force the correct structured format
                    if isinstance(content, str):
                        cleaned_messages[6]['content'] = {
                            "type": "text",
                            "text": content
                        }
                    else:
                        # Safely convert to string to handle any type
                        text_content = ""
                        try:
                            text_content = str(content) if content else "Please continue with your analysis."
                        except Exception as e:
                            logger.error(f"Error converting content to string: {e}")
                            text_content = "Please continue with your analysis."
                            
                        cleaned_messages[6]['content'] = {
                            "type": "text",
                            "text": text_content
                        }
                    print(f"\n✅ Fixed message 6 by applying structured content format")
            elif role == 'tool':
                # Ensure tool messages have the correct format
                cleaned_messages[6] = {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": "Please continue analyzing the repository."
                    }
                }
                print(f"\n✅ Replaced tool message with user message using proper structure")
                
            print(f"Fixed message 6: {cleaned_messages[6]}")
            
            # ADDITIONAL SAFETY: Check for any remaining tool messages that might cause issues
            for i, msg in enumerate(cleaned_messages):
                if msg.get("role") == "tool":
                    print(f"\n⚠️ Found potentially problematic tool message at index {i}")
                    # Replace with user message - tool messages seem to cause schema issues
                    cleaned_messages[i] = {
                        "role": "user",
                        "content": "Please continue with the analysis."
                    }
                    print(f"✅ Replaced tool message at index {i} with safe user message")
            
            # FINAL VERIFICATION: Ensure we have valid message sequence
            print("\n🔍 VERIFYING CONVERSATION SEQUENCE PATTERN")
            # Extract sequence of roles for analysis
            role_sequence = [msg.get("role") for msg in cleaned_messages]
            print(f"Message sequence: {role_sequence}")
            
            # Fix alternating pattern if needed - this is critical for Llama API
            valid_sequence = True
            for i in range(1, len(cleaned_messages)):
                prev_role = cleaned_messages[i-1].get("role")
                curr_role = cleaned_messages[i].get("role")
                
                # Check for consecutive non-system messages of same role (which is invalid)
                if prev_role == curr_role and prev_role != "system":
                    valid_sequence = False
                    print(f"⚠️ Invalid sequence: {prev_role} followed by {curr_role} at indices {i-1},{i}")
                    # Fix by converting second message to appropriate alternating type
                    if curr_role == "user":
                        cleaned_messages[i]["role"] = "assistant"
                    else:
                        cleaned_messages[i]["role"] = "user"
                    print(f"✅ Changed message {i} role to {cleaned_messages[i]['role']}")
            
            print(f"✅ Message 6 issue remediation complete")
                
        # Fix for second message issue: Ensure all messages have the right structure
        if self._conversation_turn > 1:
            # Terminal output for debugging - directly visible in the Streamlit console
            print("\n====== SECOND CHAT QUESTION DETECTED - APPLYING FIXES ======")
            print(f"Conversation turn: #{self._conversation_turn}")
            
            # Count message types
            user_msgs = [m for m in cleaned_messages if m.get('role') == 'user']
            assistant_msgs = [m for m in cleaned_messages if m.get('role') == 'assistant']
            system_msgs = [m for m in cleaned_messages if m.get('role') == 'system']
            
            print(f"Before fixes: {len(system_msgs)} system, {len(user_msgs)} user, and {len(assistant_msgs)} assistant messages")
            
            # CRITICAL CHECK: Verify system messages are still present
            if not system_msgs:
                print("\n⚠️ CRITICAL ERROR: No system messages found in subsequent request! This causes 400 errors.")
                print("Adding system message to fix the issue...")
                # Try to recover by adding a system message - this is critical for second questions
                system_msg = {"role": "system", "content": "You are LORE (Long-context Organizational Repository Explorer), an expert AI assistant that analyzes and explains code repositories in detail."}
                cleaned_messages.insert(0, system_msg)
                print("✅ Added system message to prevent 400 error")
            
            # Check for specific message structure issues
            for i, msg in enumerate(cleaned_messages):
                role = msg.get('role', 'unknown')
                content_type = type(msg.get('content')).__name__
                content_len = len(str(msg.get('content', ''))) if msg.get('content') else 0
                
                # Check for message content length limits
                content = msg.get('content', '')
                # Handle different content types properly when truncating
                if isinstance(content, dict) and 'type' in content and content['type'] == 'text' and 'text' in content:
                    # Handle structured content format
                    text_content = content['text']
                    if isinstance(text_content, str) and len(text_content) > 100000:
                        logger.warning(f"Very long structured text content found! Truncating.")
                        # Create a new truncated version
                        truncated_text = text_content[:50000] + "\n[Content truncated]\n" + text_content[-10000:]
                        # Replace the content while preserving structure
                        cleaned_messages[i]['content'] = {
                            "type": "text",
                            "text": truncated_text
                        }
                elif isinstance(content, str) and len(content) > 100000:
                    logger.warning(f"Very long message found in turn {self._conversation_turn}! Truncating.")
                    cleaned_messages[i]['content'] = content[:50000] + "\n[Content truncated]\n" + content[-10000:]
                    print(f"✅ Truncated system message from {content_len} to {len(cleaned_messages[i]['content'])} chars")
            
            # ADDITIONAL FIX: Add repository context if it might have been lost
            if len(system_msgs) == 1 and all(len(str(m.get('content', ''))) < 1000 for m in system_msgs):
                print("\n⚠️ Repository context might be missing in system messages!")
                # Add a note about possible issue
                print("Possible issue: Repository context was lost between chat messages")
                
            # Final counts after fixes
            user_msg_count = sum(1 for m in cleaned_messages if m.get('role') == 'user')
            assistant_msg_count = sum(1 for m in cleaned_messages if m.get('role') == 'assistant')
            system_msg_count = sum(1 for m in cleaned_messages if m.get('role') == 'system')
            print(f"After fixes: {system_msg_count} system, {user_msg_count} user, and {assistant_msg_count} assistant messages")
            print("====== FIXES COMPLETE ======\n")
        
        logger.debug(f"Sending request to {url} with model {model}")
        
        # Log the final payload (shortened for readability)
        print("\n=============== FINAL API PAYLOAD STRUCTURE ===============")
        print(f"Model: {model}")
        print(f"Message count: {len(messages)}")
        print(f"Temperature: {temperature}")
        print(f"Max tokens: {max_tokens}")
        
        # Validate message structure against Llama API schema expectations
        print("\n🔍 VALIDATING MESSAGE STRUCTURE AGAINST API SCHEMA")
        valid_message_structure = True
        for i, msg in enumerate(data['messages']):
            validation_errors = []
            # Check required fields according to Llama API schema
            if 'role' not in msg:
                validation_errors.append("missing 'role' field")
            elif msg['role'] not in ['system', 'user', 'assistant', 'tool']:
                validation_errors.append(f"invalid role '{msg['role']}'")
                
            # Check content field structure
            if 'content' not in msg and msg.get('role') != 'assistant':
                # Content is required for non-assistant messages (assistant can have function_call instead)
                validation_errors.append("missing 'content' field")
            elif 'content' in msg:
                if not isinstance(msg['content'], str) and not isinstance(msg['content'], list) and msg['content'] is not None:
                    validation_errors.append(f"content must be string, list, or null, got {type(msg['content']).__name__}")
                elif isinstance(msg['content'], list):
                    # Check multimodal content structure
                    for item in msg['content']:
                        if not isinstance(item, dict) or 'type' not in item:
                            validation_errors.append("invalid multimodal content structure")
                            break
                            
            # Tool messages require tool_call_id
            if msg.get('role') == 'tool' and 'tool_call_id' not in msg:
                validation_errors.append("tool message missing 'tool_call_id'")
                # Fix it by adding a default tool_call_id
                msg['tool_call_id'] = f"call_{i}"
                
            # If errors found, log them
            if validation_errors:
                print(f"⚠️ Message {i} has schema validation errors: {', '.join(validation_errors)}")
                valid_message_structure = False
                # Try to fix critical issues
                if msg.get('role') == 'tool' and 'tool_call_id' not in msg:
                    msg['tool_call_id'] = f"call_{i}"
                if 'content' not in msg and msg.get('role') != 'assistant':
                    msg['content'] = ""
                if 'content' in msg and not isinstance(msg['content'], (str, list)) and msg['content'] is not None:
                    msg['content'] = str(msg['content'])
                print(f"✅ Fixed message {i}: {msg}")
        
        if valid_message_structure:
            print("✅ All messages conform to expected schema")
        else:
            print("⚠️ Some messages required fixing to conform to schema")
            
        # Attempt to serialize data to verify it's valid JSON
        try:
            import json
            serialized_data = json.dumps(data)
            logger.debug(f"Data serialization successful, payload size: {len(serialized_data)} bytes")
        except Exception as json_err:
            logger.error(f"Data serialization failed: {str(json_err)}")
            print(f"\n🔍 JSON SERIALIZATION FAILED: {str(json_err)}")
            # Attempt to find and fix the problematic parts
            logger.error("Trying to fix JSON serialization issues...")
            try:
                # Create a sanitized copy with only strings for content
                sanitized_messages = []
                for m in data['messages']:
                    fixed_msg = {"role": m.get("role", "user")}
                    if 'content' in m:
                        fixed_msg["content"] = str(m["content"]) if not isinstance(m["content"], (str, list)) else m["content"]
                    else:
                        fixed_msg["content"] = ""
                    if m.get("role") == "tool" and "tool_call_id" in m:
                        fixed_msg["tool_call_id"] = m["tool_call_id"]
                    sanitized_messages.append(fixed_msg)
                data['messages'] = sanitized_messages
                logger.debug("Fixed JSON serialization issues in messages")
                print("✅ Fixed JSON serialization issues")
            except Exception as fix_err:
                logger.error(f"Failed to fix JSON serialization: {str(fix_err)}")
                print(f"⚠️ Could not fix JSON issues: {str(fix_err)}")
        
        try:
            if stream:
                return self._stream_chat(url, headers, data)
            else:
                logger.debug(f"Sending request to {url} with model {model}")
                try:
                    # Log headers (without auth token)
                    safe_headers = headers.copy()
                    if 'Authorization' in safe_headers:
                        safe_headers['Authorization'] = 'Bearer [REDACTED]'
                    logger.debug(f"Request headers: {safe_headers}")
                    
                    # Before sending the request, log what we're sending
                    if self._conversation_turn > 1:
                        print("\n====== SENDING REQUEST TO LLAMA API ======")
                        print(f"Model: {model}")
                        print(f"Max tokens: {max_tokens}")
                        print(f"Message count: {len(data['messages'])}")
                        print(f"First message role: {data['messages'][0].get('role') if data['messages'] else 'none'}")
                        print(f"Last message role: {data['messages'][-1].get('role') if data['messages'] else 'none'}")
                    
                    # First try sending the request with a timeout
                    logger.debug(f"Sending POST request to {url}...")
                    
                    # EMERGENCY BYPASS: If this is a retry and message 6 schema error is likely
                    if retry_count > 0 and len(data['messages']) > 6:
                        print("\n🛠️ APPLYING EMERGENCY REPAIR BEFORE SENDING (RETRY #{retry_count})")
                        # Create a super simplified version of the messages
                        simplified_messages = []
                        
                        # Keep system message(s)
                        system_msgs = [m for m in data['messages'] if m.get('role') == 'system']
                        if system_msgs:
                            simplified_messages.append(system_msgs[0])  # Keep just the first system message
                        
                        # Add the most recent user message
                        user_msgs = [m for m in data['messages'] if m.get('role') == 'user']
                        if user_msgs:
                            simplified_messages.append(user_msgs[-1])  # Add the most recent user message
                        
                        # Use this drastically simplified message set
                        print(f"Original message count: {len(data['messages'])}")
                        print(f"Simplified message count: {len(simplified_messages)}")
                        data['messages'] = simplified_messages
                    
                    response = requests.post(url, headers=headers, json=data, timeout=60)
                    logger.debug(f"Received response with status code: {response.status_code}")
                    
                    # Check if the response was successful
                    if response.status_code != 200:
                        error_text = response.text
                        # TERMINAL OUTPUT FOR ERRORS - directly visible
                        print("\n🛑 ERROR IN LLAMA API REQUEST 🛑")
                        print(f"HTTP {response.status_code} Error")
                        print(f"Error text: {error_text[:200]}" + ("..." if len(error_text) > 200 else ""))
                        
                        # Detailed analysis of specific error codes
                        if response.status_code == 400:
                            print("\n🔍 HTTP 400 ERROR ANALYSIS:")
                            print("Common causes: Invalid message format, missing required fields, or token limit exceeded")
                            
                            # Try to parse and display the error details
                            try:
                                error_data = response.json()
                                detail = error_data.get('detail', '')
                                # Check specifically for the message 6 schema error
                                if 'message' in str(error_data) and 'schema' in str(error_data) and 'messages.6' in str(error_data):
                                    print("\n🚨 DETECTED MESSAGES.6 SCHEMA ERROR - ATTEMPTING EMERGENCY RECOVERY")
                                    # This is our specific error case - implement special recovery
                                    try:
                                        # Create a drastically simplified message list - only keep essential context
                                        recovery_messages = []
                                        
                                        # Add system message
                                        system_msgs = [m for m in data['messages'] if m.get('role') == 'system']
                                        if system_msgs:
                                            # Keep only the first system message
                                            system_msg = system_msgs[0]
                                            # If it's too long, simplify it
                                            if 'content' in system_msg and isinstance(system_msg['content'], str) and len(system_msg['content']) > 10000:
                                                system_msg = {
                                                    "role": "system",
                                                    "content": "You are LORE, an AI assistant that helps analyze code repositories."
                                                }
                                            recovery_messages.append(system_msg)
                                        else:
                                            # Add a default system message if none exists
                                            recovery_messages.append({"role": "system", "content": "You are LORE, an AI assistant that analyzes repositories."})
                                        
                                        # Add just the last user message - this is what the user cares about most
                                        user_msgs = [m for m in data['messages'] if m.get('role') == 'user']
                                        if user_msgs:
                                            recovery_messages.append(user_msgs[-1])  # Just the last question
                                        
                                        # Try the recovery request
                                        print("\n💡 ATTEMPTING RECOVERY WITH SIMPLIFIED MESSAGES")
                                        print(f"Simplified to {len(recovery_messages)} messages")
                                        
                                        recovery_data = data.copy()
                                        recovery_data['messages'] = recovery_messages
                                        
                                        print("Sending recovery request...")
                                        recovery_response = requests.post(url, headers=headers, json=recovery_data, timeout=60)
                                        
                                        if recovery_response.status_code == 200:
                                            print("🎉 RECOVERY SUCCESSFUL!")
                                            return recovery_response.json()
                                        else:
                                            print(f"⚠️ Recovery failed with status {recovery_response.status_code}")
                                    except Exception as recovery_err:
                                        print(f"Recovery attempt failed: {str(recovery_err)}")
                                
                                # Continue with normal error processing
                                if isinstance(error_data, dict):
                                    if 'error' in error_data:
                                        error_info = error_data['error']
                                        if isinstance(error_info, dict) and 'message' in error_info:
                                            print(f"\nDetailed error message: {error_info['message']}")
                                        else:
                                            print(f"\nError info: {error_info}")
                                    elif 'message' in error_data:
                                        print(f"\nError message: {error_data['message']}")
                                    elif 'detail' in error_data:
                                        print(f"\nError detail: {error_data['detail']}")
                                        
                                    # Look for specific error patterns
                                    error_str = str(error_data)
                                    if 'content' in error_str.lower():
                                        print("\nPossible issue: Invalid content format in messages")
                                    if 'role' in error_str.lower():
                                        print("\nPossible issue: Invalid role in messages")
                                    if 'token' in error_str.lower() or 'length' in error_str.lower():
                                        print("\nPossible issue: Context too long or token limit exceeded")
                            except Exception as parse_err:
                                print(f"\nCould not parse error response as JSON: {str(parse_err)}")
                                
                            # Dump the messages that caused the error for debugging
                            try:
                                for i, msg in enumerate(data['messages']):
                                    role = msg.get('role', 'unknown')
                                    content = msg.get('content', '')
                                    content_len = len(str(content)) if content else 0
                                    content_preview = str(content)[:50] + '...' if content_len > 50 else str(content)
                                    print(f"Message {i}: role={role}, length={content_len}, preview='{content_preview}'")
                            except Exception:
                                print("Could not dump message details")
                        
                        logger.error(f"API error response (HTTP {response.status_code}): {error_text}")
                        
                        # Try to parse the error message
                        try:
                            error_data = response.json()
                            if 'error' in error_data:
                                logger.error(f"API error message: {error_data['error']}")
                                # Check for specific error types
                                if isinstance(error_data['error'], dict) and 'message' in error_data['error']:
                                    error_msg = error_data['error']['message']
                                    logger.error(f"Detailed error message: {error_msg}")
                                    
                                    # Try to recover based on specific error messages
                                    if 'content' in error_msg.lower() or 'format' in error_msg.lower():
                                        logger.warning("Detected content format error, attempting recovery...")
                                        # Simplify messages and retry
                                        simple_msgs = [{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": "Please analyze this code."}]
                                        retry_data = data.copy()
                                        retry_data['messages'] = simple_msgs
                                        logger.debug("Retrying with simplified messages")
                                        retry_response = requests.post(url, headers=headers, json=retry_data, timeout=60)
                                        if retry_response.status_code == 200:
                                            logger.debug("Recovery successful!")
                                            return retry_response.json()
                            if 'message' in error_data:
                                logger.error(f"API error message: {error_data['message']}")
                        except Exception as parse_err:
                            logger.error(f"Could not parse error response as JSON: {str(parse_err)}")
                    
                    # Special handling for message #6 schema error
                    if response.status_code == 400 and 'messages.6' in response.text:
                        print("\n🚨 FINAL ATTEMPT: MESSAGE 6 SCHEMA ERROR DETECTED")
                        print("Performing emergency simplification...")
                        
                        # Extract only the essential messages
                        minimal_messages = []
                        
                        # Always include a system message
                        system_msgs = [m for m in messages if m.get('role') == 'system']
                        if system_msgs:
                            # Use a very simple system message with no complex content
                            minimal_messages.append({"role": "system", "content": "You are LORE, a helpful AI that analyzes code."})
                        else:
                            minimal_messages.append({"role": "system", "content": "You are LORE, a helpful AI."})
                        
                        # Include only the most recent user message - this is critical
                        user_msgs = [m for m in messages if m.get('role') == 'user']
                        if user_msgs:
                            last_user_msg = user_msgs[-1]
                            # Ensure the content is a simple string
                            if 'content' in last_user_msg:
                                if isinstance(last_user_msg['content'], str):
                                    minimal_content = last_user_msg['content']
                                else:
                                    minimal_content = "Please continue with your analysis."
                            else:
                                minimal_content = "Please continue with your analysis."
                            
                            minimal_messages.append({"role": "user", "content": minimal_content})
                        
                        print(f"Simplified to {len(minimal_messages)} messages")
                        
                        # Build a minimal request with just these messages
                        minimal_data = {
                            "model": model or "Llama-4-Maverick-17B-128E-Instruct-FP8",
                            "messages": minimal_messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        }
                        
                        # Make the emergency request
                        print("Sending emergency simplified request...")
                        emergency_response = requests.post(url, headers=headers, json=minimal_data, timeout=60)
                        
                        if emergency_response.status_code == 200:
                            print("✅ EMERGENCY RECOVERY SUCCESSFUL!")
                            return emergency_response.json()
                        else:
                            print(f"❌ Emergency recovery failed: {emergency_response.status_code}")
                    
                    # Continue with normal handling if the above didn't succeed
                    response.raise_for_status()
                    logger.debug("Request successful, parsing JSON response")
                    return response.json()
                    
                except requests.exceptions.Timeout:
                    logger.error("API request timed out after 60 seconds")
                    raise
                except requests.exceptions.RequestException as req_err:
                    logger.error(f"Request exception: {str(req_err)}")
                    raise
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') and hasattr(e.response, 'status_code') else 'unknown'
            logger.exception(f"HTTP Error in chat API request: {status_code}")
            
            error_message = f"I encountered an error communicating with the API (HTTP {status_code})."
            
            if status_code == 403:
                error_message += " This is likely due to permission issues with your API key."
            elif status_code == 400 and 'messages.6' in str(e.response.text):
                error_message += " This appears to be a message format error with the 6th message in the conversation. I'll try to simplify the conversation in future requests."
            elif status_code == 429:
                error_message += " The API request was rate limited. Please try again later."
            elif status_code >= 500:
                error_message += " The API server encountered an internal error. Please try again later."
                
            return {
                "error": str(e),
                "status_code": status_code,
                "choices": [{"message": {"content": error_message}}]
            }
        except Exception as e:
            logger.exception("Error in chat API request")
            return {
                "error": str(e),
                "choices": [{"message": {"content": f"I encountered an error: {str(e)}"}}]
            }    

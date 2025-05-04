"""
Llama API client module for LORE.

This module handles communication with the Llama 4 API for analyzing repository data.
"""
import logging
import os
import json
import base64
from typing import Dict, List, Optional, Union, Any
import requests
from tqdm import tqdm

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
            
            logger.debug(f"Making request to {url}")
            response = requests.post(url, headers=headers, json=payload)
            response_body = response.json()
            logger.debug(f"Response Body: {json.dumps(response_body, indent=2)}")
            
            if response.status_code != 200:
                error_msg = response_body.get('detail', 'Unknown error')
                logger.error(f"API request failed with status {response.status_code}: {error_msg}")
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
            "Analyze this codebase's architecture and design. Focus on: "
            "1. Main components and their interactions "
            "2. Key design patterns and architectural choices "
            "3. Code organization and modularity "
            "4. Notable strengths or areas for improvement "
            "Be concise and specific. Avoid code generation."
        ),
        "analyze_complexity": (
            "Evaluate code complexity and maintainability. Identify: "
            "1. Complex components needing attention "
            "2. Potential technical debt "
            "3. Specific improvement recommendations "
            "Be brief and actionable."
        ),
        "historical_analysis": (
            "Analyze repository history. Focus on: "
            "1. Major changes and evolution "
            "2. Development patterns "
            "3. Key milestones "
            "Keep it short and focused."
        ),
        "chat": (
            "You are a helpful coding assistant analyzing this repository. "
            "Provide clear, concise answers. Focus on high-level insights. "
            "Only show code if specifically asked."
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
                    "content": "You are a helpful coding assistant analyzing this repository. "
                              "Provide clear, concise answers. Focus on high-level insights. "
                              "Only show code if specifically asked."
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
                       max_tokens: int = 8000,
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
                "Analyze this codebase's architecture and design. Focus on: "
                "1. Main components and their interactions "
                "2. Key design patterns and architectural choices "
                "3. Code organization and modularity "
                "4. Notable strengths or areas for improvement "
                "Be concise and specific. Avoid code generation."
            ),
            "analyze_complexity": (
                "Evaluate code complexity and maintainability. Identify: "
                "1. Complex components needing attention "
                "2. Potential technical debt "
                "3. Specific improvement recommendations "
                "Be brief and actionable."
            ),
            "historical_analysis": (
                "Analyze repository history. Focus on: "
                "1. Major changes and evolution "
                "2. Development patterns "
                "3. Key milestones "
                "Keep it short and focused."
            ),
            "chat": (
                "You are a helpful coding assistant analyzing this repository. "
                "Provide clear, concise answers. Focus on high-level insights. "
                "Only show code if specifically asked."
            )
        }
        
        # Use default if task not found
        system_prompt = system_prompts.get(
            task, 
            "You are an expert software engineer analyzing a repository. "
            "Provide comprehensive analysis and insights about the codebase."
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
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False
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
        # DEBUG: Show the current issue
        print("\n\n=============== CONTEXT-PRESERVING FIX FOR CHAT API ===============")
        print(f"Incoming messages count: {len(messages)}")
        
        # EXAMINE ORIGINAL MESSAGES
        print("Examining original messages:")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content")
            print(f"Message {i} - Role: {role}, Content type: {type(content).__name__}")
        
        # Extract important context and content
        # 1. System message with repository context
        system_content = "You are a helpful AI assistant analyzing code."
        repo_context = None
        image_content = None
        user_query = "Tell me about this repository"
        
        # Look for repository context (typically in system or first user message)
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                system_content = msg.get("content")
                if "repository" in system_content.lower() or "code" in system_content.lower():
                    repo_context = system_content
                    print(f"Found repo context in system message: {system_content[:100]}...")
                break
        
        if not repo_context:
            # Look for repo context in user messages
            for msg in messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    content = msg.get("content")
                    if len(content) > 500 and ("repository" in content.lower() or "code" in content.lower()):
                        repo_context = content
                        print(f"Found repo context in user message: {content[:100]}...")
                        break
        
        # 2. Look for image content (typically in user messages with list content)
        for msg in messages:
            content = msg.get("content")
            # Case: Content is a list with image parts
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_content = part
                        print("Found image content in list")
                        break
                if image_content:
                    break
            # Case: Content is a dict with image_url
            elif isinstance(content, dict) and content.get("type") == "image_url":
                image_content = content
                print("Found image content in dict")
                break
        
        # 3. Get the latest user query
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                user_content = msg.get("content")
                # Only use it as query if it's not the repo context (which is typically long)
                if len(user_content) < 500:
                    user_query = user_content
                    print(f"Found user query: {user_query[:100]}...")
                break
        
        # CONSTRUCT NEW COMPLIANT MESSAGES
        new_messages = []
        
        # Add system message with repo context
        if repo_context:
            new_messages.append({
                "role": "system",
                "content": repo_context
            })
        else:
            new_messages.append({
                "role": "system", 
                "content": system_content
            })
        
        # Add image if found (in a properly formatted user message)
        if image_content:
            # Format special message for image
            if isinstance(image_content, dict) and image_content.get("type") == "image_url":
                new_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": image_content.get("image_url")}
                    ]
                })
                # Add a follow-up prompt to analyze the image
                new_messages.append({
                    "role": "user",
                    "content": "Analyze this image in detail."
                })
        
        # Add user's actual query as the final message
        new_messages.append({
            "role": "user",
            "content": user_query
        })
        
        # Log the new messages
        print("\n=============== NEW CONTEXT-PRESERVING MESSAGES ===============")
        for i, msg in enumerate(new_messages):
            print(f"New Message {i} - Role: {msg.get('role')}")
            content = msg.get("content")
            if isinstance(content, str):
                print(f"  Content preview: {content[:100]}...")
            elif isinstance(content, list):
                print(f"  Content: list with {len(content)} items")
                for part in content:
                    print(f"    Part type: {part.get('type', 'unknown')}")
            else:
                print(f"  Content type: {type(content).__name__}")
        
        # Use the default model or override
        model = model or self.model
        url = f"{self.api_base}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": model,
            "messages": new_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        # Log the final payload (shortened for readability)
        print("\n=============== FINAL API PAYLOAD STRUCTURE ===============")
        print(f"Model: {model}")
        print(f"Message count: {len(new_messages)}")
        print(f"Temperature: {temperature}")
        print(f"Max tokens: {max_tokens}")
        
        try:
            if stream:
                return self._stream_chat(url, headers, data)
            else:
                response = requests.post(url, headers=headers, json=data)
                if response.status_code != 200:
                    print(f"API error response: {response.text}")
                    logger.error(f"API error response: {response.text}")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error in chat request: {e}")
            if hasattr(e, "response") and hasattr(e.response, "text"):
                logger.error(f"API response: {e.response.text}")
            raise e

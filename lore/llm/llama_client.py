"""
Llama API client module for LORE.

This module handles communication with the Llama 4 API for analyzing repository data.
"""
import logging
import os
import json
from typing import Dict, List, Optional, Union, Any

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        
        logger.debug(f"API Key: {self.api_key}")
        logger.debug(f"API Base: {self.api_base}")
        
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
    
    def _sanitize_response(self, text: str) -> str:
        """
        Clean up the response text to remove repetitive or nonsensical content.
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned response text
        """
        # Handle non-string input
        if not isinstance(text, str):
            if isinstance(text, dict):
                if 'text' in text:
                    text = text['text']
                elif 'content' in text:
                    text = text['content']
                else:
                    logger.warning(f"Unexpected dict format in _sanitize_response: {text}")
                    return str(text)
            else:
                logger.warning(f"Unexpected response type in _sanitize_response: {type(text)}")
                return str(text)
                
        if not text:
            return text
            
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

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
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the Llama API.
        
        Args:
            endpoint: API endpoint
            data: Request payload
            
        Returns:
            API response as dictionary
        """
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.debug("=== API Request Details ===")
        logger.debug(f"URL: {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Payload: {json.dumps(data, indent=2)}")
        
        try:
            logger.debug("Sending POST request to API...")
            response = requests.post(url, headers=headers, json=data)
            
            logger.debug("=== API Response Details ===")
            logger.debug(f"Status Code: {response.status_code}")
            logger.debug(f"Response Headers: {dict(response.headers)}")
            
            try:
                response_json = response.json()
                logger.debug(f"Response Body: {json.dumps(response_json, indent=2)}")
                
                # Transform response to OpenAI format if needed
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    choice = response_json['choices'][0]
                    if 'text' in choice and 'message' not in choice:
                        # Clean up response text
                        sanitized_text = self._sanitize_response(choice['text'])
                        # Convert to OpenAI format
                        choice['message'] = {'content': sanitized_text}
                        del choice['text']
                    elif 'content' in choice and 'message' not in choice:
                        # Clean up response text
                        sanitized_text = self._sanitize_response(choice['content'])
                        # Convert to OpenAI format
                        choice['message'] = {'content': sanitized_text}
                        del choice['content']
                    elif 'message' in choice and 'content' in choice['message']:
                        # Clean up response text in OpenAI format
                        choice['message']['content'] = self._sanitize_response(choice['message']['content'])
                
                return response_json
                
            except json.JSONDecodeError:
                logger.error("Failed to parse response as JSON")
                logger.debug(f"Raw Response Text: {response.text}")
                raise
            
            response.raise_for_status()
            return response_json
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Error response body: {e.response.text}")
            raise
    
    def _sanitize_response(self, text: str) -> str:
        """
        Sanitize the response text to remove repetitive patterns and truncate if too long.
        """
        if not text:
            return text
            
        # Detect repetitive patterns (3 or more repetitions)
        pattern_length = 10  # Look for patterns of this length or longer
        for i in range(pattern_length, len(text) // 2):
            pattern = text[0:i]
            if text.count(pattern) >= 3:
                # Found a repetitive pattern, truncate at first occurrence
                end_idx = text.find(pattern) + len(pattern)
                return text[:end_idx]
                
        # Check for repetitive code blocks
        if text.count('```python') > 2:
            # Too many code blocks, truncate after second one
            parts = text.split('```python')
            return '```python'.join(parts[:3])
            
        return text
        
    SYSTEM_PROMPTS = {
        "analyze_architecture": (
            "You are an expert software architect analyzing a codebase. "
            "Provide a comprehensive analysis of the repository architecture, design patterns, "
            "and code organization. Identify strengths, weaknesses, architectural drift, and anti-patterns. "
            "Focus on high-level insights and avoid generating code snippets unless specifically asked. "
            "Keep responses concise and to the point."
        ),
        "analyze_history": (
            "You are an expert in software development history analysis. "
            "Analyze the Git history to understand the evolution of the codebase, key milestones, "
            "and development patterns. Focus on meaningful insights about code changes and development "
            "trends. Avoid generating code snippets. Keep responses concise and to the point."
        ),
        "chat": (
            "You are LORE (Long-context Organizational Repository Explorer), an expert AI assistant "
            "that helps developers understand codebases. You have been provided with the repository context. "
            "Provide clear, concise, and accurate answers. Keep responses focused and relevant to the "
            "questions asked. If you don't know something, say so directly. Do not generate code unless "
            "specifically asked. End your response when you've fully answered the question."
        )
    }
    
    def analyze_repository(
        self,
        content: str,
        model: str = "llama-2-70b-chat",  # Updated to standard model name
        temperature: float = 0.7,
        max_tokens: int = 2000,
        task: str = "analyze_architecture",
        messages: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a repository using the Llama API.
        
        Args:
            content: Content to analyze
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            task: Task type (analyze_architecture, historical_analysis, etc.)
            messages: Optional list of chat messages
            
        Returns:
            API response
        """
        try:
            if messages is not None:
                logger.debug("Using provided messages for chat mode")
                # Validate message format
                for msg in messages:
                    if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                        raise ValueError("Each message must be a dict with 'role' and 'content' keys")
                    if msg['role'] not in ['system', 'user', 'assistant']:
                        raise ValueError("Message role must be one of: system, user, assistant")
                
                # Use provided messages for chat mode
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": ["<|end|>"]  # Add stop sequence
                }
            else:
                logger.debug("Using standard format for analysis")
                # Use standard format for analysis
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPTS.get(task, self.SYSTEM_PROMPTS["analyze_architecture"])},
                        {"role": "user", "content": content}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": ["<|end|>"]  # Add stop sequence
                }
            
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
            return self._make_request("chat/completions", payload)
            
        except Exception as e:
            logger.error(f"Error in analyze_repository: {str(e)}")
            raise
    
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
                "You are an expert software architect analyzing a codebase. "
                "Provide a comprehensive analysis of the repository architecture, design patterns, "
                "and code organization. Identify strengths, weaknesses, architectural drift, and anti-patterns. "
                "Focus on high-level insights and avoid generating code snippets unless specifically asked. "
                "Keep responses concise and to the point."
            ),
            "analyze_history": (
                "You are an expert in software development history analysis. "
                "Analyze the Git history to understand the evolution of the codebase, key milestones, "
                "and development patterns. Focus on meaningful insights about code changes and development "
                "trends. Avoid generating code snippets. Keep responses concise and to the point."
            ),
            "chat": (
                "You are LORE (Long-context Organizational Repository Explorer), an expert AI assistant "
                "that helps developers understand codebases. You have been provided with the repository context. "
                "Provide clear, concise, and accurate answers. Keep responses focused and relevant to the "
                "questions asked. If you don't know something, say so directly. Do not generate code unless "
                "specifically asked. End your response when you've fully answered the question."
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
                         model: str = "llama-2-70b-chat",  # Updated to standard model name
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

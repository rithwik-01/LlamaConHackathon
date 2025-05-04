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
        self.api_base = api_base or os.environ.get("LLAMA_API_BASE", "https://api.llama.ai/v1")
        
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
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()
    
    def analyze_repository(self, 
                          content: str, 
                          model: str = "llama-4-10m", 
                          temperature: float = 0.2,
                          max_tokens: int = 8000,
                          task: str = "analyze_architecture",
                          messages: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Send repository data to Llama for analysis.
        
        Args:
            content: Repository content to analyze
            model: Llama model to use (defaults to llama-4-10m)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            task: Analysis task to perform
            messages: Optional list of messages for chat mode
            
        Returns:
            Llama API response
        """
        # Define system prompts based on task
        system_prompts = {
            "analyze_architecture": (
                "You are an expert software architect analyzing a codebase. "
                "Provide a comprehensive analysis of the repository architecture, "
                "design patterns, and code organization. Identify strengths, weaknesses, "
                "architectural drift, and anti-patterns."
            ),
            "historical_analysis": (
                "You are an expert software historian examining the evolution of a codebase. "
                "Analyze the commit history, changes over time, and key developmental milestones. "
                "Identify important architectural decisions, pivots, and the reasoning behind them."
            ),
            "onboarding": (
                "You are an expert software developer creating comprehensive documentation "
                "for new team members. Create an onboarding guide that explains the repository "
                "structure, key components, workflow, and development practices."
            ),
            "refactoring_guide": (
                "You are an expert software refactoring consultant. Examine the codebase "
                "and provide a detailed refactoring plan. Identify areas that need improvement, "
                "technical debt, and prioritized actionable steps for refactoring."
            ),
            "dependency_analysis": (
                "You are an expert software architect analyzing dependencies. "
                "Map out all internal and external dependencies in the codebase, "
                "highlight critical paths, identify tight coupling, and suggest "
                "improvements for better modularity."
            ),
            "chat": (
                "You are LORE (Long-context Organizational Repository Explorer), an expert AI assistant "
                "that helps developers understand codebases. You have been provided with the full context "
                "of a repository including its code, Git history, documentation, and potentially issues and PRs. "
                "Answer questions about the codebase based on this context. Be specific and reference relevant "
                "parts of the code when appropriate. If you don't know the answer, say so."
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
                "max_tokens": max_tokens
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
                "max_tokens": max_tokens
            }
        
        try:
            return self._make_request("chat/completions", payload)
        except Exception as e:
            logger.error(f"Error calling Llama API: {e}")
            raise
    
    def stream_analysis(self, 
                       content: str, 
                       model: str = "llama-4-10m", 
                       temperature: float = 0.2,
                       max_tokens: int = 8000,
                       task: str = "analyze_architecture",
                       messages: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Stream repository analysis from Llama API.
        
        Args:
            content: Repository content to analyze
            model: Llama model to use (defaults to llama-4-10m)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            task: Analysis task to perform
            messages: Optional list of messages for chat mode
            
        Returns:
            Complete response after streaming
        """
        # Define system prompts based on task (same as above)
        system_prompts = {
            "analyze_architecture": (
                "You are an expert software architect analyzing a codebase. "
                "Provide a comprehensive analysis of the repository architecture, "
                "design patterns, and code organization. Identify strengths, weaknesses, "
                "architectural drift, and anti-patterns."
            ),
            "historical_analysis": (
                "You are an expert software historian examining the evolution of a codebase. "
                "Analyze the commit history, changes over time, and key developmental milestones. "
                "Identify important architectural decisions, pivots, and the reasoning behind them."
            ),
            "onboarding": (
                "You are an expert software developer creating comprehensive documentation "
                "for new team members. Create an onboarding guide that explains the repository "
                "structure, key components, workflow, and development practices."
            ),
            "refactoring_guide": (
                "You are an expert software refactoring consultant. Examine the codebase "
                "and provide a detailed refactoring plan. Identify areas that need improvement, "
                "technical debt, and prioritized actionable steps for refactoring."
            ),
            "dependency_analysis": (
                "You are an expert software architect analyzing dependencies. "
                "Map out all internal and external dependencies in the codebase, "
                "highlight critical paths, identify tight coupling, and suggest "
                "improvements for better modularity."
            ),
            "chat": (
                "You are LORE (Long-context Organizational Repository Explorer), an expert AI assistant "
                "that helps developers understand codebases. You have been provided with the full context "
                "of a repository including its code, Git history, documentation, and potentially issues and PRs. "
                "Answer questions about the codebase based on this context. Be specific and reference relevant "
                "parts of the code when appropriate. If you don't know the answer, say so."
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
                         model: str = "llama-4-10m",
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

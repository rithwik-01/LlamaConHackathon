"""
Web content fetcher for LORE.

This module handles fetching content from web URLs for use in repository analysis.
"""
import logging
import requests
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def fetch_url_content(url: str) -> Optional[str]:
    """
    Fetch content from a URL.
    
    Args:
        url: URL to fetch content from
        
    Returns:
        Content as string, or None if fetch fails
    """
    if not is_valid_url(url):
        logger.error(f"Invalid URL: {url}")
        return None
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Handle HTML content - extract readable text
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        
        # Handle plain text or Markdown
        elif 'text/plain' in content_type or 'text/markdown' in content_type:
            return response.text
        
        # For other content types, just return the raw text
        else:
            return response.text
            
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None

def fetch_multiple_urls(urls: List[str]) -> Dict[str, str]:
    """
    Fetch content from multiple URLs.
    
    Args:
        urls: List of URLs to fetch content from
        
    Returns:
        Dictionary mapping URLs to their content
    """
    result = {}
    
    for url in urls:
        content = fetch_url_content(url)
        if content:
            result[url] = content
    
    return result

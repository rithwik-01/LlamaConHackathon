"""
Repository utilities for LORE.

This module provides utilities for working with repositories.
"""
import os
import tempfile
import re
import logging
from pathlib import Path
from typing import Optional, Tuple
import subprocess

import git

logger = logging.getLogger(__name__)

def is_github_url(url: str) -> bool:
    """
    Check if a URL is a valid GitHub repository URL.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a valid GitHub repository URL, False otherwise
    """
    github_patterns = [
        r'^https?://github\.com/[^/]+/[^/]+/?$',
        r'^https?://github\.com/[^/]+/[^/]+\.git$',
        r'^git@github\.com:[^/]+/[^/]+\.git$'
    ]
    
    return any(re.match(pattern, url) for pattern in github_patterns)

def extract_repo_info(url: str) -> Tuple[str, str]:
    """
    Extract repository owner and name from a GitHub URL.
    
    Args:
        url: GitHub repository URL
        
    Returns:
        Tuple of (owner, name)
    """
    # Handle HTTPS URLs
    https_match = re.match(r'^https?://github\.com/([^/]+)/([^/.]+)', url)
    if https_match:
        return https_match.group(1), https_match.group(2)
    
    # Handle SSH URLs
    ssh_match = re.match(r'^git@github\.com:([^/]+)/([^/.]+)', url)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)
    
    raise ValueError(f"Could not extract repository information from URL: {url}")

def clone_github_repo(url: str, target_dir: Optional[str] = None) -> str:
    """
    Clone a GitHub repository.
    
    Args:
        url: GitHub repository URL
        target_dir: Directory to clone into (temporary directory if None)
        
    Returns:
        Path to the cloned repository
    """
    if not is_github_url(url):
        raise ValueError(f"Invalid GitHub URL: {url}")
    
    # Create target directory if not provided
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="lore_repo_")
    
    try:
        logger.info(f"Cloning repository from {url} to {target_dir}")
        git.Repo.clone_from(url, target_dir)
        logger.info(f"Repository cloned successfully")
        return target_dir
    except git.GitCommandError as e:
        logger.error(f"Error cloning repository: {e}")
        raise

def get_default_branch(repo_path: str) -> str:
    """
    Get the default branch of a repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Name of the default branch (usually 'main' or 'master')
    """
    try:
        repo = git.Repo(repo_path)
        return repo.active_branch.name
    except git.GitCommandError as e:
        logger.error(f"Error getting default branch: {e}")
        # Fallback to common default branch names
        for branch in ['main', 'master']:
            if branch in repo.heads:
                return branch
        return 'master'  # Final fallback

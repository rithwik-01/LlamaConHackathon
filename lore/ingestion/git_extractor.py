"""
Git repository data extraction module for LORE.

This module handles extracting code, commit history, and other data from Git repositories.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

import git
from tqdm import tqdm

logger = logging.getLogger(__name__)

class GitExtractor:
    """Extract information from Git repositories for analysis."""
    
    def __init__(self, repo_path: str, max_history: Optional[int] = None):
        """
        Initialize the Git extractor.
        
        Args:
            repo_path: Path to the Git repository
            max_history: Maximum number of commits to extract (None for all)
        """
        self.repo_path = Path(repo_path).absolute()
        self.max_history = max_history
        
        if not self._is_git_repo(self.repo_path):
            raise ValueError(f"Path {repo_path} is not a valid Git repository")
        
        self.repo = git.Repo(self.repo_path)
        
    @staticmethod
    def _is_git_repo(path: Path) -> bool:
        """Check if a directory is a Git repository."""
        git_dir = path / ".git"
        return git_dir.exists() and git_dir.is_dir()
    
    def get_file_contents(self, ignore_patterns: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Get the contents of all files in the repository.
        
        Args:
            ignore_patterns: List of glob patterns to ignore
            
        Returns:
            Dictionary mapping file paths to their contents
        """
        if ignore_patterns is None:
            ignore_patterns = [".git/", "*.pyc", "__pycache__/", "*.so", "*.o", "node_modules/"]
            
        file_contents = {}
        
        for root, _, files in os.walk(self.repo_path):
            rel_root = Path(root).relative_to(self.repo_path)
            
            # Skip ignored directories/patterns
            if any(str(rel_root).startswith(pattern.rstrip("/")) 
                   for pattern in ignore_patterns if pattern.endswith("/")):
                continue
                
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.repo_path)
                
                # Skip ignored file patterns
                if any(rel_path.match(pattern) for pattern in ignore_patterns):
                    continue
                    
                try:
                    # Only read text files, skip binary files
                    if self._is_text_file(file_path):
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            file_contents[str(rel_path)] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    
        return file_contents
    
    @staticmethod
    def _is_text_file(file_path: Path) -> bool:
        """Determine if a file is a text file (vs binary)."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' not in chunk
        except Exception:
            return False
            
    def get_commit_history(self) -> List[Dict]:
        """
        Get the commit history of the repository.
        
        Returns:
            List of dictionaries containing commit information
        """
        commits = []
        commit_iter = list(self.repo.iter_commits())
        
        if self.max_history:
            commit_iter = commit_iter[:self.max_history]
            
        for commit in tqdm(commit_iter, desc="Processing commits"):
            try:
                # Get commit data
                commit_data = {
                    'hash': commit.hexsha,
                    'author': f"{commit.author.name} <{commit.author.email}>",
                    'authored_date': datetime.fromtimestamp(commit.authored_date),
                    'committer': f"{commit.committer.name} <{commit.committer.email}>",
                    'committed_date': datetime.fromtimestamp(commit.committed_date),
                    'message': commit.message,
                    'summary': commit.summary,
                    'files_changed': [],
                    'insertions': 0,
                    'deletions': 0,
                }
                
                # Get diff stats for files changed
                if commit.parents:
                    diff_index = commit.parents[0].diff(commit)
                    for diff in diff_index:
                        try:
                            file_path = diff.b_path if diff.b_path else diff.a_path
                            stats = diff.stats
                            commit_data['files_changed'].append({
                                'path': file_path,
                                'insertions': stats.get('insertions', 0),
                                'deletions': stats.get('deletions', 0),
                                'lines': stats.get('lines', 0),
                                'type': self._get_change_type(diff),
                            })
                            commit_data['insertions'] += stats.get('insertions', 0)
                            commit_data['deletions'] += stats.get('deletions', 0)
                        except Exception as e:
                            logger.warning(f"Error processing diff in commit {commit.hexsha}: {e}")
                
                commits.append(commit_data)
            except Exception as e:
                logger.warning(f"Error processing commit {commit.hexsha}: {e}")
                
        return commits
    
    @staticmethod
    def _get_change_type(diff) -> str:
        """Determine the type of change represented by a diff."""
        if diff.new_file:
            return "added"
        elif diff.deleted_file:
            return "deleted"
        elif diff.renamed:
            return "renamed"
        else:
            return "modified"
            
    def get_branches(self) -> List[Dict]:
        """
        Get information about all branches in the repository.
        
        Returns:
            List of dictionaries containing branch information
        """
        branches = []
        
        for branch in self.repo.branches:
            try:
                branches.append({
                    'name': branch.name,
                    'commit': branch.commit.hexsha,
                    'is_active': branch.name == self.repo.active_branch.name,
                })
            except Exception as e:
                logger.warning(f"Error processing branch {branch}: {e}")
                
        return branches
    
    def get_documentation(self) -> Dict[str, str]:
        """
        Extract documentation files (Markdown, reStructuredText, etc.).
        
        Returns:
            Dictionary mapping file paths to their contents
        """
        doc_extensions = {'.md', '.rst', '.txt', '.wiki', '.adoc'}
        doc_filenames = {'README', 'CONTRIBUTING', 'CHANGELOG', 'LICENSE', 'SECURITY', 'CODE_OF_CONDUCT'}
        
        documentation = {}
        
        for root, _, files in os.walk(self.repo_path):
            rel_root = Path(root).relative_to(self.repo_path)
            
            # Check for docs directories
            if any(part.lower() in ('docs', 'documentation', 'wiki') for part in rel_root.parts):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.repo_path)
                    
                    if file_path.suffix.lower() in doc_extensions:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                documentation[str(rel_path)] = f.read()
                        except Exception as e:
                            logger.warning(f"Failed to read documentation file {file_path}: {e}")
            
            # Check for common documentation files in all directories
            for file in files:
                filename = Path(file).stem.upper()
                if filename in doc_filenames:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.repo_path)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            documentation[str(rel_path)] = f.read()
                    except Exception as e:
                        logger.warning(f"Failed to read documentation file {file_path}: {e}")
        
        return documentation

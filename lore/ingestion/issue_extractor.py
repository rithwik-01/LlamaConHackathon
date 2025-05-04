"""
Issue and PR extraction module for LORE.

This module handles retrieving issues and pull requests from GitHub, GitLab, or other platforms.
"""
import logging
import time
from typing import Dict, List, Optional
import os

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

class IssueExtractor:
    """Extract issues and pull requests from code hosting platforms."""
    
    def __init__(self, platform: str = 'github', token: Optional[str] = None):
        """
        Initialize the issue extractor.
        
        Args:
            platform: The platform to extract issues from ('github' or 'gitlab')
            token: Authentication token for the API
        """
        self.platform = platform.lower()
        self.token = token or os.environ.get(f"{self.platform.upper()}_TOKEN")
        
        if not self.token:
            logger.warning(f"No {platform} token provided. API rate limits will be restricted.")
        
        if self.platform == 'github':
            self.api_base_url = 'https://api.github.com'
        elif self.platform == 'gitlab':
            self.api_base_url = 'https://gitlab.com/api/v4'
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make an authenticated request to the API.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        headers = {}
        if self.token:
            if self.platform == 'github':
                headers['Authorization'] = f"token {self.token}"
            elif self.platform == 'gitlab':
                headers['PRIVATE-TOKEN'] = self.token
        
        url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
        response = requests.get(url, headers=headers, params=params)
        
        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Rate limited. Waiting for {retry_after} seconds.")
            time.sleep(retry_after)
            return self._make_request(endpoint, params)
        
        response.raise_for_status()
        return response.json()
    
    def get_issues(self, repo_owner: str, repo_name: str, 
                  state: str = 'all', max_issues: Optional[int] = None) -> List[Dict]:
        """
        Get issues from a repository.
        
        Args:
            repo_owner: Owner of the repository
            repo_name: Name of the repository
            state: Issue state ('open', 'closed', or 'all')
            max_issues: Maximum number of issues to retrieve
            
        Returns:
            List of dictionaries containing issue information
        """
        issues = []
        page = 1
        per_page = 100
        
        if self.platform == 'github':
            endpoint = f"/repos/{repo_owner}/{repo_name}/issues"
            params = {
                'state': state,
                'per_page': per_page,
                'page': page,
                # Include PRs unless they're requested separately
                'pull_request': True
            }
        elif self.platform == 'gitlab':
            endpoint = f"/projects/{repo_owner}%2F{repo_name}/issues"
            params = {
                'state': state,
                'per_page': per_page,
                'page': page
            }
        
        with tqdm(desc="Retrieving issues", unit="page") as pbar:
            while True:
                params['page'] = page
                try:
                    page_issues = self._make_request(endpoint, params)
                    
                    if not page_issues:
                        break
                    
                    issues.extend(page_issues)
                    pbar.update(1)
                    
                    if max_issues and len(issues) >= max_issues:
                        issues = issues[:max_issues]
                        break
                    
                    page += 1
                except Exception as e:
                    logger.error(f"Error retrieving issues: {e}")
                    break
        
        return issues
    
    def get_pull_requests(self, repo_owner: str, repo_name: str,
                         state: str = 'all', max_prs: Optional[int] = None) -> List[Dict]:
        """
        Get pull requests from a repository.
        
        Args:
            repo_owner: Owner of the repository
            repo_name: Name of the repository
            state: PR state ('open', 'closed', 'all')
            max_prs: Maximum number of PRs to retrieve
            
        Returns:
            List of dictionaries containing PR information
        """
        prs = []
        page = 1
        per_page = 100
        
        if self.platform == 'github':
            endpoint = f"/repos/{repo_owner}/{repo_name}/pulls"
            params = {
                'state': state,
                'per_page': per_page,
                'page': page
            }
        elif self.platform == 'gitlab':
            endpoint = f"/projects/{repo_owner}%2F{repo_name}/merge_requests"
            params = {
                'state': state,
                'per_page': per_page,
                'page': page
            }
        
        with tqdm(desc="Retrieving pull requests", unit="page") as pbar:
            while True:
                params['page'] = page
                try:
                    page_prs = self._make_request(endpoint, params)
                    
                    if not page_prs:
                        break
                    
                    prs.extend(page_prs)
                    pbar.update(1)
                    
                    if max_prs and len(prs) >= max_prs:
                        prs = prs[:max_prs]
                        break
                    
                    page += 1
                except Exception as e:
                    logger.error(f"Error retrieving pull requests: {e}")
                    break
        
        return prs
    
    def get_pr_comments(self, repo_owner: str, repo_name: str, pr_number: int) -> List[Dict]:
        """
        Get comments on a pull request.
        
        Args:
            repo_owner: Owner of the repository
            repo_name: Name of the repository
            pr_number: Pull request number
            
        Returns:
            List of dictionaries containing comment information
        """
        if self.platform == 'github':
            endpoint = f"/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/comments"
        elif self.platform == 'gitlab':
            endpoint = f"/projects/{repo_owner}%2F{repo_name}/merge_requests/{pr_number}/notes"
        
        try:
            return self._make_request(endpoint)
        except Exception as e:
            logger.error(f"Error retrieving PR comments: {e}")
            return []
    
    def enrich_pr_with_comments(self, repo_owner: str, repo_name: str, pr: Dict) -> Dict:
        """
        Add comments to a pull request.
        
        Args:
            repo_owner: Owner of the repository
            repo_name: Name of the repository
            pr: Pull request dictionary
            
        Returns:
            Enriched pull request dictionary
        """
        pr_number = pr['number'] if self.platform == 'github' else pr['iid']
        pr['comments'] = self.get_pr_comments(repo_owner, repo_name, pr_number)
        return pr

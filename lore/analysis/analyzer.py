"""
Repository analyzer module for LORE.

This module handles analyzing repository data using the Llama API.
"""
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from ..llm.llama_client import LlamaClient
from ..ingestion.git_extractor import GitExtractor
from ..ingestion.issue_extractor import IssueExtractor

logger = logging.getLogger(__name__)

class RepositoryAnalyzer:
    """Analyze repository data using Llama 4."""
    
    def __init__(self, 
                llm_client: LlamaClient,
                output_dir: Optional[str] = None):
        """
        Initialize the repository analyzer.
        
        Args:
            llm_client: Llama API client
            output_dir: Directory to save analysis results
        """
        self.llm_client = llm_client
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
    
    def prepare_repository_context(self, 
                                  git_extractor: GitExtractor, 
                                  issue_extractor: Optional[IssueExtractor] = None,
                                  repo_owner: Optional[str] = None,
                                  repo_name: Optional[str] = None,
                                  include_issues: bool = False,
                                  include_prs: bool = False) -> str:
        """
        Prepare the repository context for analysis.
        
        Args:
            git_extractor: Git repository extractor
            issue_extractor: Issue and PR extractor
            repo_owner: Repository owner (for issue extraction)
            repo_name: Repository name (for issue extraction)
            include_issues: Whether to include issues
            include_prs: Whether to include pull requests
            
        Returns:
            Repository context as a string
        """
        context_parts = []
        
        # Add repository information
        context_parts.append("# REPOSITORY ANALYSIS CONTEXT\n\n")
        context_parts.append(f"Repository path: {git_extractor.repo_path}\n\n")
        
        # Get file contents
        file_contents = git_extractor.get_file_contents()
        context_parts.append(f"## FILE CONTENTS ({len(file_contents)} files)\n\n")
        
        for file_path, content in file_contents.items():
            # Skip binary files or very large files
            if len(content) > 100000:  # Skip files larger than 100KB
                context_parts.append(f"### {file_path}\n\n(File too large, skipped)\n\n")
                continue
                
            context_parts.append(f"### {file_path}\n\n```\n{content}\n```\n\n")
        
        # Get commit history
        commits = git_extractor.get_commit_history()
        context_parts.append(f"## COMMIT HISTORY ({len(commits)} commits)\n\n")
        
        for commit in commits[:100]:  # Limit to 100 commits to avoid context overflow
            context_parts.append(
                f"### Commit {commit['hash'][:8]}\n\n"
                f"Author: {commit['author']}\n"
                f"Date: {commit['authored_date']}\n"
                f"Message: {commit['message']}\n"
                f"Files changed: {len(commit['files_changed'])}\n"
                f"Insertions: {commit['insertions']}, Deletions: {commit['deletions']}\n\n"
            )
        
        # Get documentation
        documentation = git_extractor.get_documentation()
        if documentation:
            context_parts.append(f"## DOCUMENTATION ({len(documentation)} files)\n\n")
            
            for doc_path, content in documentation.items():
                context_parts.append(f"### {doc_path}\n\n{content}\n\n")
        
        # Get issues and PRs if requested
        if include_issues and issue_extractor and repo_owner and repo_name:
            try:
                issues = issue_extractor.get_issues(repo_owner, repo_name, max_issues=50)
                context_parts.append(f"## ISSUES ({len(issues)} issues)\n\n")
                
                for issue in issues:
                    title = issue.get('title', 'No title')
                    number = issue.get('number', 0)
                    state = issue.get('state', 'unknown')
                    body = issue.get('body', 'No description')
                    
                    context_parts.append(
                        f"### Issue #{number}: {title} ({state})\n\n"
                        f"{body}\n\n"
                    )
            except Exception as e:
                logger.error(f"Error retrieving issues: {e}")
        
        if include_prs and issue_extractor and repo_owner and repo_name:
            try:
                prs = issue_extractor.get_pull_requests(repo_owner, repo_name, max_prs=50)
                context_parts.append(f"## PULL REQUESTS ({len(prs)} PRs)\n\n")
                
                for pr in prs:
                    title = pr.get('title', 'No title')
                    number = pr.get('number', 0)
                    state = pr.get('state', 'unknown')
                    body = pr.get('body', 'No description')
                    
                    context_parts.append(
                        f"### PR #{number}: {title} ({state})\n\n"
                        f"{body}\n\n"
                    )
            except Exception as e:
                logger.error(f"Error retrieving pull requests: {e}")
        
        return "".join(context_parts)
    
    def analyze_repository(self,
                          context: str,
                          task: str = "analyze_architecture",
                          model: str = "llama-4-10m") -> Dict[str, Any]:
        """
        Analyze a repository using Llama 4.
        
        Args:
            context: Repository context to analyze
            task: Analysis task to perform
            model: Llama model to use
            
        Returns:
            Analysis results
        """
        # Log the context size
        logger.info(f"Analyzing repository with context size: {len(context)} characters")
        
        # Check if context is too large for a single request
        if len(context) > 300000:  # Assuming 300K chars is our limit
            logger.info("Context too large, chunking and analyzing separately")
            results = self.llm_client.chunk_and_analyze(
                context,
                model=model,
                task=task
            )
            
            # Save intermediate results if output_dir is specified
            if self.output_dir:
                chunks_dir = self.output_dir / "chunks"
                chunks_dir.mkdir(exist_ok=True)
                
                for i, result in enumerate(results):
                    chunk_file = chunks_dir / f"chunk_{i+1}_analysis.json"
                    with open(chunk_file, 'w') as f:
                        json.dump(result, f, indent=2)
            
            # Combine results
            combined_analysis = self._combine_chunked_analyses(results, task)
            
            # Save combined results
            if self.output_dir:
                combined_file = self.output_dir / f"{task}_analysis.json"
                with open(combined_file, 'w') as f:
                    json.dump(combined_analysis, f, indent=2)
            
            return combined_analysis
        else:
            # Analyze in a single request
            result = self.llm_client.analyze_repository(
                context,
                model=model,
                task=task
            )
            
            # Save result if output_dir is specified
            if self.output_dir:
                result_file = self.output_dir / f"{task}_analysis.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
            return result
    
    def _combine_chunked_analyses(self, 
                                 results: List[Dict[str, Any]], 
                                 task: str) -> Dict[str, Any]:
        """
        Combine chunked analysis results.
        
        Args:
            results: List of analysis results
            task: Original analysis task
            
        Returns:
            Combined analysis result
        """
        # Extract content from each result
        contents = []
        for result in results:
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                contents.append(content)
            elif 'content' in result:
                contents.append(result['content'])
        
        # Combine contents
        combined_content = "\n\n".join(contents)
        
        # Perform a meta-analysis to synthesize the results
        meta_prompt = (
            f"Below are separate analyses of different parts of the same code repository. "
            f"Synthesize these analyses into a cohesive {task} that covers the entire codebase. "
            f"Eliminate redundancies and reconcile any contradictions.\n\n"
            f"{combined_content}"
        )
        
        meta_analysis = self.llm_client.analyze_repository(
            meta_prompt,
            task=f"synthesize_{task}"
        )
        
        return meta_analysis
    
    def generate_report(self, analysis: Dict[str, Any], report_format: str = "markdown") -> str:
        """
        Generate a report from the analysis results.
        
        Args:
            analysis: Analysis results
            report_format: Report format ("markdown" or "html")
            
        Returns:
            Formatted report
        """
        # Extract content from the analysis
        if 'choices' in analysis and len(analysis['choices']) > 0:
            content = analysis['choices'][0]['message']['content']
        elif 'content' in analysis:
            content = analysis['content']
        else:
            content = "No analysis content found."
        
        if report_format == "markdown":
            report = content  # Already in markdown format
        elif report_format == "html":
            # Convert markdown to HTML
            from markdown import markdown
            html_content = markdown(content)
            
            report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LORE Repository Analysis</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }}
                    h1, h2, h3 {{ color: #333; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                    code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
                </style>
            </head>
            <body>
                <h1>LORE Repository Analysis</h1>
                {html_content}
            </body>
            </html>
            """
        else:
            raise ValueError(f"Unsupported report format: {report_format}")
        
        # Save report if output_dir is specified
        if self.output_dir:
            extension = "md" if report_format == "markdown" else "html"
            report_file = self.output_dir / f"repository_analysis.{extension}"
            with open(report_file, 'w') as f:
                f.write(report)
        
        return report

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
        
        # Get commit history - get all commits
        commits = git_extractor.get_commit_history()
        significant_commits = [
            commit for commit in commits  # Remove limit, use all commits
            if len(commit['files_changed']) > 0 and  # Has actual changes
            not any(word in commit['message'].lower() for word in ['merge', 'typo', 'fix typo', 'formatting'])
        ]
        
        # Track which files were changed in commits
        changed_files = {}
        for commit in significant_commits:
            for file_change in commit['files_changed']:
                file_path = file_change['path']
                if file_path not in changed_files:
                    changed_files[file_path] = []
                changed_files[file_path].append({
                    'hash': commit['hash'][:8],
                    'message': commit['message'],
                    'date': commit['authored_date'],
                    'insertions': file_change['insertions'],
                    'deletions': file_change['deletions']
                })
        
        # Get file contents - ONLY include key files
        file_contents = git_extractor.get_file_contents()
        important_files = {
            path: content for path, content in file_contents.items() 
            if any(path.endswith(ext) for ext in ['.py', '.java', '.kt', '.js', '.ts', '.go', '.rs', '.cpp', '.h'])
            and not any(part in path.lower() for part in ['test', 'mock', 'stub', 'fixture'])
            and len(content) < 50000  # Limit individual file size
        }
        
        # Sort files by number of changes and size
        sorted_files = sorted(
            important_files.items(),
            key=lambda x: (len(changed_files.get(x[0], [])), len(x[1])),
            reverse=True
        )
        
        context_parts.append(f"## KEY SOURCE FILES ({len(important_files)} files)\n\n")
        for file_path, content in sorted_files[:10]:  # Only include top 10 files
            file_changes = changed_files.get(file_path, [])
            context_parts.append(f"### {file_path}\n")
            if file_changes:
                context_parts.append("\nRecent changes:\n")
                for change in file_changes[:3]:  # Show last 3 changes
                    context_parts.append(
                        f"- [{change['hash']}] {change['message']} "
                        f"(+{change['insertions']}, -{change['deletions']})\n"
                    )
            context_parts.append(f"\n```\n{content}\n```\n\n")
        
        context_parts.append(f"## RECENT SIGNIFICANT COMMITS ({len(significant_commits)} commits)\n\n")
        for commit in significant_commits[:10]:  # Only include top 10 significant commits
            context_parts.append(
                f"### Commit {commit['hash'][:8]}\n\n"
                f"Date: {commit['authored_date']}\n"
                f"Message: {commit['message']}\n"
                f"Files changed:\n"
            )
            
            # Group files by type/directory for better organization
            grouped_files = {}
            for file_change in commit['files_changed']:
                file_path = file_change['path']
                dir_path = str(Path(file_path).parent)
                if dir_path not in grouped_files:
                    grouped_files[dir_path] = []
                grouped_files[dir_path].append({
                    'path': file_path,
                    'insertions': file_change['insertions'],
                    'deletions': file_change['deletions']
                })
            
            # Output files by group
            for dir_path, files in sorted(grouped_files.items()):
                if dir_path == '.':
                    context_parts.append("\nRoot directory:\n")
                else:
                    context_parts.append(f"\nDirectory {dir_path}:\n")
                
                for file_info in files:
                    context_parts.append(
                        f"- {file_info['path']} "
                        f"(+{file_info['insertions']}, -{file_info['deletions']} lines)\n"
                    )
            
            context_parts.append(f"\nTotal: +{commit['insertions']}, -{commit['deletions']} lines\n\n")
        
        # Get documentation - only README and key docs
        documentation = git_extractor.get_documentation()
        key_docs = {
            path: content for path, content in documentation.items()
            if path.lower() in ['readme.md', 'contributing.md', 'architecture.md', 'design.md']
            or 'architecture' in path.lower()
            or 'design' in path.lower()
        }
        
        if key_docs:
            context_parts.append(f"## KEY DOCUMENTATION ({len(key_docs)} files)\n\n")
            for doc_path, content in key_docs.items():
                # Truncate very long docs
                if len(content) > 10000:
                    content = content[:10000] + "\n... (truncated)\n"
                context_parts.append(f"### {doc_path}\n\n{content}\n\n")
        
        # Get issues and PRs if requested - only recent important ones
        if include_issues and issue_extractor and repo_owner and repo_name:
            try:
                issues = issue_extractor.get_issues(repo_owner, repo_name, max_issues=10)
                if issues:
                    context_parts.append(f"## RECENT ISSUES ({len(issues)} issues)\n\n")
                    for issue in issues:
                        # Only include title and first paragraph of description
                        description = issue['body'].split('\n\n')[0] if issue['body'] else ''
                        context_parts.append(
                            f"### {issue['title']}\n\n"
                            f"State: {issue['state']}\n"
                            f"Created: {issue['created_at']}\n"
                            f"{description}\n\n"
                        )
            except Exception as e:
                logger.warning(f"Error fetching issues: {e}")
        
        return "\n".join(context_parts)
    
    def analyze_repository(self,
                          context: str,
                          task: str = "analyze_architecture",
                          model: str = "llama-2-70b-chat") -> Dict[str, Any]:
        """
        Analyze a repository using Llama 4.
        
        Args:
            context: Repository context to analyze
            task: Analysis task to perform
            model: Llama model to use
            
        Returns:
            Analysis results
        """
        logger.debug(f"Analyzing repository with context size: {len(context)} characters")
        
        try:
            # Send the context to Llama for analysis
            result = self.llm_client.analyze_repository(
                context,
                model=model,
                task=task,
                temperature=0.2,  # Lower temperature for more focused responses
                max_tokens=10000   # Limit response length
            )
            
            # Save result if output_dir is specified
            if self.output_dir:
                result_file = self.output_dir / f"{task}_analysis.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing repository: {e}")
            return {}
    
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
        content = None
        
        logger.debug(f"Analysis type: {type(analysis)}")
        logger.debug(f"Analysis keys: {list(analysis.keys()) if isinstance(analysis, dict) else 'Not a dict'}")
        
        if isinstance(analysis, dict):
            if 'completion_message' in analysis:
                if isinstance(analysis['completion_message'], dict):
                    content = analysis['completion_message'].get('content', '')
                else:
                    content = str(analysis['completion_message'])
            elif 'choices' in analysis and analysis['choices']:
                choice = analysis['choices'][0]
                if isinstance(choice, dict):
                    if 'message' in choice and isinstance(choice['message'], dict):
                        content = choice['message'].get('content', '')
                    elif 'text' in choice:
                        content = choice['text']
            elif 'content' in analysis:
                content = analysis['content']
            elif 'type' in analysis and 'text' in analysis:
                # Handle the case where analysis is a dict with type and text keys
                content = analysis['text']
            else:
                logger.error(f"Unexpected analysis format: {json.dumps(analysis, indent=2)}")
                content = f"Error: Could not extract content from analysis result: {json.dumps(analysis, indent=2)}"
        else:
            content = str(analysis)
        
        # Final check if content is still a dict
        if content is None:
            content = "No analysis content found."
        elif isinstance(content, dict):
            logger.warning(f"Content is still a dictionary: {json.dumps(content, indent=2)}")
            if 'text' in content:
                content = content['text']
            elif 'content' in content:
                content = content['content']
            else:
                content = json.dumps(content, indent=2)
        
        # Now content should be a string, we can safely check if it's empty
        if not content or (isinstance(content, str) and content.strip() == ""):
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

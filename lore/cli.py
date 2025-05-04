"""
Command-line interface for LORE.

This module provides a command-line interface for the LORE tool.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from lore.ingestion.git_extractor import GitExtractor
from lore.ingestion.issue_extractor import IssueExtractor
from lore.llm.llama_client import LlamaClient
from lore.analysis.analyzer import RepositoryAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

@click.group()
def cli():
    """LORE: Long-context Organizational Repository Explorer."""
    pass

@cli.command()
@click.option('--repo-path', '-r', required=True, type=click.Path(exists=True, file_okay=False),
              help='Path to the Git repository')
@click.option('--output-dir', '-o', type=click.Path(file_okay=False),
              help='Directory to save analysis results')
@click.option('--task', '-t', default='analyze_architecture',
              type=click.Choice(['analyze_architecture', 'historical_analysis', 'onboarding',
                                'refactoring_guide', 'dependency_analysis']),
              help='Analysis task to perform')
@click.option('--model', '-m', default='Llama-4-Maverick-17B-128E-Instruct-FP8',
              help='Llama model to use (e.g.,Llama-4-Maverick-17B-128E-Instruct-FP8,-, llama-4-1m)')
@click.option('--include-issues', is_flag=True, help='Include issues in analysis')
@click.option('--include-prs', is_flag=True, help='Include pull requests in analysis')
@click.option('--max-history', type=int, help='Maximum number of commits to analyze')
@click.option('--repo-owner', help='Repository owner (for issue extraction)')
@click.option('--repo-name', help='Repository name (for issue extraction)')
@click.option('--platform', default='github', type=click.Choice(['github', 'gitlab']),
              help='Platform for issue extraction')
@click.option('--report-format', default='markdown', type=click.Choice(['markdown', 'html']),
              help='Format for the analysis report')
@click.option('--api-key', help='Llama API key (overrides environment variable)')
@click.option('--api-base', help='Llama API base URL (overrides environment variable)')
def analyze(repo_path: str, output_dir: Optional[str], task: str, model: str,
          include_issues: bool, include_prs: bool, max_history: Optional[int],
          repo_owner: Optional[str], repo_name: Optional[str], platform: str,
          report_format: str, api_key: Optional[str], api_base: Optional[str]):
    """Analyze a Git repository using Llama 4."""
    try:
        logger.info(f"Analyzing repository: {repo_path}")
        
        # Initialize Git extractor
        git_extractor = GitExtractor(repo_path, max_history=max_history)
        logger.info(f"Git repository initialized: {git_extractor.repo_path}")
        
        # Initialize issue extractor if needed
        issue_extractor = None
        if include_issues or include_prs:
            if not repo_owner or not repo_name:
                logger.warning("Repository owner and name required for issue/PR extraction")
                if click.confirm("Continue without issue/PR extraction?", default=True):
                    include_issues = False
                    include_prs = False
                else:
                    sys.exit(1)
            else:
                issue_extractor = IssueExtractor(platform=platform)
                logger.info(f"Issue extractor initialized for {platform}")
        
        # Initialize Llama client
        try:
            llm_client = LlamaClient(api_key=api_key, api_base=api_base)
            logger.info("Llama API client initialized")
        except ValueError as e:
            logger.error(f"Failed to initialize Llama client: {e}")
            logger.error("Make sure LLAMA_API_KEY is set in your environment or .env file")
            sys.exit(1)
        
        # Initialize repository analyzer
        analyzer = RepositoryAnalyzer(llm_client, output_dir=output_dir)
        logger.info("Repository analyzer initialized")
        
        # Prepare repository context
        logger.info("Preparing repository context...")
        context = analyzer.prepare_repository_context(
            git_extractor,
            issue_extractor=issue_extractor,
            repo_owner=repo_owner,
            repo_name=repo_name,
            include_issues=include_issues,
            include_prs=include_prs
        )
        logger.info(f"Repository context prepared ({len(context)} characters)")
        
        # Analyze repository
        logger.info(f"Analyzing repository with task: {task}")
        analysis = analyzer.analyze_repository(context, task=task, model=model)
        logger.info("Repository analysis completed")
        
        # Generate report
        logger.info(f"Generating {report_format} report...")
        report = analyzer.generate_report(analysis, report_format=report_format)
        
        # Output report path
        if output_dir:
            extension = "md" if report_format == "markdown" else "html"
            report_path = Path(output_dir) / f"repository_analysis.{extension}"
            logger.info(f"Analysis report saved to: {report_path}")
        else:
            # Print report to console
            click.echo("\n" + "=" * 80)
            click.echo("REPOSITORY ANALYSIS REPORT")
            click.echo("=" * 80 + "\n")
            click.echo(report)
        
        logger.info("Analysis complete")
        
    except Exception as e:
        logger.exception(f"Error during repository analysis: {e}")
        sys.exit(1)

@cli.command()
@click.option('--repo-path', '-r', required=True, type=click.Path(exists=True, file_okay=False),
              help='Path to the Git repository')
@click.option('--output-file', '-o', required=True, type=click.Path(dir_okay=False),
              help='Output file to save extracted data (JSON format)')
@click.option('--include-issues', is_flag=True, help='Include issues in extraction')
@click.option('--include-prs', is_flag=True, help='Include pull requests in extraction')
@click.option('--max-history', type=int, help='Maximum number of commits to extract')
@click.option('--repo-owner', help='Repository owner (for issue extraction)')
@click.option('--repo-name', help='Repository name (for issue extraction)')
@click.option('--platform', default='github', type=click.Choice(['github', 'gitlab']),
              help='Platform for issue extraction')
def extract(repo_path: str, output_file: str, include_issues: bool, include_prs: bool,
          max_history: Optional[int], repo_owner: Optional[str], repo_name: Optional[str],
          platform: str):
    """Extract repository data without analysis."""
    import json
    
    try:
        logger.info(f"Extracting data from repository: {repo_path}")
        
        # Initialize Git extractor
        git_extractor = GitExtractor(repo_path, max_history=max_history)
        logger.info(f"Git repository initialized: {git_extractor.repo_path}")
        
        # Extract repository data
        data = {
            "files": git_extractor.get_file_contents(),
            "commits": git_extractor.get_commit_history(),
            "branches": git_extractor.get_branches(),
            "documentation": git_extractor.get_documentation()
        }
        
        # Extract issues and PRs if requested
        if include_issues or include_prs:
            if not repo_owner or not repo_name:
                logger.warning("Repository owner and name required for issue/PR extraction")
                if click.confirm("Continue without issue/PR extraction?", default=True):
                    include_issues = False
                    include_prs = False
                else:
                    sys.exit(1)
            else:
                issue_extractor = IssueExtractor(platform=platform)
                logger.info(f"Issue extractor initialized for {platform}")
                
                if include_issues:
                    logger.info("Extracting issues...")
                    data["issues"] = issue_extractor.get_issues(repo_owner, repo_name)
                
                if include_prs:
                    logger.info("Extracting pull requests...")
                    data["pull_requests"] = issue_extractor.get_pull_requests(repo_owner, repo_name)
        
        # Save data to output file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Repository data saved to: {output_file}")
        
    except Exception as e:
        logger.exception(f"Error during data extraction: {e}")
        sys.exit(1)

if __name__ == '__main__':
    cli()

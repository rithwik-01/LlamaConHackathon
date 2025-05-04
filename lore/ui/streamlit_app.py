"""
Streamlit web interface for LORE.

This module provides a web-based interface for the LORE tool using Streamlit,
with support for GitHub repository URLs and interactive chat.
"""
import os
import sys
import logging
import shutil
import atexit
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

import streamlit as st
import pandas as pd

from lore.llm.llama_client import LlamaClient
from lore.analysis.analyzer import RepositoryAnalyzer
from lore.ingestion.git_extractor import GitExtractor
from lore.ingestion.issue_extractor import IssueExtractor
from lore.ui.chat_interface import RepoChat, ChatHistory
from lore.utils.repo_utils import clone_github_repo, extract_repo_info
from lore.utils.text_processing import extract_file_extension_stats, categorize_files, truncate_text

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Define utility functions
def is_github_url(url: str) -> bool:
    """Check if a URL is a valid GitHub repository URL."""
    return bool(re.match(r'https?://github\.com/[\w-]+/[\w.-]+/?.*', url))

def init_session_state():
    """Initialize session state variables."""
    # Initialize all session state variables at startup
    default_state = {
        # Core state
        'analyzed': False,
        'chat_initialized': False,
        'chat_history': [],
        'repo_stats': {},
        'analysis_result': None,
        'report': None,
        'temp_dir': None,
        
        # API configuration
        'api_key': os.environ.get("LLAMA_API_KEY", ""),
        'api_base': os.environ.get("LLAMA_API_BASE", "https://api.llama.com/v1"),
        'model': "Llama-4-Maverick-17B-128E-Instruct-FP8",
        
        # Repository configuration
        'git_extractor': None,
        'task': "analyze_architecture",
        'repo_source': "local",
        'github_url': "",
        'repo_path': "",
        'max_history': 500,
        
        # Issue/PR configuration
        'include_issues': False,
        'include_prs': False,
        'platform': "github",
        'repo_owner': "",
        'repo_name': "",
        
        # Repository context
        'repo_context': None,
        'repo_chat': None
    }
    
    # Initialize any missing state variables
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def cleanup():
    """Clean up temporary files."""
    try:
        if hasattr(st.session_state, 'temp_dir') and st.session_state.temp_dir:
            temp_dir = Path(st.session_state.temp_dir)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup)

def initialize_llm_client() -> Optional[LlamaClient]:
    """Initialize the Llama client."""
    try:
        api_key = st.session_state.api_key
        api_base = st.session_state.api_base
        return LlamaClient(api_key=api_key, api_base=api_base)
    except Exception as e:
        st.error(f"Failed to initialize Llama client: {e}")
        return None

def extract_repo_stats(git_extractor: GitExtractor) -> Dict[str, Any]:
    """Extract repository statistics."""
    try:
        # Get file contents
        files = git_extractor.get_file_contents()
        
        # Extract file statistics
        extension_stats = extract_file_extension_stats(files)
        file_categories = categorize_files(files)
        
        # Get commit history
        commits = git_extractor.get_commit_history()
        
        # Extract commit statistics
        commit_authors = {}
        commit_dates = []
        
        for commit in commits:
            author = commit['author'].split('<')[0].strip()
            commit_authors[author] = commit_authors.get(author, 0) + 1
            commit_dates.append(commit['authored_date'])
        
        # Get documentation
        documentation = git_extractor.get_documentation()
        
        return {
            "file_count": len(files),
            "extension_stats": extension_stats,
            "file_categories": file_categories,
            "commit_count": len(commits),
            "commit_authors": commit_authors,
            "commit_dates": commit_dates,
            "documentation_count": len(documentation)
        }
    except Exception as e:
        logger.error(f"Error extracting repository statistics: {e}")
        st.error(f"Error extracting repository statistics: {e}")
        return {}

def get_repository_path() -> Tuple[str, Optional[str], Optional[str]]:
    """
    Get the repository path based on the selected source.
    
    Returns:
        Tuple of (repo_path, repo_owner, repo_name)
    """
    repo_path = None
    repo_owner = None
    repo_name = None
    
    if st.session_state.repo_source == "local":
        repo_path = st.session_state.repo_path
    else:  # GitHub URL
        github_url = st.session_state.github_url
        
        if not is_github_url(github_url):
            st.error(f"Invalid GitHub URL: {github_url}")
            return None, None, None
        
        try:
            # Extract owner and name from URL
            repo_owner, repo_name = extract_repo_info(github_url)
            
            # Clone the repository if not already done
            if st.session_state.temp_dir is None:
                with st.spinner(f"Cloning repository from {github_url}..."):
                    temp_dir = clone_github_repo(github_url)
                    st.session_state.temp_dir = temp_dir
            
            repo_path = st.session_state.temp_dir
            
        except Exception as e:
            st.error(f"Error processing GitHub URL: {e}")
            return None, None, None
    
    return repo_path, repo_owner, repo_name

def analyze_repository():
    """Analyze the repository based on user input."""
    try:
        # Get repository path based on source
        repo_path, repo_owner, repo_name = get_repository_path()
        
        if not repo_path or not Path(repo_path).exists():
            st.error(f"Repository path does not exist: {repo_path}")
            return
        
        # Initialize Git extractor
        with st.spinner("Initializing Git repository..."):
            max_history = st.session_state.max_history
            git_extractor = GitExtractor(repo_path, max_history=max_history)
            st.session_state.git_extractor = git_extractor
        
        # Extract statistics
        with st.spinner("Extracting repository statistics..."):
            stats = extract_repo_stats(git_extractor)
            st.session_state.repo_stats = stats
        
        # Initialize LLM client
        llm_client = initialize_llm_client()
        if not llm_client:
            st.error("Failed to initialize LLM client. Please check your API key.")
            return
        
        # Initialize repository analyzer
        analyzer = RepositoryAnalyzer(llm_client)
        
        # Prepare repository context
        with st.spinner("Preparing repository context..."):
            # Issue extractor
            issue_extractor = None
            if st.session_state.include_issues or st.session_state.include_prs:
                try:
                    issue_extractor = IssueExtractor(
                        st.session_state.platform,
                        repo_owner,
                        repo_name
                    )
                except Exception as e:
                    st.warning(f"Could not initialize issue extractor: {e}")
            
            # Prepare context
            context = analyzer.prepare_repository_context(
                git_extractor,
                issue_extractor,
                repo_owner,
                repo_name,
                st.session_state.include_issues,
                st.session_state.include_prs
            )
            st.session_state.repo_context = context
        
        # Analyze repository
        with st.spinner("Analyzing repository using Llama 4..."):
            task = st.session_state.task
            model = st.session_state.model
            
            # Create a temporary file to store the analysis
            analysis = analyzer.analyze_repository(context, task=task, model=model)
            st.session_state.analysis_result = analysis
            
            # Generate report
            report = analyzer.generate_report(analysis, report_format="markdown")
            st.session_state.report = report
            
            # Initialize chat functionality
            if not st.session_state.chat_initialized:
                st.session_state.repo_chat = RepoChat(llm_client, analyzer, context)
                st.session_state.chat_initialized = True
        
        st.session_state.analyzed = True
        st.success("Repository analysis completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during repository analysis: {e}")
        st.error(f"Error during repository analysis: {e}")

def handle_chat():
    """Handle chat interactions with the repository."""
    if not st.session_state.chat_initialized or not st.session_state.repo_chat:
        st.warning("Please analyze a repository first to enable chat functionality.")
        return
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask something about the repository..."):
        # Add user message to chat display
        st.chat_message("user").write(prompt)
        
        # Add to history for display
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get response from chat interface
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.repo_chat.chat(prompt)
                st.write(response)
                
                # Add to history for display
                st.session_state.chat_history.append({"role": "assistant", "content": response})

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LORE: Repository Explorer",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("LORE: Long-context Organizational Repository Explorer")
    st.markdown(
        "Explore and analyze Git repositories using Llama 4's massive context window."
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API configuration
        st.subheader("Llama API")
        api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
        api_base = st.text_input("API Base URL", value=st.session_state.api_base)
        model = st.selectbox(
            "Model", 
            ["Llama-4-Maverick-17B-128E-Instruct-FP8", "llama-2-70b-chat"],
            index=0
        )
        
        # Update session state
        st.session_state.api_key = api_key
        st.session_state.api_base = api_base
        st.session_state.model = model
        
        # Repository configuration
        st.subheader("Repository")
        st.session_state.repo_source = st.radio(
            "Repository Source",
            options=["local", "github"],
            format_func=lambda x: "Local Repository" if x == "local" else "GitHub URL"
        )
        
        if st.session_state.repo_source == "local":
            st.session_state.repo_path = st.text_input(
                "Repository Path",
                value=os.environ.get("REPO_PATH", "")
            )
        else:  # GitHub URL
            st.session_state.github_url = st.text_input(
                "GitHub Repository URL",
                placeholder="https://github.com/username/repo"
            )
        
        st.session_state.max_history = st.number_input(
            "Max Commit History",
            min_value=10,
            max_value=10000,
            value=500
        )
        
        # Analysis configuration
        st.subheader("Analysis")
        st.session_state.task = st.selectbox(
            "Analysis Task",
            options=[
                "analyze_architecture",
                "analyze_complexity",
                "historical_analysis",
                "chat",
                "onboarding",
                "refactoring_guide",
                "dependency_analysis"
            ],
            index=0
        )
        
        # Issue/PR configuration
        st.subheader("Issues/PRs (Optional)")
        st.session_state.include_issues = st.checkbox("Include Issues", value=False)
        st.session_state.include_prs = st.checkbox("Include Pull Requests", value=False)
        
        if st.session_state.include_issues or st.session_state.include_prs:
            st.session_state.platform = st.selectbox(
                "Platform",
                options=["github", "gitlab"],
                index=0
            )
            
            # Only show these fields if using local repository
            if st.session_state.repo_source == "local":
                st.session_state.repo_owner = st.text_input("Repository Owner")
                st.session_state.repo_name = st.text_input("Repository Name")
        
        # Analyze button
        if st.button("Analyze Repository"):
            analyze_repository()
    
    # Main content with tabs
    if not st.session_state.analyzed:
        # Display instructions
        st.info(
            "Enter the repository configuration in the sidebar and click 'Analyze Repository' "
            "to start the analysis."
        )
        
        # Quick start guide
        with st.expander("Quick Start Guide"):
            st.markdown("""
            ### Getting Started with LORE
            
            1. **API Configuration**:
               - Enter your Llama API key in the sidebar
               - Verify the API base URL if using a custom endpoint
               - Select the desired Llama model (10M tokens recommended for large repos)
            
            2. **Repository Configuration**:
               - Choose between a local repository or GitHub URL
               - For local: Enter the full path to your Git repository
               - For GitHub: Enter the repository URL (e.g., https://github.com/username/repo)
               - Set the maximum number of commits to analyze
            
            3. **Analysis Options**:
               - Choose an analysis task based on your needs
               - Optionally include issues and pull requests
            
            4. **Start Analysis**:
               - Click "Analyze Repository" to begin
               - The analysis may take several minutes depending on repository size
            
            5. **Chat with Your Repository**:
               - After analysis completes, use the Chat tab to ask questions about the repository
            """)
    else:
        # Display analysis results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Analysis Report", "Repository Statistics", "Chat", "Raw Data"])
        
        with tab1:
            # Display the analysis report
            if not st.session_state.report:
                st.warning("No analysis report available. Please run the analysis first.")
            else:
                st.markdown(st.session_state.report)
        
        with tab2:
            # Display repository statistics
            repo_stats = st.session_state.repo_stats
            if repo_stats:
                st.subheader("Repository Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files", repo_stats["file_count"])
                with col2:
                    st.metric("Commits", repo_stats["commit_count"])
                with col3:
                    st.metric("Documentation Files", repo_stats["documentation_count"])
                
                # File extensions chart
                st.subheader("File Extensions")
                extensions_df = pd.DataFrame.from_dict(
                    repo_stats["extension_stats"], 
                    orient='index',
                    columns=['Count']
                ).reset_index().rename(columns={'index': 'Extension'})
                extensions_df = extensions_df.sort_values('Count', ascending=False).head(10)
                st.bar_chart(extensions_df.set_index('Extension'))
                
                # File categories
                st.subheader("File Categories")
                categories = repo_stats["file_categories"]
                categories_df = pd.DataFrame({
                    'Category': list(categories.keys()),
                    'Count': [len(files) for files in categories.values()]
                })
                st.bar_chart(categories_df.set_index('Category'))
                
                # Commit authors
                st.subheader("Top Contributors")
                authors_df = pd.DataFrame.from_dict(
                    repo_stats["commit_authors"],
                    orient='index',
                    columns=['Commits']
                ).reset_index().rename(columns={'index': 'Author'})
                authors_df = authors_df.sort_values('Commits', ascending=False).head(10)
                st.bar_chart(authors_df.set_index('Author'))
                
                # Commit timeline
                if repo_stats.get("commit_dates"):
                    st.subheader("Commit Timeline")
                    dates_df = pd.DataFrame(
                        {'Date': repo_stats["commit_dates"]}
                    )
                    dates_df['Count'] = 1
                    dates_df = dates_df.set_index('Date').resample('ME').sum()
                    st.line_chart(dates_df)
        
        with tab3:
            # Chat interface
            st.subheader("Chat with Your Repository")
            st.markdown(
                "Ask questions about the repository and get answers based on the analysis."
            )
            handle_chat()
        
        with tab4:
            # Display raw data
            st.subheader("Analysis Result")
            if st.session_state.analysis_result:
                try:
                    # Extract content and metadata
                    raw_data = {
                        "model": st.session_state.analysis_result.get("model", st.session_state.model),
                        "content_length": 0,
                        "content_preview": "",
                        "metrics": {
                            "tokens": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0
                        },
                        "response_format": {
                            "type": "markdown",
                            "version": "1.0"
                        }
                    }
                    
                    # Extract content from various response formats
                    content = None
                    if "completion_message" in st.session_state.analysis_result:
                        content = st.session_state.analysis_result["completion_message"].get("content", "")
                    elif "choices" in st.session_state.analysis_result and st.session_state.analysis_result["choices"]:
                        choice = st.session_state.analysis_result["choices"][0]
                        if isinstance(choice, dict):
                            if "message" in choice:
                                content = choice["message"].get("content", "")
                            elif "text" in choice:
                                content = choice["text"]
                    
                    if content:
                        raw_data["content_length"] = len(content)
                        raw_data["content_preview"] = truncate_text(content, 500)
                    
                    # Extract metrics
                    if "metrics" in st.session_state.analysis_result:
                        raw_data["metrics"] = st.session_state.analysis_result["metrics"]
                    elif "usage" in st.session_state.analysis_result:
                        raw_data["metrics"] = {
                            "tokens": st.session_state.analysis_result["usage"].get("total_tokens", 0),
                            "prompt_tokens": st.session_state.analysis_result["usage"].get("prompt_tokens", 0),
                            "completion_tokens": st.session_state.analysis_result["usage"].get("completion_tokens", 0)
                        }
                    
                    # Display formatted JSON
                    st.json(raw_data)
                except Exception as e:
                    logger.error(f"Error formatting raw data: {e}")
                    st.error("Error formatting raw data. Showing original response:")
                    st.json(st.session_state.analysis_result)
            else:
                st.warning("No analysis result available")
    
    # Clean up resources when the user navigates away
    import atexit
    atexit.register(cleanup)

if __name__ == "__main__":
    main()

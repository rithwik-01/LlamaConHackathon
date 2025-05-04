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
from lore.utils.web_fetcher import fetch_url_content, is_valid_url
from lore.utils.image_utils import format_image_for_llama

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
        'repo_stats': {},
        'analysis_result': None,
        'report': None,
        'temp_dir': None,
        
        # API configuration - use environment variables only, removed from UI
        'api_key': os.environ.get("LLAMA_API_KEY", ""),
        'api_base': os.environ.get("LLAMA_API_BASE", "https://api.llama.com/v1"),
        'model': "Llama-4-Maverick-17B-128E-Instruct-FP8",
        
        # Repository configuration
        'git_extractor': None,
        'task': "analyze_architecture",
        'repo_source': "local",
        'github_url': "",
        'repo_path': "",
        'max_history': 1000,  # Default value, no longer configurable in UI
        
        # Issue/PR configuration
        'include_issues': False,
        'include_prs': False,
        'platform': "github",
        'repo_owner': "",
        'repo_name': "",
        
        # Design diagrams - now supports multiple images
        'design_diagrams': [],
        'use_design_diagrams': False,
        'design_diagram_descriptions': [],
        
        # Additional resources
        'use_additional_resources': False,
        'documentation_url': "",
        'meeting_notes_url': "",
        'product_requirements_url': "",
        'additional_context_url': "",
        
        # Repository context
        'repo_context': None,
        'repo_chat': None,
        'messages': [],
        
        # Documentation and additional URLs
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
        # Get API key from environment variable, fallback to session state
        api_key = os.environ.get("LLAMA_API_KEY", "")
        if not api_key and st.session_state.api_key:
            api_key = st.session_state.api_key
        
        # Get API base from environment variable, fallback to session state
        api_base = os.environ.get("LLAMA_API_BASE", "https://api.llama.com/v1")
        if not api_base and st.session_state.api_base:
            api_base = st.session_state.api_base
            
        # Validate API key
        if not api_key:
            st.error("Llama API key not found. Please check your environment variables or provide a key.")
            return None
        
        return LlamaClient(api_key=api_key, api_base=api_base)
    except Exception as e:
        st.error(f"Failed to initialize Llama client: {e}")
        logger.exception("Error initializing Llama client")
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
    
    try:
        # Local repository
        if st.session_state.repo_source == "local":
            repo_path = st.session_state.repo_path
            
            # Validate repo path exists and is a Git repository
            if repo_path:
                repo_dir = Path(repo_path)
                if not repo_dir.exists():
                    st.error(f"Repository path does not exist: {repo_path}")
                    return None, None, None
                
                git_dir = repo_dir / ".git"
                if not git_dir.exists() or not git_dir.is_dir():
                    st.error(f"Path is not a valid Git repository: {repo_path}")
                    return None, None, None
                
                # Try to extract owner and name from remote URL
                try:
                    from git import Repo
                    repo = Repo(repo_path)
                    for remote in repo.remotes:
                        if remote.name == "origin":
                            for url in remote.urls:
                                if "github.com" in url:
                                    match = re.search(r'github\.com[:/]([^/]+)/([^/\.]+)', url)
                                    if match:
                                        repo_owner = match.group(1)
                                        repo_name = match.group(2)
                                        if repo_name.endswith('.git'):
                                            repo_name = repo_name[:-4]
                                        break
                except Exception as e:
                    logger.warning(f"Could not extract owner/name from git remote: {e}")

        # GitHub repository
        elif st.session_state.repo_source == "github":
            if not st.session_state.github_url:
                st.error("Please enter a GitHub repository URL.")
                return None, None, None
                
            if not is_github_url(st.session_state.github_url):
                st.error("Invalid GitHub repository URL.")
                return None, None, None
                
            # Extract owner and repo name
            try:
                repo_owner, repo_name = extract_repo_info(st.session_state.github_url)
            except Exception as e:
                st.error(f"Failed to extract repository information: {e}")
                return None, None, None
                
            # Clone repository if not already cloned
            try:
                # Create temporary directory if doesn't exist
                if not hasattr(st.session_state, 'temp_dir') or not st.session_state.temp_dir:
                    temp_dir = tempfile.mkdtemp(prefix="lore_repo_")
                    st.session_state.temp_dir = temp_dir
                    
                # Clone repository to temporary directory
                repo_dir = Path(st.session_state.temp_dir) / f"{repo_owner}_{repo_name}"
                if not repo_dir.exists():
                    with st.spinner(f"Cloning repository {repo_owner}/{repo_name}..."):
                        repo_path = clone_github_repo(st.session_state.github_url, str(repo_dir))
                else:
                    repo_path = str(repo_dir)
                    
                # Verify it's a valid Git repo
                git_dir = Path(repo_path) / ".git"
                if not git_dir.exists() or not git_dir.is_dir():
                    st.error(f"Cloned path is not a valid Git repository: {repo_path}")
                    return None, None, None
                    
            except Exception as e:
                st.error(f"Failed to clone repository: {e}")
                return None, None, None
        
        # Set session state
        st.session_state.repo_path = repo_path
        st.session_state.repo_owner = repo_owner
        st.session_state.repo_name = repo_name
        
        return repo_path, repo_owner, repo_name
        
    except Exception as e:
        st.error(f"Error getting repository path: {e}")
        logger.exception("Error in get_repository_path")
        return None, None, None

def analyze_repository():
    """Analyze the repository based on user input."""
    try:
        # Reset analysis state
        st.session_state.analyzed = False
        st.session_state.chat_initialized = False
        
        # Initialize LLM client
        llm_client = initialize_llm_client()
        if not llm_client:
            st.error("Failed to initialize Llama client. Please check your API key.")
            return
        
        # Get repository information
        repo_path, repo_owner, repo_name = get_repository_path()
        if not repo_path:
            # Error already displayed in get_repository_path
            return
            
        # Verify the repository path is valid
        repo_dir = Path(repo_path)
        if not repo_dir.exists():
            st.error(f"Repository path does not exist: {repo_path}")
            return
            
        git_dir = repo_dir / ".git"
        if not git_dir.exists() or not git_dir.is_dir():
            st.error(f"Path is not a valid Git repository: {repo_path}")
            return
        
        # Get git extractor
        with st.spinner("Initializing Git repository..."):
            try:
                git_extractor = GitExtractor(repo_path, max_history=1000)
                st.session_state.git_extractor = git_extractor
            except Exception as e:
                st.error(f"Failed to create git extractor: {e}")
                return
        
        # Get repository statistics
        try:
            repo_stats = extract_repo_stats(git_extractor)
            st.session_state.repo_stats = repo_stats
        except Exception as e:
            st.error(f"Failed to extract repository statistics: {e}")
            return
        
        # Get repository context
        repository_context = ""
        try:
            # Initialize repository analyzer with the correct parameters
            repo_analyzer = RepositoryAnalyzer(
                llm_client=llm_client
            )
            
            # Fetch URL content if available
            url_content = {}
            
            if st.session_state.use_additional_resources:
                # Process documentation URL
                if st.session_state.documentation_url:
                    try:
                        doc_content = fetch_url_content(st.session_state.documentation_url)
                        if doc_content:
                            url_content["documentation_url"] = {
                                "url": st.session_state.documentation_url,
                                "content": doc_content
                            }
                    except Exception as e:
                        logger.error(f"Error fetching documentation URL: {e}")
                
                # Process meeting notes URL
                if st.session_state.meeting_notes_url:
                    try:
                        meeting_content = fetch_url_content(st.session_state.meeting_notes_url)
                        if meeting_content:
                            url_content["meeting_notes_url"] = {
                                "url": st.session_state.meeting_notes_url,
                                "content": meeting_content
                            }
                    except Exception as e:
                        logger.error(f"Error fetching meeting notes URL: {e}")
                
                # Process product requirements URL
                if st.session_state.product_requirements_url:
                    try:
                        requirements_content = fetch_url_content(st.session_state.product_requirements_url)
                        if requirements_content:
                            url_content["product_requirements_url"] = {
                                "url": st.session_state.product_requirements_url,
                                "content": requirements_content
                            }
                    except Exception as e:
                        logger.error(f"Error fetching product requirements URL: {e}")
                
                # Process additional context URL
                if st.session_state.additional_context_url:
                    try:
                        context_content = fetch_url_content(st.session_state.additional_context_url)
                        if context_content:
                            url_content["additional_context_url"] = {
                                "url": st.session_state.additional_context_url,
                                "content": context_content
                            }
                    except Exception as e:
                        logger.error(f"Error fetching additional context URL: {e}")
            
            # Generate context first
            repository_context = repo_analyzer.prepare_repository_context(
                git_extractor=git_extractor,
                include_issues=st.session_state.include_issues,
                include_prs=st.session_state.include_prs,
                repo_owner=repo_owner,
                repo_name=repo_name,
                additional_resources={
                    "additional_context": "",
                    "documentation": "",
                    "product_requirements": "",
                    "meeting_notes": "",
                    "url_content": url_content
                }
            )
            
            # Now analyze the repository with the context
            task = st.session_state.task
            with st.spinner(f"Analyzing repository: {task}..."):
                analysis_result = repo_analyzer.analyze_repository(
                    context=repository_context,
                    task=task,
                    model=st.session_state.model
                )
                
                # Generate the report
                report = repo_analyzer.generate_report(analysis_result, report_format="markdown")
                
                # Store the results
                st.session_state.analysis_result = analysis_result
                st.session_state.report = report
                st.session_state.repo_context = repository_context
            
            # Initialize chat with context
            if repository_context:
                chat_history = ChatHistory()
                repo_chat = RepoChat(
                    llm_client=llm_client, 
                    repository_context=repository_context,
                    history=chat_history
                )
                
                # Add design diagrams if available
                if st.session_state.use_design_diagrams and st.session_state.design_diagrams:
                    for diagram_path in st.session_state.design_diagrams:
                        repo_chat.add_design_diagram(diagram_path)
                
                st.session_state.repo_chat = repo_chat
                st.session_state.chat_initialized = True
            
            # Mark as analyzed
            st.session_state.analyzed = True
            
            # Display success message
            st.success(f"Repository analyzed successfully: {task}")
            
        except Exception as e:
            st.error(f"Failed to analyze repository: {e}")
            logger.exception("Error during repository analysis")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.exception("Error in analyze_repository")

def main():
    """Main Streamlit application."""
    # Configure the page
    st.set_page_config(
        page_title="LORE - Code Repository Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    init_session_state()
    
    # Page header
    st.title("🔍 LORE - Code Repository Analyzer")
    st.markdown("""
    Analyze your code repository using Llama AI and get insights about its architecture, 
    dependencies, and code quality. Chat with your repository to understand it better.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Repository source selection
        st.header("Repository Source")
        repo_source = st.radio(
            "Select Repository Source",
            ["Local Repository", "GitHub URL"],
            index=0 if st.session_state.repo_source == "local" else 1,
            key="repo_source_radio"
        )
        st.session_state.repo_source = "local" if repo_source == "Local Repository" else "github"
        
        # Repository configuration
        if st.session_state.repo_source == "local":
            st.text_input(
                "Repository Path",
                value=st.session_state.repo_path,
                help="Enter the path to your local repository",
                key="repo_path_input"
            )
            st.session_state.repo_path = st.session_state.repo_path_input
        else:
            st.text_input(
                "GitHub Repository URL",
                value=st.session_state.github_url,
                help="Enter the URL of a GitHub repository (e.g., https://github.com/username/repo)",
                key="github_url_input"
            )
            st.session_state.github_url = st.session_state.github_url_input
            
            # GitHub issues and PRs (shown only when GitHub URL is selected)
            with st.expander("Issues & Pull Requests"):
                st.checkbox(
                    "Include Issues",
                    value=st.session_state.include_issues,
                    help="Include open issues in the analysis",
                    key="include_issues_input"
                )
                st.session_state.include_issues = st.session_state.include_issues_input
                
                st.checkbox(
                    "Include Pull Requests",
                    value=st.session_state.include_prs,
                    help="Include open pull requests in the analysis",
                    key="include_prs_input"
                )
                st.session_state.include_prs = st.session_state.include_prs_input
        
        # Task selection
        st.header("Analysis Task")
        task = st.selectbox(
            "Select Task",
            ["Architecture Analysis", "Dependency Analysis", "Code Quality Analysis", "Documentation Analysis"],
            index=0 if st.session_state.task == "analyze_architecture" else 
                 (1 if st.session_state.task == "analyze_dependencies" else 
                 (2 if st.session_state.task == "analyze_quality" else 3)),
            key="task_selectbox"
        )
        
        # Map user-friendly task names to internal task identifiers
        task_mapping = {
            "Architecture Analysis": "analyze_architecture",
            "Dependency Analysis": "analyze_dependencies",
            "Code Quality Analysis": "analyze_quality",
            "Documentation Analysis": "analyze_documentation"
        }
        st.session_state.task = task_mapping.get(task, "analyze_architecture")
        
        # Multiple design diagrams upload
        st.header("Design Diagrams")
        st.checkbox(
            "Include Design Diagrams",
            value=st.session_state.use_design_diagrams,
            help="Include design diagrams in the analysis",
            key="use_design_diagrams_input"
        )
        st.session_state.use_design_diagrams = st.session_state.use_design_diagrams_input
        
        if st.session_state.use_design_diagrams:
            uploaded_files = st.file_uploader(
                "Upload Design Diagrams",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                help="Upload design diagrams to aid in the analysis"
            )
            
            # Clear existing diagrams if the user uploads new ones
            if uploaded_files:
                # Save uploaded files to disk
                st.session_state.design_diagrams = []
                st.session_state.design_diagram_descriptions = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Create temp directory if not exists
                    if not st.session_state.temp_dir:
                        st.session_state.temp_dir = tempfile.mkdtemp()
                    
                    # Save file
                    file_path = Path(st.session_state.temp_dir) / f"design_diagram_{i}_{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Store file path
                    st.session_state.design_diagrams.append(str(file_path))
                    
                    # Optional description for this diagram
                    description = st.text_area(
                        f"Description for {uploaded_file.name}",
                        help="Provide a brief description of this diagram",
                        key=f"diagram_description_{i}"
                    )
                    st.session_state.design_diagram_descriptions.append(description)
        
        # Additional resources
        with st.expander("Additional Resources"):
            st.checkbox(
                "Use Additional Resources",
                value=st.session_state.use_additional_resources,
                help="Include additional resources in the analysis",
                key="use_additional_resources_input"
            )
            st.session_state.use_additional_resources = st.session_state.use_additional_resources_input
            
            if st.session_state.use_additional_resources:
                # Only show URL inputs, no text boxes
                st.text_input(
                    "Documentation URL",
                    value=st.session_state.documentation_url,
                    help="Enter the URL of the documentation",
                    key="documentation_url_input"
                )
                st.session_state.documentation_url = st.session_state.documentation_url_input
                
                st.text_input(
                    "Meeting Notes URL",
                    value=st.session_state.meeting_notes_url,
                    help="Enter the URL of the meeting notes",
                    key="meeting_notes_url_input"
                )
                st.session_state.meeting_notes_url = st.session_state.meeting_notes_url_input
                
                st.text_input(
                    "Product Requirements URL",
                    value=st.session_state.product_requirements_url,
                    help="Enter the URL of the product requirements",
                    key="product_requirements_url_input"
                )
                st.session_state.product_requirements_url = st.session_state.product_requirements_url_input
                
                st.text_input(
                    "Additional Context URL",
                    value=st.session_state.additional_context_url,
                    help="Enter any additional context URL",
                    key="additional_context_url_input"
                )
                st.session_state.additional_context_url = st.session_state.additional_context_url_input
        
        # Analysis button
        st.button("Analyze Repository", on_click=analyze_repository)
        
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Report", "Stats", "Chat", "Raw Data"])
    
    # Report tab content
    with tab1:
        st.subheader("Repository Analysis Report")
        if st.session_state.analyzed and st.session_state.report:
            st.markdown(st.session_state.report)
        else:
            st.info("No analysis has been performed yet. Configure the repository source and click 'Analyze Repository'.")
    
    # Stats tab content
    with tab2:
        st.subheader("Repository Statistics")
        if st.session_state.analyzed and st.session_state.repo_stats:
            repo_stats = st.session_state.repo_stats
            
            # Display repository information
            if "name" in repo_stats and "owner" in repo_stats:
                st.markdown(f"### {repo_stats['name']}")
                st.markdown(f"**Owner:** {repo_stats['owner']}")
            
            # File statistics
            if "languages" in repo_stats and repo_stats["languages"]:
                st.subheader("Language Distribution")
                languages_df = pd.DataFrame(
                    [(lang, count) for lang, count in repo_stats["languages"].items()],
                    columns=["Language", "Files"]
                ).sort_values("Files", ascending=False)
                st.bar_chart(languages_df.set_index("Language"))
            
            # Extensions statistics
            if "file_extensions" in repo_stats and repo_stats["file_extensions"]:
                st.subheader("File Extensions")
                extensions_df = pd.DataFrame(
                    [(ext, count) for ext, count in repo_stats["file_extensions"].items()],
                    columns=["Extension", "Count"]
                ).sort_values("Count", ascending=False).head(10)
                st.bar_chart(extensions_df.set_index("Extension"))
            
            # Commit statistics
            if "authors" in repo_stats and repo_stats["authors"]:
                st.subheader("Top Contributors")
                authors_df = pd.DataFrame(
                    [(author, commits) for author, commits in repo_stats["authors"].items()],
                    columns=["Author", "Commits"]
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
    
    # Chat tab content
    with tab3:
        st.subheader("Chat with Your Repository")
        st.markdown(
            "Ask questions about the repository and get answers based on the analysis."
        )
        
        # Simple chat implementation
        if not st.session_state.analyzed:
            st.warning("Please analyze a repository first.")
        else:
            # Display previous messages if any
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display existing messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Accept user input
            prompt = st.chat_input("Ask a question about the repository")
            
            # Initialize chat if needed
            if prompt and not st.session_state.chat_initialized:
                llm_client = initialize_llm_client()
                if llm_client and st.session_state.repo_context:
                    chat_history = ChatHistory(max_history=st.session_state.max_history)
                    st.session_state.repo_chat = RepoChat(
                        llm_client=llm_client,
                        repository_context=st.session_state.repo_context,
                        history=chat_history
                    )
                    
                    # Initialize diagrams if available
                    if st.session_state.use_design_diagrams and st.session_state.design_diagrams:
                        for diagram_path in st.session_state.design_diagrams:
                            st.session_state.repo_chat.add_design_diagram(diagram_path)
                    
                    st.session_state.chat_initialized = True
            
            # Process the message
            if prompt and st.session_state.chat_initialized:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking...")
                    
                    try:
                        response = st.session_state.repo_chat.chat(prompt)
                        message_placeholder.markdown(response)
                        # Add response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        message_placeholder.markdown(f"Error: {str(e)}")
                        st.error(f"Error processing chat: {e}")
                        logger.exception("Error in chat processing")
    
    # Raw Data tab
    with tab4:
        st.subheader("Raw Repository Data")
        if st.session_state.analyzed:
            # Show git extractor data
            if st.session_state.git_extractor:
                st.subheader("Git Repository Data")
                
                # File statistics
                with st.expander("File Statistics"):
                    st.json(st.session_state.repo_stats)
                
                # Commit history
                with st.expander("Commit History"):
                    # Get all commits and display only the first 10
                    commits = st.session_state.git_extractor.get_commit_history()
                    # Only display the first 10 commits in the UI for better performance
                    for commit in commits[:10]:
                        st.markdown(f"**{commit['hash'][:8]}** - {commit['message']}")
                        st.markdown(f"*Author:* {commit['author']} *Date:* {commit['authored_date']}")
                        st.markdown("---")
        else:
            st.info("No analysis has been performed yet.")
    
    # Create tab buttons to allow direct navigation
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Go to Report"):
            st.session_state.active_tab = 0
            st.rerun()
            
    with col2:
        if st.button("Go to Stats"):
            st.session_state.active_tab = 1
            st.rerun()
            
    with col3:
        if st.button("Go to Chat"):
            st.session_state.active_tab = 2
            st.rerun()
            
    with col4:
        if st.button("Go to Raw Data"):
            st.session_state.active_tab = 3
            st.rerun()
    
# Call the main function
if __name__ == "__main__":
    main()

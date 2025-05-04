# LORE: Long-context Organizational Repository Explorer

LORE (Long-context Organizational Repository Explorer) is a powerful tool that leverages Llama 4's massive context window capabilities to provide deep insights into software repositories.

## Features

- **Holistic Codebase Analysis**: Ingest entire codebases along with their Git history, documentation, and discussions
- **Deep Historical Analysis**: Understand why code evolved by cross-referencing changes with commit messages and issue discussions
- **Architectural Insights**: Detect inconsistencies, anti-patterns, and architectural drift across the entire system
- **Knowledge Transfer**: Generate comprehensive explanations of specific modules or features for new developers
- **Refactoring Planning**: Identify dependencies and potential impacts of major refactoring efforts
- **GitHub Repository Support**: Analyze repositories directly from GitHub URLs
- **Interactive Chat**: Converse with your codebase and ask questions about any aspect of the repository        

## Requirements

- Python 3.9+
- Git
- Llama 4 API access
- Nebius/Lambda compute (optional, for large repositories)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lore.git
cd lore

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env with your Llama 4 API key
```

## Usage

### Command Line Interface

```bash
# Analyze local repository
python -m lore.cli analyze --repo-path /path/to/repository

# Advanced options with local repository
python -m lore.cli analyze --repo-path /path/to/repository --include-issues --include-prs --max-history 500
```

### Web Interface

```bash
# Start the web interface
python -m lore.ui.streamlit_app
```

In the web interface, you can:
1. Enter a local repository path or GitHub URL
2. Configure analysis options
3. View detailed analysis results
4. Chat interactively with your repository

## Architecture

LORE consists of several components:
1. **Data Ingestion**: Collects code, Git history, documentation, and discussions
2. **Preprocessing**: Structures and formats the data for efficient analysis
3. **LLM Interface**: Communicates with Llama 4 API to analyze the repository data
4. **Insight Engine**: Processes and organizes LLM responses into actionable insights
5. **User Interface**: CLI and web interface for interacting with the system
6. **Chat Interface**: Enables interactive conversations with the analyzed repository
7. **Repository Manager**: Handles GitHub repository cloning and local repo management

## License

MIT

## Screenshots

### Analysis View
(Screenshots would go here in a production README)

### Chat Interface
(Screenshots would go here in a production README)

## Development

To contribute to LORE:

1. Clone the repository
2. Install development dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest tests/`
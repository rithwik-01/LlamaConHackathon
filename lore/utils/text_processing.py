"""
Text processing utilities for LORE.

This module provides utilities for processing and formatting text data.
"""
import re
from typing import Dict, List, Set, Tuple, Optional

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to the specified maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of the truncated text
        suffix: Suffix to append to truncated text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def count_tokens(text: str, model: str = "llama-4") -> int:
    """
    Estimate the number of tokens in a text string.
    
    This is a very rough estimation based on common tokenization patterns.
    For accurate counts, you should use the model's actual tokenizer.
    
    Args:
        text: Text to count tokens in
        model: Model name (used to select tokenization approach)
        
    Returns:
        Estimated token count
    """
    # Very rough estimation: ~4 chars per token for English text
    # This is just a rough approximation
    return len(text) // 4

def format_code_blocks(text: str) -> str:
    """
    Format code blocks in markdown text with syntax highlighting hints.
    
    Args:
        text: Markdown text with code blocks
        
    Returns:
        Formatted text with language hints
    """
    # Add language hints to code blocks without them
    # This regex finds ```code blocks``` without a language specified
    pattern = r'```\s*\n'
    
    # Try to determine the language based on content and add appropriate hint
    def replacement(match):
        # Default to 'text' if we can't determine the language
        return '```text\n'
    
    return re.sub(pattern, replacement, text)

def extract_file_extension_stats(files: Dict[str, str]) -> Dict[str, int]:
    """
    Extract statistics about file extensions in a repository.
    
    Args:
        files: Dictionary mapping file paths to contents
        
    Returns:
        Dictionary mapping file extensions to counts
    """
    extension_counts = {}
    
    for file_path in files:
        # Extract the file extension
        extension = file_path.split('.')[-1] if '.' in file_path else 'no_extension'
        extension_counts[extension] = extension_counts.get(extension, 0) + 1
    
    return extension_counts

def categorize_files(files: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Categorize files by type.
    
    Args:
        files: Dictionary mapping file paths to contents
        
    Returns:
        Dictionary mapping file categories to lists of file paths
    """
    categories = {
        'code': [],
        'documentation': [],
        'configuration': [],
        'data': [],
        'other': []
    }
    
    # Extensions by category
    code_extensions = {'py', 'js', 'ts', 'java', 'c', 'cpp', 'h', 'hpp', 'cs', 'go', 'rb', 'php', 'swift', 'kt', 'rs'}
    doc_extensions = {'md', 'rst', 'txt', 'docx', 'pdf', 'html', 'adoc'}
    config_extensions = {'json', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf', 'xml'}
    data_extensions = {'csv', 'tsv', 'xlsx', 'db', 'sql', 'json', 'xml'}
    
    for file_path in files:
        # Extract the file extension
        extension = file_path.split('.')[-1].lower() if '.' in file_path else ''
        
        # Categorize based on extension
        if extension in code_extensions:
            categories['code'].append(file_path)
        elif extension in doc_extensions:
            categories['documentation'].append(file_path)
        elif extension in config_extensions:
            categories['configuration'].append(file_path)
        elif extension in data_extensions:
            categories['data'].append(file_path)
        else:
            categories['other'].append(file_path)
    
    return categories

def extract_imports_from_python(file_content: str) -> Set[str]:
    """
    Extract imported libraries and modules from Python code.
    
    Args:
        file_content: Python file content
        
    Returns:
        Set of imported module names
    """
    imports = set()
    
    # Match import statements
    import_pattern = r'^import\s+([\w\.]+)(?:\s+as\s+\w+)?'
    from_pattern = r'^from\s+([\w\.]+)\s+import'
    
    for line in file_content.split('\n'):
        line = line.strip()
        
        # Check for import statements
        import_match = re.match(import_pattern, line)
        if import_match:
            imports.add(import_match.group(1).split('.')[0])
            continue
        
        # Check for from ... import statements
        from_match = re.match(from_pattern, line)
        if from_match:
            imports.add(from_match.group(1).split('.')[0])
    
    return imports

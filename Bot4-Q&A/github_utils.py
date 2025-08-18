import requests
from bs4 import BeautifulSoup
import base64
import re
from urllib.parse import urlparse

def is_github_url(url):
    """Check if a URL is a GitHub repository URL."""
    parsed = urlparse(url)
    return parsed.netloc in ['github.com', 'www.github.com']

def parse_github_url(url):
    """Extract owner and repo from GitHub URL."""
    parts = url.strip('/').split('/')
    if 'github.com' in parts:
        idx = parts.index('github.com')
        if len(parts) > idx + 2:
            owner = parts[idx + 1]
            repo = parts[idx + 2]
            return owner, repo
    return None, None

def get_repo_content(url):
    """
    Get content from a GitHub repository using GitHub's API.
    Returns a dictionary of filenames and their content.
    """
    owner, repo = parse_github_url(url)
    if not owner or not repo:
        return {"error": "Invalid GitHub URL format"}
    
    try:
        # Fetch repository contents
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        response = requests.get(api_url)
        response.raise_for_status()
        
        contents = response.json()
        repo_content = {}
        
        # Process each file/directory
        for item in contents:
            if item['type'] == 'file' and item['name'].endswith(('.py', '.js', '.html', '.css', '.md')):
                # Get file content
                file_response = requests.get(item['url'])
                file_response.raise_for_status()
                file_data = file_response.json()
                
                if 'content' in file_data:
                    content = base64.b64decode(file_data['content']).decode('utf-8')
                    repo_content[item['name']] = content
                    
            # Limit to first 5 files to avoid exceeding API limits
            if len(repo_content) >= 5:
                break
                
        return repo_content
    
    except Exception as e:
        return {"error": f"Error fetching repository: {str(e)}"}

def get_repo_structure(url):
    """
    Get the structure of a GitHub repository.
    Returns a list of file paths in the repository.
    """
    owner, repo = parse_github_url(url)
    if not owner or not repo:
        return {"error": "Invalid GitHub URL format"}
    
    try:
        # Use GitHub's API to get repository contents
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
        response = requests.get(api_url)
        
        # If 'main' branch doesn't exist, try 'master'
        if response.status_code != 200:
            api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"
            response = requests.get(api_url)
            
        response.raise_for_status()
        data = response.json()
        
        # Extract file paths
        files = [item['path'] for item in data['tree'] if item['type'] == 'blob']
        return files
    
    except Exception as e:
        return {"error": f"Error fetching repository structure: {str(e)}"}

def get_repo_metadata(url):
    """
    Get metadata about a GitHub repository such as description, stars, etc.
    """
    owner, repo = parse_github_url(url)
    if not owner or not repo:
        return {"error": "Invalid GitHub URL format"}
    
    try:
        # Use GitHub's API to get repository information
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(api_url)
        response.raise_for_status()
        
        data = response.json()
        return {
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "stars": data.get("stargazers_count", 0),
            "forks": data.get("forks_count", 0),
            "language": data.get("language", ""),
            "url": data.get("html_url", "")
        }
    
    except Exception as e:
        return {"error": f"Error fetching repository metadata: {str(e)}"}

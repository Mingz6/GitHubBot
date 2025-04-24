import requests
from bs4 import BeautifulSoup
import base64
import re
import os
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

def parse_github_pr_url(url):
    """Extract owner, repo, and PR number from GitHub PR URL."""
    pattern = r'https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
    match = re.match(pattern, url)
    if match:
        owner, repo, pr_number = match.groups()
        return owner, repo, pr_number
    return None, None, None

def get_pr_details(pr_url, max_files=25, file_types=None):
    """
    Get details of a GitHub Pull Request including changed files and their contents.
    Returns a dictionary with PR metadata and changes.
    
    Args:
        pr_url: URL of the GitHub PR
        max_files: Maximum number of files to fetch (default: 25)
        file_types: List of file extensions to include (default: None = all code files)
    """
    owner, repo, pr_number = parse_github_pr_url(pr_url)
    if not owner or not repo or not pr_number:
        return {"error": "Invalid GitHub PR URL format"}
    
    try:
        # Fetch PR information
        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(api_url)
        response.raise_for_status()
        
        pr_data = response.json()
        
        # Get PR metadata
        pr_details = {
            "title": pr_data.get("title", ""),
            "description": pr_data.get("body", ""),
            "user": pr_data.get("user", {}).get("login", ""),
            "state": pr_data.get("state", ""),
            "created_at": pr_data.get("created_at", ""),
            "updated_at": pr_data.get("updated_at", ""),
            "target_branch": pr_data.get("base", {}).get("ref", ""),
            "source_branch": pr_data.get("head", {}).get("ref", ""),
            "changed_files": [],
            "total_file_count": pr_data.get("changed_files", 0)
        }
        
        # Default file types to include if not specified
        if file_types is None:
            file_types = ['.py', '.js', '.html', '.css', '.md', '.java', '.ts', '.jsx', 
                          '.tsx', '.go', '.c', '.cpp', '.h', '.hpp', '.json', '.yml', 
                          '.yaml', '.sh', '.txt', '.sql']
        
        # Fetch PR changed files with pagination
        page = 1
        files_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files?per_page=100&page={page}"
        
        while True:
            files_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files?per_page=100&page={page}"
            files_response = requests.get(files_url)
            files_response.raise_for_status()
            
            files_data = files_response.json()
            
            # If no more files, break the loop
            if not files_data:
                break
                
            # Process each file in this page
            for file_data in files_data:
                filename = file_data.get("filename", "")
                
                # Skip binary files and non-code files
                file_ext = os.path.splitext(filename)[1].lower()
                if file_types and file_ext not in file_types:
                    continue
                    
                file_info = {
                    "filename": filename,
                    "status": file_data.get("status", ""),  # added, modified, removed
                    "additions": file_data.get("additions", 0),
                    "deletions": file_data.get("deletions", 0),
                    "patch": file_data.get("patch", "")
                }
                
                # Add file content if it exists in the PR
                if file_data.get("status") != "removed":
                    try:
                        file_content_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{pr_data['head']['sha']}/{filename}"
                        content_response = requests.get(file_content_url)
                        
                        if content_response.status_code == 200:
                            file_info["content"] = content_response.text
                    except Exception as e:
                        file_info["content_error"] = str(e)
                
                pr_details["changed_files"].append(file_info)
                
                # Stop when we reach the maximum number of files
                if len(pr_details["changed_files"]) >= max_files:
                    break
            
            # If we've reached max files or there are no more pages, break
            if len(pr_details["changed_files"]) >= max_files or len(files_data) < 100:
                break
                
            # Move to next page
            page += 1
        
        return pr_details
    
    except Exception as e:
        return {"error": f"Error fetching PR details: {str(e)}"}

def get_target_branch_code(pr_url, max_files=25, file_types=None):
    """
    Get the code from the target branch of a PR.
    Returns a dictionary of filenames and their content from the target branch.
    
    Args:
        pr_url: URL of the GitHub PR
        max_files: Maximum number of files to fetch (default: 25)
        file_types: List of file extensions to include (default: None = all code files)
    """
    owner, repo, pr_number = parse_github_pr_url(pr_url)
    if not owner or not repo or not pr_number:
        return {"error": "Invalid GitHub PR URL format"}
    
    try:
        # First get the PR to find the target branch name
        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(api_url)
        response.raise_for_status()
        
        pr_data = response.json()
        target_branch = pr_data.get("base", {}).get("ref", "main")  # Default to main if not found
        
        # Default file types to include if not specified
        if file_types is None:
            file_types = ['.py', '.js', '.html', '.css', '.md', '.java', '.ts', '.jsx', 
                          '.tsx', '.go', '.c', '.cpp', '.h', '.hpp', '.json', '.yml', 
                          '.yaml', '.sh', '.txt', '.sql']
        
        # Get files that were changed in the PR with pagination
        page = 1
        target_branch_code = {}
        
        while True:
            files_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files?per_page=100&page={page}"
            files_response = requests.get(files_url)
            files_response.raise_for_status()
            
            files_data = files_response.json()
            
            # If no more files, break the loop
            if not files_data:
                break
                
            # Get the changed filenames from this page
            for file_data in files_data:
                filename = file_data.get("filename")
                
                # Skip if filename is None or non-matching extension
                if not filename:
                    continue
                    
                file_ext = os.path.splitext(filename)[1].lower()
                if file_types and file_ext not in file_types:
                    continue
                
                try:
                    # Get file content from target branch
                    file_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{target_branch}/{filename}"
                    file_response = requests.get(file_url)
                    
                    if file_response.status_code == 200:
                        target_branch_code[filename] = file_response.text
                except Exception as e:
                    print(f"Error fetching {filename} from target branch: {str(e)}")
                    
                # Stop when we reach the maximum number of files
                if len(target_branch_code) >= max_files:
                    break
            
            # If we've reached max files or there are no more pages, break
            if len(target_branch_code) >= max_files or len(files_data) < 100:
                break
                
            # Move to next page
            page += 1
        
        return target_branch_code
    
    except Exception as e:
        return {"error": f"Error fetching target branch code: {str(e)}"}

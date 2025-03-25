from flask import Flask, render_template, request
import requests
import json
import re
import urllib.parse

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""
    result_count = 5  # Default value
    search_type = "keyword"  # Default search type
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        # Get the user-selected result count
        result_count = int(request.form.get('result_count', 5))
        
        if query:
            # Check if query might contain username/repo format
            username_repo = extract_username_repo(query)
            if username_repo:
                search_type = "username_repo"
                results = search_github_specific_repo(username_repo, result_count)
            else:
                search_type = "keyword"
                results = search_github_repos(query, result_count)
    
    return render_template('index.html', 
                           results=results, 
                           query=query, 
                           result_count=result_count, 
                           search_type=search_type)

def extract_username_repo(query):
    """
    Extract potential username/repo patterns from the query string
    Patterns like:
    - "UserName Ming Repo GitHub" -> "Ming/GitHub"
    - "user Ming repo GitHub" -> "Ming/GitHub"
    - "Ming/GitHub" (already formatted)
    """
    # Pattern 1: Already in correct format (username/repo)
    if '/' in query and len(query.split('/')) == 2:
        parts = query.split('/')
        if len(parts[0].strip()) > 0 and len(parts[1].strip()) > 0:
            return query.strip()
    
    # Pattern 2: Words indicating username/repo relationship
    username_indicators = ['username', 'user', 'owner', 'developer', 'by']
    repo_indicators = ['repo', 'repository', 'project']
    
    words = query.lower().split()
    username = None
    repo = None
    
    # Look for patterns like "user Ming repo GitHub"
    for i, word in enumerate(words):
        if word in username_indicators and i+1 < len(words):
            username = words[i+1]
        elif word in repo_indicators and i+1 < len(words):
            repo = words[i+1]
    
    if username and repo:
        return f"{username}/{repo}"
    
    return None

def search_github_specific_repo(username_repo, result_count=5):
    """Search for a specific GitHub repository by username/repo pattern"""
    try:
        # Format for API search
        formatted_query = f"repo:{username_repo}"
        url = f"https://api.github.com/search/repositories?q={formatted_query}&per_page={result_count}"
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Repository-Finder"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # If no exact match found, try a broader search with the parts
            if len(data.get('items', [])) == 0:
                parts = username_repo.split('/')
                if len(parts) == 2:
                    username, repo = parts
                    # Search by username and repository name separately
                    return search_github_repos(f"user:{username} {repo}", result_count)
            
            for repo in data.get('items', []):
                results.append({
                    'title': repo.get('full_name', ''),
                    'url': repo.get('html_url', ''),
                    'description': repo.get('description', 'No description available')
                })
                
            return results
        else:
            print(f"Error with GitHub API: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error searching specific repository: {e}")
        return []

def search_github_repos(query, result_count=5):
    """Search for GitHub repositories using GitHub API"""
    try:
        # Format query for GitHub search
        formatted_query = urllib.parse.quote(query)
        
        # Use GitHub's search API with the user-specified result count
        url = f"https://api.github.com/search/repositories?q={formatted_query}&sort=stars&order=desc&per_page={result_count}"
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Repository-Finder"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            for repo in data.get('items', []):
                results.append({
                    'title': repo.get('full_name', ''),
                    'url': repo.get('html_url', ''),
                    'description': repo.get('description', 'No description available')
                })
                
            return results
        else:
            print(f"Error with GitHub API: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error searching repositories: {e}")
        return []

if __name__ == '__main__':
    app.run(debug=True, port=5000)

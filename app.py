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
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        # Get the user-selected result count
        result_count = int(request.form.get('result_count', 5))
        if query:
            results = search_github_repos(query, result_count)
    
    return render_template('index.html', results=results, query=query, result_count=result_count)

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

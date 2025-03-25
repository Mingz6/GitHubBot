# GitHubBot

A Flask web application that allows users to search for GitHub repositories using DuckDuckGo search integration. This application provides a clean and user-friendly interface for discovering GitHub projects.

## Features

- **Simple Search Interface**: Clean UI with a search bar and customizable result count
- **Smart Repository Search**: Search GitHub repositories with keyword-based queries
- **Direct Repository Lookup**: Find specific repositories using username/repository format
- **Flexible Search Options**: Control the number of search results displayed (5, 10, 15, 20, or 30)
- **Helpful Search Tips**: Built-in guidance on effective search techniques

## Screenshots

*[Add screenshots of your application here]*

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/GitHubBot.git
   cd GitHubBot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Required Dependencies

- Flask 2.3.3
- Requests 2.31.0
- DuckDuckGo-Search 3.0.2

## Usage

### Basic Search
Enter keywords in the search bar and click "Search" to find GitHub repositories that match your query.

### Username/Repository Search
You can search for specific repositories using these formats:
- Direct format: `username/repository` (e.g., `microsoft/vscode`)
- Natural language: `user username repo repository` (e.g., `user microsoft repo vscode`)

### Customizing Results
Use the dropdown menu to select how many search results you want to see per page.

## Development

The application structure:
- `app.py`: Main Flask application logic and API integration
- `templates/index.html`: Frontend HTML template with embedded styles
- `requirements.txt`: Python dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- Built with Flask
- Uses DuckDuckGo search integration
- GitHub API for repository information
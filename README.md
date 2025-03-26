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

## Setup and Installation

For detailed setup instructions and environment configuration, please refer to [Setup Guide](/Docs/SetupEnvironment.md).

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
- `Python/app.py`: Main Flask application logic and API integration
- `templates/index.html`: Frontend HTML template with embedded styles

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- Built with Flask
- Uses DuckDuckGo search integration
- GitHub API for repository information
# GitHub Copilot Instructions for GitHubBot

## Project Overview
This project is a GitHub bot application built with Python. It uses agent-based architecture to interact with GitHub repositories and provides a web interface for controlling and monitoring these interactions.

## Code Structure
- `Python/agents.py`: Contains agent classes for handling GitHub interactions
- `Python/github_utils.py`: Utility functions for GitHub API operations
- `Python/app.py`: Main application entry point and web server setup
- `Python/templates/`: Contains HTML templates for the web interface
- `Python/config.json`: Configuration settings for the application
- `Python/PromptTemplate.json`: Templates for AI prompts

## Common Development Tasks

### Adding New Agent Functionality
When extending agent capabilities in `agents.py`:
1. Determine if new functionality belongs in an existing agent or requires a new agent class
2. Maintain consistent error handling patterns
3. Update any corresponding UI elements in templates

### Working with GitHub API
When modifying `github_utils.py`:
1. Use existing utility functions when possible
2. Follow rate limiting best practices
3. Add proper error handling for API failures

### Modifying Web Interface
When updating the web interface:
1. Keep styles consistent with `templates/cat_style.css`
2. Ensure responsive design principles are followed
3. Test UI changes across different devices and browsers

## Best Practices for Using Copilot

### Effective Prompting
- Be specific about which file you're working with
- Specify language requirements (Python 3.11+)
- Include error handling in your prompts

### Project-Specific Conventions
- Follow PEP 8 style guidelines for Python code
- Document all functions and classes with docstrings
- Add meaningful comments to complex algorithms
- Use type hints for function parameters and return values

### Testing Suggestions
- Write unit tests for all new utility functions
- Test GitHub API interactions with mock objects
- Validate web interface changes with browser testing

## Setup Guide
For detailed setup instructions, refer to `Docs/SetupEnvironment.md`.
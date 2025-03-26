# GitHub Bot Project Setup Guide

This document provides instructions for setting up and running the GitHub Bot project.

## Environment Requirements

### Python
- Python 3.8 or higher
- Conda package manager

## Required Packages

The project requires the following Python packages:
- Flask 2.3.3
- Requests 2.31.0
- DuckDuckGo-Search 3.0.2
- PyGithub
- langchain
- openai
- dotenv

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/GitHubBot.git
cd GitHubBot
```

### 2. Set up Conda Environment

First, ensure you have Conda installed:

```bash
# Check Conda version
conda --version

# If Conda is not installed, download and install Miniconda or Anaconda
# Follow the instructions at: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
```

### 3. Create and Activate a Conda Environment

```bash
# Create a new conda environment with Python 3.8
conda create -n githubbot python=3.8

# Activate the conda environment
conda activate githubbot

# To switch between Python versions if needed
conda install python=3.9  # Replace with desired version
```

### 4. Install Required Packages

```bash
# Install all required packages
pip install Flask==2.3.3 requests==2.31.0 duckduckgo-search==3.0.2 PyGithub langchain openai python-dotenv

# Alternatively, if a requirements.txt file is included, use:
# pip install -r requirements.txt
```

### 5. Configuration Setup

Create a `config.json` file in the project root directory with the following structure:

```json
{
    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "hf_token": "your_huggingface_token",
    "together_ai_token": "your_together_ai_token"
}
```

Replace the placeholder values with your actual API tokens:
- `your_huggingface_token`: Your Hugging Face API token
- `your_together_ai_token`: Your Together AI API token

## Running the Project

Navigate to the Python directory and run the app.py file:

```bash
# Ensure your conda environment is activated
conda activate githubbot

# Run the application
cd Python
python app.py
```

The application will be available at: `http://localhost:5000`

## Project Structure

- `Python/` - Contains all Python source code
  - `app.py` - Main application file
  - `agents.py` - Agent-related functionality
  - `github_utils.py` - GitHub API utility functions
- `Docs/` - Documentation
- `templates/` - Frontend HTML templates

## Troubleshooting

If you encounter any issues:

1. Ensure the `config.json` file is correctly formatted and contains valid API tokens
2. Verify that all required packages are installed in your conda environment
3. Make sure you have activated the conda environment before running the application
4. Check conda environment with `conda list` to verify package installations

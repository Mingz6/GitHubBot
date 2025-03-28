---
title: Demo1
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Paw-fect Website Researcher

A tool that helps you analyze websites and answer questions about their content.

## Description

This application allows users to:
- Analyze any website by providing its URL
- Get an overview of the website's content
- Ask questions about the website and receive AI-powered answers

## Deployment to Hugging Face Spaces

### Manual Deployment

1. Create a new Space on Hugging Face:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose a name for your Space
   - Select "Docker" as the SDK
   - Choose your visibility settings (Public or Private)

2. Clone your new Space repository:
   ```
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```

3. Copy all files from this project to the cloned repository

4. Push the changes to deploy:
   ```
   git add .
   git commit -m "Initial deployment"
   git push
   ```

### Automated Deployment via GitHub

1. Push this repository to GitHub.

2. Create a new Space on Hugging Face:
   - Choose "From GitHub" during Space creation
   - Connect your GitHub account and select this repository
   - Choose "Docker" as the Space SDK
   - The `space.yml` file will configure the deployment settings

3. Hugging Face will automatically build and deploy your application.

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Access the app at http://localhost:5009

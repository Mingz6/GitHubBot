from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import requests
from bs4 import BeautifulSoup
from agents import SummarizerAgent, InsightAgent, RecommenderAgent, QuestionGeneratorAgent, CLISetupAgent, ChatbotAgent, PRReviewAgent
from github_utils import get_repo_content, get_repo_structure, get_repo_metadata, is_github_url, get_pr_details, get_target_branch_code, verify_github_credentials
import os

app = Flask(__name__)
summarizer = SummarizerAgent()
insight_agent = InsightAgent()
recommender_agent = RecommenderAgent()
question_generator = QuestionGeneratorAgent()
cli_setup_agent = CLISetupAgent()
chatbot_agent = ChatbotAgent()  # Initialize the new ChatbotAgent
pr_review_agent = PRReviewAgent()  # Initialize the new PRReviewAgent

@app.route('/templates/<path:filename>')
def serve_template_file(filename):
    """Serve static files from the templates directory."""
    return send_from_directory(os.path.join(app.root_path, 'templates'), filename)

@app.route("/", methods=["GET", "POST"])
def index():
    repo_data = None
    insights = None
    
    if request.method == "POST":
        repo_url = request.form.get("repo_url", "")
        if repo_url and is_github_url(repo_url):
            # Get repository content
            repo_data = {
                "url": repo_url,
                "metadata": get_repo_metadata(repo_url),
                "structure": get_repo_structure(repo_url),
                "content": get_repo_content(repo_url)
            }
            
    return render_template(
        "index.html", repo_data=repo_data, insights=insights
    )

@app.route("/summarize", methods=["POST"])
def summarize():
    repo_content = request.json.get("content", {})
    if not repo_content:
        return jsonify({"error": "No content provided"}), 400

    try:
        # Generate summaries for each file
        summaries = {}
        for filename, content in repo_content.items():
            if isinstance(content, str) and len(content) > 0:
                # Limit content to 1000 words to avoid token limits
                words = content.split()
                if len(words) > 1000:
                    content_for_summary = " ".join(words[:1000])
                else:
                    content_for_summary = content

                # Generate summary
                summary = summarizer.process(content_for_summary)
                summaries[filename] = summary

        return jsonify({"summaries": summaries})
    except Exception as e:
        return jsonify({"error": f"Error generating summaries: {str(e)}"}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    summaries = request.json.get("summaries", {})
    if not summaries:
        return jsonify({"error": "No summaries provided"}), 400

    try:
        # Generate insights from all summaries
        summary_texts = list(summaries.values())
        insights = insight_agent.process_text(summary_texts)

        return jsonify({"insights": insights})
    except Exception as e:
        return jsonify({"error": f"Error generating insights: {str(e)}"}), 500

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    insights = data.get("insights", "")
    summaries = data.get("summaries", [])
    user_goal = data.get("goal", "")
    persona = data.get("persona", "")

    if not insights or not summaries:
        return jsonify({"error": "Missing required data"}), 400

    try:
        recommendations = recommender_agent.process(
            insights, summaries, user_goal, persona
        )
        next_query = recommender_agent.suggest_next_query(
            insights, summaries, user_goal, persona
        )
        return jsonify({"recommendations": recommendations, "next_query": next_query})
    except Exception as e:
        return jsonify({"error": f"Error generating recommendations: {str(e)}"}), 500

@app.route("/generate_questions", methods=["POST"])
def generate_questions():
    data = request.json
    content = data.get("content", "")
    category = data.get("category", "repository")
    source = data.get("source", "")

    if not content or not source:
        return jsonify({"error": "Missing required data"}), 400

    try:
        questions = question_generator.generate_questions(content, category, source)
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": f"Error generating questions: {str(e)}"}), 500

@app.route("/workflow", methods=["POST"])
def workflow():
    """Complete workflow from repository URL to recommendations."""
    repo_url = request.json.get("repo_url", "")
    user_goal = request.json.get("goal", "Understand the codebase")
    persona = request.json.get("persona", "Developer")
    github_auth = request.json.get("github_auth", None)
    
    if not repo_url or not is_github_url(repo_url):
        return jsonify({"error": "Valid GitHub repository URL required"}), 400
    
    try:
        # Prepare authentication if provided
        auth = None
        if github_auth and 'username' in github_auth and 'token' in github_auth:
            auth = (github_auth['username'], github_auth['token'])
        
        # Step 1: Get repository content
        repo_content = get_repo_content(repo_url, auth=auth)
        if "error" in repo_content:
            return jsonify({"error": repo_content["error"]}), 500
        
        # Get repository metadata
        repo_metadata = get_repo_metadata(repo_url, auth=auth)
        # Ensure repo_metadata has the URL
        if "url" not in repo_metadata:
            repo_metadata["url"] = repo_url
            
        # Step 2: Generate summaries
        summaries = {}
        for filename, content in repo_content.items():
            words = content.split()
            if len(words) > 1000:
                content_for_summary = " ".join(words[:1000])
            else:
                content_for_summary = content
                
            summary = summarizer.process(content_for_summary)
            summaries[filename] = summary
        
        # New Step: Generate CLI setup instructions
        try:
            cli_setup = cli_setup_agent.generate_setup_instructions(repo_content, repo_metadata)
            if not cli_setup or len(cli_setup.strip()) < 10:
                cli_setup = "Sorry, couldn't generate setup instructions for this repository."
        except Exception as e:
            print(f"Error in CLI setup generation: {str(e)}")
            cli_setup = "Error generating setup instructions. Please check the repository and try again."
            
        # Step 3: Generate insights
        summary_texts = list(summaries.values())
        insights = insight_agent.process_text(summary_texts)
        
        # Step 4: Generate recommendations
        recommendations = recommender_agent.process(
            insights, summary_texts, user_goal, persona
        )
        
        # Step 5: Suggest next exploration area
        next_area = recommender_agent.suggest_next_query(
            insights, summary_texts, user_goal, persona
        )
        
        # Step 6: Generate questions
        repo_name = repo_metadata.get("name", "GitHub Repository")
        questions = question_generator.generate_questions(
            repo_name, "repository", repo_url
        )
        
        return jsonify({
            "summaries": summaries,
            "cli_setup": cli_setup,
            "insights": insights,
            "recommendations": recommendations,
            "next_area": next_area,
            "questions": questions,
            "repo_content": repo_content,  # Add repository content for the chatbot
            "repo_metadata": repo_metadata  # Add repository metadata for the chatbot
        })
    
    except Exception as e:
        return jsonify({"error": f"Error in workflow: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chatbot questions about a repository."""
    data = request.json
    question = data.get("question", "")
    repo_url = data.get("repo_url", "")
    repo_content = data.get("repo_content", {})
    repo_metadata = data.get("repo_metadata", {})
    summaries = data.get("summaries", {})
    insights = data.get("insights", "")
    github_auth = data.get("github_auth", None)
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Prepare authentication if provided
    auth = None
    if github_auth and 'username' in github_auth and 'token' in github_auth:
        auth = (github_auth['username'], github_auth['token'])
        
    if not repo_content and repo_url:
        # If content isn't provided but URL is, fetch the repository content
        if is_github_url(repo_url):
            repo_content = get_repo_content(repo_url, auth=auth)
            repo_metadata = get_repo_metadata(repo_url, auth=auth)
            # Ensure repo_metadata has the URL
            if "url" not in repo_metadata:
                repo_metadata["url"] = repo_url
        else:
            return jsonify({"error": "Valid GitHub repository URL required"}), 400
    
    if not repo_content:
        return jsonify({"error": "No repository content provided"}), 400
    
    try:
        # Use the chatbot agent to answer the question
        answer = chatbot_agent.answer_question(
            question=question,
            repo_content=repo_content,
            repo_metadata=repo_metadata,
            summaries=summaries,
            insights=insights
        )
        
        return jsonify({
            "answer": answer,
            "question": question
        })
    except Exception as e:
        return jsonify({"error": f"Error answering question: {str(e)}"}), 500

@app.route("/review_pr", methods=["POST"])
def review_pr():
    """Review a GitHub Pull Request and provide professional code suggestions."""
    pr_url = request.json.get("pr_url", "")
    max_files = request.json.get("max_files", 25)  # Default to 25 files
    file_types = request.json.get("file_types", None)  # Default to all code files
    github_auth = request.json.get("github_auth", None)  # GitHub authentication
    
    if not pr_url:
        return jsonify({"error": "No PR URL provided"}), 400
    
    try:
        # Prepare authentication if provided
        auth = None
        if github_auth and 'username' in github_auth and 'token' in github_auth:
            auth = (github_auth['username'], github_auth['token'])
            
        # Step 1: Fetch PR details
        pr_details = get_pr_details(pr_url, max_files=max_files, file_types=file_types, auth=auth)
        if "error" in pr_details:
            return jsonify({"error": pr_details["error"]}), 500
        
        # Step 2: Fetch target branch code
        target_branch_code = get_target_branch_code(pr_url, max_files=max_files, file_types=file_types, auth=auth)
        if "error" in target_branch_code:
            return jsonify({"error": target_branch_code["error"]}), 500
        
        # Step 3: Generate PR review
        review_result = pr_review_agent.review_pr(pr_details, target_branch_code)
        
        # Step 4: Return the results
        return jsonify({
            "pr_title": pr_details.get("title", ""),
            "pr_user": pr_details.get("user", ""),
            "target_branch": pr_details.get("target_branch", ""),
            "source_branch": pr_details.get("source_branch", ""),
            "changed_files_count": len(pr_details.get("changed_files", [])),
            "total_file_count": pr_details.get("total_file_count", 0),
            "review": review_result.get("review", "Error generating review"),
            "analyzed_files": [file["filename"] for file in pr_details.get("changed_files", [])]
        })
    
    except Exception as e:
        return jsonify({"error": f"Error reviewing PR: {str(e)}"}), 500

@app.route("/verify_github_credentials", methods=["POST"])
def verify_credentials():
    """Verify GitHub credentials and return status."""
    data = request.json
    github_username = data.get("github_username", "")
    github_token = data.get("github_token", "")
    
    if not github_username or not github_token:
        return jsonify({"valid": False, "error": "Missing username or token"}), 400
    
    # Verify the credentials
    is_valid = verify_github_credentials(github_username, github_token)
    
    if is_valid:
        return jsonify({"valid": True, "message": "Successfully authenticated with GitHub"})
    else:
        return jsonify({"valid": False, "error": "Invalid GitHub credentials"}), 401

if __name__ == "__main__":
    # Use environment variables for port if available (needed for Hugging Face)
    port = int(os.environ.get('PORT', 5001))
    # Bind to 0.0.0.0 instead of 127.0.0.1 to be accessible from outside the container
    app.run(debug=False, host='0.0.0.0', port=port)

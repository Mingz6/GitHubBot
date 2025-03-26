from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import requests
from bs4 import BeautifulSoup
from agents import SummarizerAgent, InsightAgent, RecommenderAgent, QuestionGeneratorAgent, CLISetupAgent, WebsiteContentSummarizer, QuestionAnswerer, WebsiteSummarizer
from github_utils import get_repo_content, get_repo_structure, get_repo_metadata, is_github_url
from web_utils import extract_webpage_content, extract_relevant_sections, is_valid_url
import os
import json

app = Flask(__name__)
summarizer = SummarizerAgent()
insight_agent = InsightAgent()
recommender_agent = RecommenderAgent()
question_generator = QuestionGeneratorAgent()
cli_setup_agent = CLISetupAgent()
website_summarizer = WebsiteContentSummarizer()
question_answerer = QuestionAnswerer()
website_overview = WebsiteSummarizer()

# Create a cache for website content to avoid repeated scraping
website_cache = {}

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
    
    if not repo_url or not is_github_url(repo_url):
        return jsonify({"error": "Valid GitHub repository URL required"}), 400
    
    try:
        # Step 1: Get repository content
        repo_content = get_repo_content(repo_url)
        if "error" in repo_content:
            return jsonify({"error": repo_content["error"]}), 500
        
        # Get repository metadata
        repo_metadata = get_repo_metadata(repo_url)
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
            "questions": questions
        })
    
    except Exception as e:
        return jsonify({"error": f"Error in workflow: {str(e)}"}), 500

@app.route("/extract_website", methods=["POST"])
def extract_website():
    """Extract content from a website URL."""
    website_url = request.json.get("website_url", "")
    
    if not website_url or not is_valid_url(website_url):
        return jsonify({"error": "Valid website URL required"}), 400
        
    try:
        # Check if website is already in cache
        if website_url in website_cache:
            content = website_cache[website_url]
        else:
            # Extract content from the website
            content = extract_webpage_content(website_url)
            
            # Cache the content
            website_cache[website_url] = content
            
        if "error" in content:
            return jsonify({"error": content["error"]}), 500
            
        # Generate an overview of the website
        overview = website_overview.create_website_overview(content)
        
        # Get the first 3 page URLs for reference
        page_urls = list(content.keys())[:3]
        
        return jsonify({
            "status": "success",
            "overview": overview,
            "page_count": len(content),
            "sample_pages": page_urls
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing website: {str(e)}"}), 500

@app.route("/answer_question", methods=["POST"])
def answer_question():
    """Answer a user's question based on website content."""
    data = request.json
    website_url = data.get("website_url", "")
    question = data.get("question", "")
    
    if not website_url or not is_valid_url(website_url) or not question:
        return jsonify({"error": "Valid website URL and question required"}), 400
        
    try:
        # Check if website is already in cache
        if website_url in website_cache:
            content = website_cache[website_url]
        else:
            # Extract content from the website
            content = extract_webpage_content(website_url)
            
            # Cache the content
            website_cache[website_url] = content
            
        if "error" in content:
            return jsonify({"error": content["error"]}), 500
            
        # Extract relevant sections for the question
        relevant_content, source_urls = extract_relevant_sections(content, question)
        
        if not relevant_content:
            return jsonify({
                "answer": "Sorry, no information found on the website that answers your question.",
                "sources": []
            })
            
        # Generate an answer to the question
        answer = question_answerer.answer_question(question, relevant_content, source_urls)
        
        return jsonify({
            "answer": answer,
            "sources": source_urls[:3]  # Return up to 3 source URLs
        })
        
    except Exception as e:
        return jsonify({"error": f"Error answering question: {str(e)}"}), 500

@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Clear the website content cache."""
    global website_cache
    website_cache = {}
    return jsonify({"status": "Cache cleared successfully"})

if __name__ == "__main__":
    app.run(debug=True, port=5009, host="0.0.0.0")

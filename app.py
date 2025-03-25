from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import requests
from bs4 import BeautifulSoup
from agents import SummarizerAgent, InsightAgent, RecommenderAgent, QuestionGeneratorAgent
from github_utils import get_repo_content, get_repo_structure, get_repo_metadata, is_github_url
import os

app = Flask(__name__)
summarizer = SummarizerAgent()
insight_agent = InsightAgent()
recommender_agent = RecommenderAgent()
question_generator = QuestionGeneratorAgent()

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
        repo_metadata = get_repo_metadata(repo_url)
        repo_name = repo_metadata.get("name", "GitHub Repository")
        questions = question_generator.generate_questions(
            repo_name, "repository", repo_url
        )
        
        return jsonify({
            "summaries": summaries,
            "insights": insights,
            "recommendations": recommendations,
            "next_area": next_area,
            "questions": questions
        })
    
    except Exception as e:
        return jsonify({"error": f"Error in workflow: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5008, host="0.0.0.0")

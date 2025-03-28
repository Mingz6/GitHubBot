from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import requests
from bs4 import BeautifulSoup
from agents import SummarizerAgent, InsightAgent, RecommenderAgent, QuestionGeneratorAgent, CLISetupAgent, WebsiteContentSummarizer, QuestionAnswerer, WebsiteSummarizer
from github_utils import get_repo_content, get_repo_structure, get_repo_metadata, is_github_url
from web_utils import extract_webpage_content, extract_relevant_sections, is_valid_url, parse_knowledge_file
from rag_utils import RAGProcessor  # Import our new RAG processor
import os
import json

# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Flask app
app = Flask(__name__)

# Read configuration for local development
def load_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as config_file:
                return json.load(config_file)
    except Exception as e:
        print(f"Error loading config.json: {str(e)}")
    return {}

# Initialize config
config = load_config()

# Set model from config or use default
model_name = config.get("model", "mistralai/Mixtral-8x7B-Instruct-v0.1")

# Initialize agents
summarizer = SummarizerAgent(model=model_name)
insight_agent = InsightAgent(model=model_name)
recommender_agent = RecommenderAgent(model=model_name)
question_generator = QuestionGeneratorAgent(model=model_name)
cli_setup_agent = CLISetupAgent(model=model_name)
website_summarizer = WebsiteContentSummarizer(model=model_name)
question_answerer = QuestionAnswerer(model=model_name)
website_overview = WebsiteSummarizer(model=model_name)

# Initialize the RAG processor
rag_processor = RAGProcessor(model_name=model_name)

# Create a cache for website content to avoid repeated scraping
website_cache = {}

# Add a route for serving static files - crucial for Hugging Face Spaces
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

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

@app.route("/process_knowledge_file", methods=["POST"])
def process_knowledge_file():
    """Process an uploaded knowledge text file."""
    try:
        # Check if file was uploaded
        if 'knowledge_file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['knowledge_file']
        if file.filename == '':
            return jsonify({"error": "Empty file name"}), 400
            
        if file and file.filename.endswith('.txt'):
            # Read the file content
            file_content = file.read().decode('utf-8')
            
            # Parse the knowledge file
            knowledge_data = parse_knowledge_file(file_content)
            if not knowledge_data or 'id' not in knowledge_data:
                return jsonify({"error": "Failed to parse knowledge file"}), 400
                
            kb_id = knowledge_data["id"]
            
            # Keep track of processed websites and combined content
            processed_websites = []
            combined_content = knowledge_data["content"].copy() if "content" in knowledge_data else []
                
            # Process any websites mentioned in the knowledge file
            for website_info in knowledge_data.get("websites", []):
                website_url = website_info.get("url")
                max_pages = website_info.get("max_pages", 10)
                
                if website_url and is_valid_url(website_url):
                    processed_websites.append(website_url)
                    print(f"Processing website from knowledge file: {website_url} (max pages: {max_pages})")
                    
                    # Extract content from website (if not already cached)
                    if website_url not in website_cache:
                        print(f"Crawling website: {website_url}")
                        web_content = extract_webpage_content(website_url, max_pages=max_pages)
                        
                        # Check for errors in extraction
                        if "error" in web_content:
                            print(f"Error extracting website content: {web_content['error']}")
                            continue
                            
                        website_cache[website_url] = web_content
                    else:
                        print(f"Using cached content for: {website_url}")
                        web_content = website_cache[website_url]
                    
                    # Extract key information from the website to add to the combined knowledge base
                    website_summary = []
                    
                    # First get the main pages for important information
                    for page_url, page_data in web_content.items():
                        # Extract page content based on format
                        page_content = ""
                        page_title = ""
                        
                        if isinstance(page_data, dict) and "content" in page_data:
                            page_content = page_data["content"]
                            page_title = page_data.get("title", page_url)
                        elif isinstance(page_data, str):
                            page_content = page_data
                            page_title = page_url
                            
                        # Skip if no content
                        if not page_content:
                            continue
                        
                        # Process each paragraph to extract key information
                        paragraphs = page_content.split('\n\n')
                        
                        for para in paragraphs:
                            # Skip very short paragraphs
                            if len(para) < 50:
                                continue
                                
                            # Extract key information about the website, organization names, etc.
                            if any(term in para.lower() for term in ["about", "welcome", "mission", "overview", "description", "crna", "college"]):
                                website_summary.append(f"From {page_url}: {para}")
                    
                    # Add a summary of the website to the combined content
                    if website_summary:
                        print(f"Adding {len(website_summary)} items from website to knowledge base")
                        combined_content.extend(website_summary)
                    
                    # Now process with RAG as well so we can search it directly too
                    print(f"Processing website content with RAG: {website_url}")
                    rag_result = rag_processor.process_website_content(website_url, web_content)
                    
                    if "error" in rag_result:
                        print(f"Error processing website with RAG: {rag_result['error']}")
                    else:
                        print(f"Successfully processed website with RAG: {rag_result.get('message', 'No message')}")
            
            # Update the knowledge base with the combined content
            knowledge_data["content"] = combined_content
            
            # Process the updated knowledge base with RAG
            print(f"Processing combined knowledge base with {len(combined_content)} items")
            rag_result = rag_processor.process_knowledge_base(knowledge_data)
            if "error" in rag_result:
                print(f"RAG processing warning: {rag_result['error']}")
            
            # Add more detailed response
            response_message = {
                "status": "success",
                "id": knowledge_data["id"],
                "content_preview": knowledge_data["content_preview"],
                "line_count": len(combined_content),
                "url_count": len(knowledge_data.get("websites", [])),
                "processed_websites": processed_websites
            }
            
            # Add RAG status information
            if processed_websites:
                response_message["rag_status"] = "Websites have been processed and combined with text content"
                
            return jsonify(response_message)
        else:
            return jsonify({"error": "Invalid file format. Only .txt files are supported"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

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
    """Extract content from a website URL and process with RAG."""
    website_url = request.json.get("website_url", "")
    
    if not website_url or not is_valid_url(website_url):
        return jsonify({"error": "Valid website URL required"}), 400
        
    try:
        # Check if website is already in cache
        if website_url in website_cache:
            content = website_cache[website_url]
        else:
            # Extract content from the website
            max_pages = int(request.json.get('max_pages', 10))
            content = extract_webpage_content(website_url, max_pages=max_pages)
            
            # Cache the content
            website_cache[website_url] = content
            
        if "error" in content:
            return jsonify({"error": content["error"]}), 500
        
        # Process the content with RAG
        rag_result = rag_processor.process_website_content(website_url, content)
        if "error" in rag_result:
            print(f"RAG processing warning: {rag_result['error']}")
            
        # Generate an overview of the website
        overview = website_overview.create_website_overview(content)
        
        # Get the first 3 page URLs for reference
        page_urls = list(content.keys())[:3]
        
        return jsonify({
            "status": "success",
            "overview": overview,
            "page_count": len(content),
            "sample_pages": page_urls,
            "rag_enabled": "error" not in rag_result,
            "rag_status": rag_result.get("message", "RAG processing complete")
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing website: {str(e)}"}), 500

@app.route("/answer_question", methods=["POST"])
def answer_question():
    """Answer a user's question based on knowledge file or website content."""
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        # Check if we're using a knowledge base or a website
        knowledge_id = data.get("knowledge_id", None)
        website_url = data.get("website_url", None)
        
        # Create a list of common variations/typos for important terms
        keyword_variations = {
            "toll": ["toll", "troll", "tool", "toil"],
            "free": ["free", "freed", "fee", "fees"],
            "crna": ["crna", "cna", "rna", "nurse", "nursing"]  # Add CRNA variations
        }
        
        # Normalize the question by replacing common variations
        normalized_question = question.lower()
        for base_word, variations in keyword_variations.items():
            for variant in variations:
                if variant in normalized_question:
                    # Add both the variant and base form to improve matching
                    if variant != base_word:
                        normalized_question = normalized_question.replace(variant, f"{variant} {base_word}")
        
        print(f"Original question: {question}")
        print(f"Normalized question: {normalized_question}")
        
        if knowledge_id:
            # Using a knowledge base
            print(f"Answering question using knowledge ID: {knowledge_id}")
            
            # First try to find the answer in the knowledge base
            kb_result = rag_processor.retrieve_from_knowledge_base(knowledge_id, normalized_question)
            if "error" in kb_result:
                return jsonify({"error": kb_result["error"]}), 404
                
            knowledge_content = kb_result.get("content", [])
            knowledge_sources = kb_result.get("sources", [])
            websites_to_check = kb_result.get("websites", [])
            nursing_related = kb_result.get("nursing_related", False)
            name_related = kb_result.get("name_related", False)
            
            if knowledge_content:
                print(f"Found {len(knowledge_content)} relevant items in knowledge base")
                # Format the retrieved content
                relevant_content = "\n\n".join(knowledge_content)
                source_urls = knowledge_sources
                
                # Generate an answer to the question
                answer = question_answerer.answer_question(normalized_question, relevant_content, knowledge_sources)
                
                return jsonify({
                    "answer": answer,
                    "sources": source_urls[:5],  # Return up to 5 sources
                    "method": "knowledge_base"
                })
            
            # If no content was found in the knowledge base text directly, or if it's a nursing-related query,
            # we should still check any websites referenced in the knowledge file
            if websites_to_check and (not knowledge_content or nursing_related):
                print(f"Checking {len(websites_to_check)} websites from knowledge file")
                
                all_website_content = {}
                all_website_sources = []
                
                # Process all websites first to make sure they're indexed
                for website_info in websites_to_check:
                    website_url = website_info.get("url")
                    max_pages = website_info.get("max_pages", 10)
                    
                    if website_url and is_valid_url(website_url):
                        print(f"Checking website: {website_url}")
                        
                        # Always check cached content first
                        web_content = {}
                        if website_url in website_cache:
                            web_content = website_cache[website_url]
                            print(f"Using cached content for {website_url}")
                        else:
                            # Fetch content if not in cache
                            print(f"Fetching website content for {website_url}")
                            web_content = extract_webpage_content(website_url, max_pages=max_pages)
                            if "error" in web_content:
                                print(f"Error fetching website: {web_content['error']}")
                                continue
                                
                            website_cache[website_url] = web_content
                        
                        # Make sure website is processed for RAG
                        if website_url not in rag_processor.vectorstores:
                            print(f"Processing website with RAG first: {website_url}")
                            rag_result = rag_processor.process_website_content(website_url, web_content)
                            if "error" in rag_result:
                                print(f"Warning: Could not process with RAG: {rag_result['error']}")
                        
                        # Try both the original question and normalized question for broader matching
                        # First, check for case-insensitive match on specific terms like "CRNA"
                        found_content = False
                        search_terms = [question, normalized_question]
                        
                        # Add special case for acronyms
                        if question.isupper() and len(question) <= 6:
                            # It's an acronym, so add variations for searching
                            search_terms.append(question.lower())
                            search_terms.append(question.upper())
                        
                        # Try RAG search first (more accurate)
                        if website_url in rag_processor.vectorstores:
                            print("Using RAG to search website content")
                            for search_question in search_terms:
                                retrieval_result = rag_processor.retrieve_relevant_content(website_url, search_question, top_k=5)
                                if "error" not in retrieval_result and retrieval_result.get("results"):
                                    retrieved_docs = retrieval_result["results"]
                                    print(f"Found {len(retrieved_docs)} relevant documents via RAG")
                                    
                                    # Add retrieved content to our collection
                                    relevant_content = "\n\n".join([doc["content"] for doc in retrieved_docs])
                                    all_website_content[website_url] = relevant_content
                                    
                                    for doc in retrieved_docs:
                                        source_url = doc.get("source")
                                        if source_url and source_url not in all_website_sources:
                                            all_website_sources.append(source_url)
                                    
                                    found_content = True
                        
                        # Fallback to manual search methods
                        if not found_content:
                            for search_question in search_terms:
                                print(f"Searching with term: {search_question}")
                                
                                # Use direct string matching for acronyms
                                if len(search_question) <= 6 and search_question.isupper():
                                    for url, page_data in web_content.items():
                                        page_content = ""
                                        if isinstance(page_data, dict) and "content" in page_data:
                                            page_content = page_data["content"]
                                        elif isinstance(page_data, str):
                                            page_content = page_data
                                        
                                        # Look for the acronym specifically
                                        if search_question in page_content or search_question.lower() in page_content.lower():
                                            print(f"Found direct match for {search_question} in {url}")
                                            # Extract the paragraph containing the match
                                            paragraphs = page_content.split('\n\n')
                                            for para in paragraphs:
                                                if search_question in para or search_question.lower() in para.lower():
                                                    all_website_content[url] = para
                                                    if url not in all_website_sources:
                                                        all_website_sources.append(url)
                                                    found_content = True
                                
                                # Use keyword search for all questions
                                content_text, urls = extract_relevant_sections(web_content, search_question)
                                if content_text:
                                    print(f"Found relevant content via keyword search from {len(urls)} URLs")
                                    all_website_content[website_url] = content_text
                                    for url in urls:
                                        if url not in all_website_sources:
                                            all_website_sources.append(url)
                                    found_content = True
                
                # If we found relevant content from any website, combine with knowledge content for a complete answer
                if all_website_content:
                    print(f"Combining {len(knowledge_content)} knowledge items with website content")
                    
                    # Start with the knowledge content
                    combined_sources = knowledge_sources.copy() if knowledge_sources else []
                    combined_content = []
                    
                    # Add the knowledge text content
                    if knowledge_content:
                        combined_content.append("\n\n".join(knowledge_content))
                    
                    # Add the website content
                    combined_content.extend(list(all_website_content.values()))
                    combined_sources.extend([url for url in all_website_sources if url not in combined_sources])
                    
                    # Join all content together
                    all_content = "\n\n".join(combined_content)
                    print(f"Answering with combined content: {len(all_content)} chars from {len(combined_sources)} sources")
                    
                    # Trim content if it's too long
                    max_content_length = 15000
                    if len(all_content) > max_content_length:
                        all_content = all_content[:max_content_length] + "...[content truncated]"
                    
                    # Generate the answer using both the original and normalized questions for context
                    full_question = f"{question} (also looking for: {normalized_question})"
                    answer = question_answerer.answer_question(full_question, all_content, combined_sources)
                    
                    return jsonify({
                        "answer": answer,
                        "sources": combined_sources[:5],
                        "method": "combined_knowledge"
                    })
                elif knowledge_content:
                    # If we have knowledge content but no website content, use just the knowledge content
                    print("Using only knowledge base content (no relevant website content found)")
                    relevant_content = "\n\n".join(knowledge_content)
                    answer = question_answerer.answer_question(normalized_question, relevant_content, knowledge_sources)
                    
                    return jsonify({
                        "answer": answer,
                        "sources": knowledge_sources[:5],
                        "method": "knowledge_base_only"
                    })
                else:
                    print("No relevant content found in any of the websites")
            
            # If we still don't have relevant content, try direct web search
            if websites_to_check:
                # Try to directly search the web for this specific question
                # Format a direct search URL to the websites
                for website_info in websites_to_check:
                    website_url = website_info.get("url")
                    if website_url and is_valid_url(website_url):
                        domain = get_domain(website_url)
                        search_keywords = "+".join(normalized_question.lower().split())
                        # If the question contains "toll free" or similar, provide special guidance
                        if any(term in normalized_question.lower() for term in ['toll', 'troll', 'free', 'number', 'contact']):
                            return jsonify({
                                "answer": f"I couldn't find specific information about '{question}' in the knowledge base or the linked websites. For toll-free or contact numbers, please check the Contact Us or About Us pages on the {domain} website directly.",
                                "sources": [website_url],
                                "method": "suggested_search"
                            })
                        else:
                            return jsonify({
                                "answer": f"I couldn't find specific information about '{question}' in the knowledge base or the linked websites. You might try searching directly on {domain} for more information.",
                                "sources": [website_url],
                                "method": "suggested_search"
                            })
            
            # If we still don't have relevant content, return a generic response
            return jsonify({
                "answer": "I couldn't find information about this in your knowledge file or any linked websites. The information may not be present in these sources.",
                "sources": [],
                "method": "none"
            })
                
        elif website_url:
            # Using a website directly
            print(f"Answering question using website URL: {website_url}")
            
            use_rag = True  # Default to using RAG
            
            # Always do a fresh crawl for toll-free related questions to ensure we have the latest info
            force_fresh_crawl = any(term in normalized_question.lower() for term in 
                                ['toll', 'troll', 'free', 'fee', 'cost', 'phone', 'number'])
            
            # Check if website is already in cache and determine if we should refresh
            if website_url in website_cache and not force_fresh_crawl:
                content = website_cache[website_url]
                print("Using cached website content")
            else:
                # Extract content from the website
                max_pages = int(data.get('max_pages', 10))
                print(f"Fetching fresh website content (force_fresh={force_fresh_crawl})")
                content = extract_webpage_content(website_url, max_pages=max_pages)
                
                # Cache the content
                website_cache[website_url] = content
                
                # Clear existing RAG data if we're refreshing
                if website_url in rag_processor.vectorstores:
                    print(f"Clearing existing RAG data for {website_url}")
                    rag_processor.clear_website_data(website_url)
                
            if "error" in content:
                return jsonify({"error": content["error"]}), 500
                
            # If RAG is enabled and the website hasn't been processed yet, process it now
            if use_rag and website_url not in rag_processor.vectorstores:
                print("Processing website content with RAG")
                rag_result = rag_processor.process_website_content(website_url, content)
                if "error" in rag_result:
                    print(f"Warning: Could not process with RAG: {rag_result['error']}")
                    use_rag = False
            
            # Try both original and normalized question forms
            all_relevant_content = []
            all_source_urls = []
            
            # Add special case for acronyms like CRNA
            search_terms = [question, normalized_question]
            if question.isupper() and len(question) <= 6:
                # It's an acronym, so add variations for searching
                search_terms.append(question.lower())
                print(f"Adding acronym variation: {question.lower()}")
                
                # For "What is X?" questions about acronyms, add specific pattern searches
                if normalized_question.startswith("what is"):
                    search_terms.append(f"what is {question}")
                    search_terms.append(f"what is {question.lower()}")
                    search_terms.append(f"{question} is")
                    search_terms.append(f"{question} stands for")
                    print(f"Adding definition search patterns for acronym {question}")
                    
            if use_rag:
                # Use RAG to retrieve relevant content with all search terms
                for search_question in search_terms:
                    print(f"Using RAG to find relevant content with query: {search_question}")
                    retrieval_result = rag_processor.retrieve_relevant_content(website_url, search_question, top_k=5)
                    
                    if "error" not in retrieval_result:
                        # Format the retrieved content for the question answerer
                        retrieved_docs = retrieval_result["results"]
                        if retrieved_docs:
                            print(f"Found {len(retrieved_docs)} relevant documents via RAG")
                            relevant_content = "\n\n".join([doc["content"] for doc in retrieved_docs])
                            source_urls = [doc["source"] for doc in retrieved_docs]
                            
                            all_relevant_content.append(relevant_content)
                            all_source_urls.extend([url for url in source_urls if url not in all_source_urls])
            
            # Direct string matching for acronyms
            if question.isupper() and len(question) <= 6:
                print(f"Performing direct acronym search for: {question}")
                for url, page_data in content.items():
                    page_content = ""
                    if isinstance(page_data, dict) and "content" in page_data:
                        page_content = page_data["content"]
                    elif isinstance(page_data, str):
                        page_content = page_data
                    
                    # Look for the acronym specifically
                    if question in page_content or question.lower() in page_content.lower():
                        print(f"Found direct match for {question} in {url}")
                        # Extract paragraphs containing the match
                        paragraphs = page_content.split('\n\n')
                        for para in paragraphs:
                            if question in para or question.lower() in para.lower():
                                # Add a special note to highlight this is a direct match
                                matched_para = f"DIRECT MATCH: {para}"
                                all_relevant_content.append(matched_para)
                                if url not in all_source_urls:
                                    all_source_urls.append(url)
            
            # Also try traditional keyword search with all search terms
            for search_question in search_terms:
                print(f"Using traditional search with query: {search_question}")
                keyword_content, keyword_urls = extract_relevant_sections(content, search_question)
                if keyword_content:
                    all_relevant_content.append(keyword_content)
                    all_source_urls.extend([url for url in keyword_urls if url not in all_source_urls])
            
            # Combine all found content
            if all_relevant_content:
                combined_content = "\n\n".join(all_relevant_content)
                print(f"Combined content length: {len(combined_content)}")
                
                # Limit content size to avoid token limits
                max_chars = 10000
                if len(combined_content) > max_chars:
                    print(f"Truncating content from {len(combined_content)} to {max_chars} characters")
                    combined_content = combined_content[:max_chars] + "... [content truncated due to length]"
                    
                # Generate an answer to the question
                print(f"Generating answer with {len(combined_content)} chars of content from {len(all_source_urls)} sources")
                full_question = f"{question} (also checking for: {normalized_question})"
                answer = question_answerer.answer_question(full_question, combined_content, all_source_urls)
                
                return jsonify({
                    "answer": answer,
                    "sources": all_source_urls[:5],  # Return up to 5 source URLs
                    "method": "combined_search"
                })
            else:
                print("No relevant content found for the question")
                return jsonify({
                    "answer": f"I searched {website_url} but couldn't find information about '{question}'. If you're looking for contact information or toll-free numbers, try checking the Contact Us or About Us pages directly.",
                    "sources": [],
                    "method": "no_match"
                })
        else:
            return jsonify({
                "error": "Either knowledge_id or website_url must be provided"
            }), 400
    
    except Exception as e:
        # Add the missing except block to handle any errors that occur during question answering
        print(f"Error answering question: {str(e)}")
        return jsonify({
            "error": f"Error processing question: {str(e)}"
        }), 500

@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Clear the website content cache and RAG data."""
    global website_cache
    website_url = request.json.get("website_url", None)
    knowledge_id = request.json.get("knowledge_id", None)
    
    if website_url:
        # Clear specific website from cache
        if website_url in website_cache:
            del website_cache[website_url]
        # Also clear RAG data for this website
        rag_result = rag_processor.clear_website_data(website_url)
        return jsonify({"status": "Cache and RAG data cleared for specific website"})
    elif knowledge_id:
        # Clear specific knowledge base
        rag_result = rag_processor.clear_knowledge_base(knowledge_id)
        return jsonify({"status": "Knowledge base data cleared"})
    else:
        # Clear all cache
        website_cache = {}
        # Clear all RAG data
        rag_processor.clear_website_data()
        rag_processor.clear_knowledge_base()
        return jsonify({"status": "All cache and RAG data cleared successfully"})

# New endpoint for RAG-specific operations
@app.route("/rag_status", methods=["POST"])
def rag_status():
    """Get status of RAG processing for a website or knowledge base."""
    website_url = request.json.get("website_url", "")
    knowledge_id = request.json.get("knowledge_id", "")
    
    source_id = website_url or knowledge_id
    
    if not source_id:
        return jsonify({"error": "Website URL or knowledge ID is required"}), 400
        
    # Check if source has been processed with RAG
    if source_id in rag_processor.vectorstores:
        data = rag_processor.vectorstores[source_id]
        source_type = data.get("type", "unknown")
        return jsonify({
            "status": "processed",
            "document_count": data.get("document_count", 0),
            "type": source_type,
            "message": f"{source_type.title()} content has been processed with RAG ({data.get('document_count', 0)} chunks)"
        })
    else:
        return jsonify({
            "status": "not_processed",
            "message": "Content has not been processed with RAG yet"
        })

if __name__ == "__main__":
    # Use environment variables for port if available (needed for Hugging Face)
    port = int(os.environ.get('PORT', 5009))
    app.run(debug=False, host='0.0.0.0', port=port)

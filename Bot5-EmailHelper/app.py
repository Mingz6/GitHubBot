# Convert to flask
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Removing ApiKey import and using config instead
from together import Together
from flask import Flask, render_template, request, jsonify, session
import json
import os

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

config = load_config()
# Get API tokens: prefer environment variables, fallback to config.json for local dev
API_KEY = os.environ.get('TOGETHER_API_KEY') or config.get('together_ai_token')
HGToken = os.environ.get('HF_TOKEN') or config.get('hf_token')

# Helper function to load SentenceTransformer model safely with more detailed logging
def load_sentence_transformer():
    models_to_try = [
        "all-MiniLM-L6-v2",              # Try shorter model name first
        "paraphrase-MiniLM-L3-v2",       # Try an even smaller alternative model
        "sentence-transformers/all-MiniLM-L6-v2",  # Try with full path
        "distilbert-base-nli-mean-tokens"  # Try a different model as final fallback
    ]
    
    last_error = None
    for model in models_to_try:
        try:
            print(f"Attempting to load SentenceTransformer model: {model}")
            transformer = SentenceTransformer(model)
            print(f"Successfully loaded model: {model}")
            return transformer
        except Exception as e:
            last_error = e
            print(f"Failed to load model {model}: {str(e)}")
    
    # If we've tried all models and none worked, raise the last error
    print("All SentenceTransformer model loading attempts failed")
    if last_error:
        raise last_error
    else:
        raise Exception("Unable to load any SentenceTransformer model")

# Function to interact with LLM using Together API
def prompt_llm(prompt, client=None, max_tokens=300):  # Increased default max_tokens from 150 to 300
    if client is None:
        return "Error: Together client not initialized"
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling Together API: {str(e)}"

# New class to handle direct knowledge lookup from Questions.txt
class KnowledgeRetriever:
    def __init__(self):
        self.questions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Questions.txt')
        self.encoder = None
        try:
            self.encoder = load_sentence_transformer()
            print("Successfully loaded SentenceTransformer for KnowledgeRetriever")
        except Exception as e:
            print(f"Error loading SentenceTransformer model in KnowledgeRetriever: {str(e)}")
        self.qa_pairs = self._load_qa_pairs()
        # Precompute embeddings for questions only if encoder is available
        self.question_embeddings = {}
        if self.encoder is not None and self.qa_pairs:
            try:
                self.question_embeddings = {
                    q: self.encoder.encode(q) for q, a in self.qa_pairs.items()
                }
                print(f"Successfully encoded {len(self.question_embeddings)} question embeddings")
            except Exception as e:
                print(f"Error encoding questions: {str(e)}")
        
    def _load_qa_pairs(self):
        """Load question-answer pairs from Questions.txt"""
        qa_pairs = {}
        try:
            if os.path.exists(self.questions_file):
                with open(self.questions_file, 'r') as f:
                    lines = f.readlines()
                    
                i = 0
                while i < len(lines):
                    # Skip empty lines
                    if not lines[i].strip():
                        i += 1
                        continue
                    
                    # Get question
                    question = lines[i].strip()
                    i += 1
                    
                    # Skip empty lines
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    
                    # Get answer if available
                    answer = ""
                    if i < len(lines):
                        answer = lines[i].strip()
                        i += 1
                    
                    if question and not question.startswith("http"):  # Filter out website URLs
                        qa_pairs[question] = answer
            
            print(f"Loaded {len(qa_pairs)} QA pairs from knowledge base")
            return qa_pairs
        except Exception as e:
            print(f"Error loading Questions.txt: {str(e)}")
            return {}

    def get_direct_answer(self, query):
        """Check if the query exactly matches a question in the knowledge base"""
        # Direct lookup (case insensitive)
        query_lower = query.lower().strip()
        for question, answer in self.qa_pairs.items():
            if question.lower().strip() == query_lower:
                return answer
        return None

    def get_relevant_knowledge(self, query, top_k=3):
        """Find the most relevant knowledge for the query using semantic search"""
        if not self.qa_pairs:
            return "No knowledge base found."

        # Fallback if encoder is not available
        if self.encoder is None or not self.question_embeddings:
            print("Warning: Using fallback method in KnowledgeRetriever due to missing encoder")
            # Use simple keyword matching as fallback
            keywords = query.lower().split()
            stop_words = {"a", "an", "the", "is", "are", "was", "were", "how", "what", "why", "where", "when", "who", "can", "could", "to", "for", "in", "on", "with", "by", "about"}
            keywords = [k for k in keywords if k not in stop_words and len(k) > 2]
            
            keyword_matches = []
            for question, answer in self.qa_pairs.items():
                question_lower = question.lower()
                if any(keyword in question_lower for keyword in keywords):
                    # Calculate simple match score based on number of matching keywords
                    match_count = sum(1 for kw in keywords if kw in question_lower)
                    keyword_matches.append({
                        "question": question,
                        "answer": answer,
                        "matches": match_count
                    })
            
            if keyword_matches:
                # Sort by number of keyword matches
                keyword_matches.sort(key=lambda x: x["matches"], reverse=True)
                result = ""
                for item in keyword_matches[:top_k]:
                    result += f"Q: {item['question']}\nA: {item['answer']}\n\n"
                return result
                
            return "No relevant knowledge found."

        try:
            query_embedding = self.encoder.encode(query)
            similarities = {
                q: cosine_similarity([query_embedding], [emb])[0][0]
                for q, emb in self.question_embeddings.items()
            }
            sorted_questions = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

            relevant_knowledge = []
            for question, score in sorted_questions[:top_k]:
                if score > 0.65:  # Higher threshold for confidence
                    relevant_knowledge.append({
                        "question": question,
                        "answer": self.qa_pairs[question],
                        "score": float(score)
                    })

            # If we have high confidence matches, format them nicely
            if relevant_knowledge:
                result = ""
                for item in relevant_knowledge:
                    result += f"Q: {item['question']}\nA: {item['answer']}\n\n"
                return result
            
            # Check for keyword matches if no semantic matches found
            keywords = query.lower().split()
            stop_words = {"a", "an", "the", "is", "are", "was", "were", "how", "what", "why", "where", "when", "who", "can", "could", "to", "for", "in", "on", "with", "by", "about"}
            keywords = [k for k in keywords if k not in stop_words and len(k) > 2]
            
            keyword_matches = []
            for question, answer in self.qa_pairs.items():
                question_lower = question.lower()
                if any(keyword in question_lower for keyword in keywords):
                    match_count = sum(1 for kw in keywords if kw in question_lower)
                    keyword_matches.append({
                        "question": question,
                        "answer": answer,
                        "matches": match_count
                    })
            
            if keyword_matches:
                keyword_matches.sort(key=lambda x: x["matches"], reverse=True)
                result = ""
                for item in keyword_matches[:top_k]:
                    result += f"Q: {item['question']}\nA: {item['answer']}\n\n"
                return result
                
            return "No relevant knowledge found."
        except Exception as e:
            print(f"Error in get_relevant_knowledge: {str(e)}")
            # Use simple keyword matching as fallback (same as above)
            keywords = query.lower().split()
            stop_words = {"a", "an", "the", "is", "are", "was", "were", "how", "what", "why", "where", "when", "who", "can", "could", "to", "for", "in", "on", "with", "by", "about"}
            keywords = [k for k in keywords if k not in stop_words and len(k) > 2]
            
            keyword_matches = []
            for question, answer in self.qa_pairs.items():
                question_lower = question.lower()
                if any(keyword in question_lower for keyword in keywords):
                    match_count = sum(1 for kw in keywords if kw in question_lower)
                    keyword_matches.append({
                        "question": question,
                        "answer": answer,
                        "matches": match_count
                    })
            
            if keyword_matches:
                keyword_matches.sort(key=lambda x: x["matches"], reverse=True)
                result = ""
                for item in keyword_matches[:top_k]:
                    result += f"Q: {item['question']}\nA: {item['answer']}\n\n"
                return result
                
            return "No relevant knowledge found."


class EmailResponseRetriever:
    def __init__(self):
        self.encoder = None
        try:
            self.encoder = load_sentence_transformer()
            print("Successfully loaded SentenceTransformer for EmailResponseRetriever")
        except Exception as e:
            print(f"Error loading SentenceTransformer model in EmailResponseRetriever: {str(e)}")
        # Sample email response examples with professional yet approachable tone for Toys"R"Us Canada
        self.examples = {
"store_location": """
ORIGINAL EMAIL:
Hello, I need to find the nearest Toys"R"Us store. How can I locate one?
MY RESPONSE:
Hi there! You can easily find your nearest Toys"R"Us Canada store using our "Store Locator" feature on our website. Simply enter your city or postal code, and you'll get a list of nearby stores with their addresses and contact information. Would you like me to help you locate a specific store?
""",
"returns_policy": """
ORIGINAL EMAIL:
I want to return an item I purchased. What's your return policy?
MY RESPONSE:
Hello! We accept returns within 30 days of purchase, provided the item is unused, in its original packaging, and accompanied by the original receipt. You can return items either to our stores or by mail for online purchases. Please note some items may have specific return restrictions. Would you like more details about the return process?
""",
"gift_card_balance": """
ORIGINAL EMAIL:
How do I check my gift card balance?
MY RESPONSE:
Hi! You can check your gift card balance easily by visiting the "My Gift Card" section on our website. You'll need to enter your 16-digit gift card number and 4-digit PIN (found under the scratch-off label on the back of your card). Would you like me to guide you to the specific webpage?
""",
"online_ordering": """
ORIGINAL EMAIL:
I'd like to place an order online. What are the shipping options?
MY RESPONSE:
Hello! We ship to most locations within Canada (excluding some remote areas). Shipping rates are based on weight, dimensions, and destination. We offer free shipping on orders over $75, and you can also choose our "Buy Online, Pick Up In-Store" option. Would you like more information about specific shipping rates?
""",
"loyalty_program": """
ORIGINAL EMAIL:
Can you tell me about your loyalty program?
MY RESPONSE:
Hi! Our "R" Club loyalty program is free to join and offers exclusive deals, promotions, and early access to sales events. Members can earn rewards on purchases through our "R" Cash program. Would you like help signing up for the program?
"""
        }
        # Pre-compute embeddings for examples only if encoder is available
        self.example_embeddings = {}
        if self.encoder is not None:
            try:
                self.example_embeddings = {
                    k: self.encoder.encode(v) for k, v in self.examples.items()
                }
                print(f"Successfully encoded {len(self.example_embeddings)} examples")
            except Exception as e:
                print(f"Error encoding examples: {str(e)}")

    def get_relevant_response(self, query, top_k=2):
        # Fallback if encoder is not available
        if self.encoder is None or not self.example_embeddings:
            print("Warning: Using fallback method in EmailResponseRetriever due to missing encoder")
            # Return all examples as a fallback
            return "\n\n".join(self.examples.values())
            
        try:
            query_embedding = self.encoder.encode(query)
            similarities = {
                k: cosine_similarity([query_embedding], [emb])[0][0]
                for k, emb in self.example_embeddings.items()
            }
            sorted_examples = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

            relevant_examples = []
            for example_name, score in sorted_examples[:top_k]:
                if score > 0.3:  # Similarity threshold
                    relevant_examples.append(self.examples[example_name])

            return (
                "\n\n".join(relevant_examples)
                if relevant_examples
                else "No relevant example found."
            )
        except Exception as e:
            print(f"Error in get_relevant_response: {str(e)}")
            # Fallback to returning a subset of examples
            sample_examples = list(self.examples.values())[:2]
            return "\n\n".join(sample_examples)


class PolicyRetriever:
    def __init__(self):
        self.encoder = None
        try:
            self.encoder = load_sentence_transformer()
            print("Successfully loaded SentenceTransformer for PolicyRetriever")
        except Exception as e:
            print(f"Error loading SentenceTransformer model in PolicyRetriever: {str(e)}")
        # Toys"R"Us Canada policies based on Questions.txt
        self.policies = {
"returns_refunds": """
Returns and Refunds Policy:
- Returns accepted within 30 days of purchase
- Items must be unused, in original packaging, and in resalable condition
- Original receipt required
- Some items excluded (opened DVDs, video games, breast pumps)
- Refunds processed in 7-10 business days for warehouse returns
- Store returns processed immediately
""",
"shipping_delivery": """
Shipping and Delivery Policy:
- Ships to most locations within Canada (some remote areas excluded)
- Rates based on weight, dimensions, and destination
- Free shipping on orders over $75
- Buy Online, Pick Up In-Store option available
- Oversized items may require freight trucking (7-14 business days)
- Multiple shipping carriers used for cost-effectiveness
""",
"gift_cards": """
Gift Card Policy:
- Gift cards can be used online and in-store
- Maximum of three gift cards per online order
- Cannot be redeemed for cash or refunded
- Cannot be used to purchase other gift cards
- Balance checkable online with card number and PIN
- E-gift cards typically delivered within 1-24 hours
""",
"loyalty_rewards": """
"R" Club Loyalty Program:
- Free membership program
- Exclusive deals and promotions
- Early access to sales events
- "R" Cash rewards on purchases
- Special member events and activities
- Online account management available
""",
        }
        # Pre-compute embeddings for policies only if encoder is available
        self.policy_embeddings = {}
        if self.encoder is not None:
            try:
                self.policy_embeddings = {
                    k: self.encoder.encode(v) for k, v in self.policies.items()
                }
                print(f"Successfully encoded {len(self.policy_embeddings)} policies")
            except Exception as e:
                print(f"Error encoding policies: {str(e)}")

    def get_relevant_policy(self, query, top_k=2):
        # Fallback if encoder is not available
        if self.encoder is None or not self.policy_embeddings:
            print("Warning: Using fallback method in PolicyRetriever due to missing encoder")
            # Return all policies as a fallback
            return "\n\n".join(self.policies.values())
            
        try:
            query_embedding = self.encoder.encode(query)
            similarities = {
                k: cosine_similarity([query_embedding], [emb])[0][0]
                for k, emb in self.policy_embeddings.items()
            }
            sorted_policies = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

            relevant_policies = []
            for policy_name, score in sorted_policies[:top_k]:
                if score > 0.3:  # Similarity threshold
                    relevant_policies.append(self.policies[policy_name])

            return (
                "\n\n".join(relevant_policies)
                if relevant_policies
                else "No relevant policy found."
            )
        except Exception as e:
            print(f"Error in get_relevant_policy: {str(e)}")
            # Fallback to returning a subset of policies
            sample_policies = list(self.policies.values())[:2]
            return "\n\n".join(sample_policies)


class EmailAgent:
    def __init__(self, role, client):
        self.role = role
        self.client = client
        self.response_retriever = EmailResponseRetriever()
        self.policy_retriever = PolicyRetriever()

        self.prompts = {
            "analyzer": """SYSTEM: You are an expert email analyzer for Toys"R"Us Canada.
            Your role is to break down customer emails into key components and provide clear, actionable insights.

            INSTRUCTIONS:
            • Extract main topics and key points from the email
            • Determine urgency level (Low, Medium, High)
            • List all required actions in bullet points
            • Analyze tone of the message (formal, informal, urgent, etc.)
            • Consider similar past responses
            • Highlight any customer service priorities
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            SIMILAR PAST RESPONSES:
            {examples}

            RELEVANT POLICIES:
            {policies}

            Email: {content}""",
            "drafter": """SYSTEM: You are a professional customer service specialist for Toys"R"Us Canada.
            Draft responses that align with our past successful responses while maintaining our friendly, helpful tone.

            INSTRUCTIONS:
            • Address all key points from the original email
            • Use a professional, warm tone like in our examples
            • Ensure alignment with store policies
            • Include clear next steps and action items
            • Reference specific Toys"R"Us resources or processes when appropriate
            • Add necessary product or service information
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            SIMILAR PAST RESPONSES:
            {examples}

            RELEVANT POLICIES:
            {policies}

            Based on this analysis: {content}""",
            "reviewer": """SYSTEM: You are a senior customer service quality specialist for Toys"R"Us Canada.
            Ensure responses meet our customer service standards.

            INSTRUCTIONS:
            • Verify alignment with example responses
            • Check for policy accuracy
            • Assess friendly and helpful tone
            • Review completeness of response
            • Evaluate appropriate handling of customer inquiries
            • Confirm all action items are clearly stated
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            SIMILAR PAST RESPONSES:
            {examples}

            RELEVANT POLICIES:
            {policies}

            Evaluate this draft response: {content}""",
            "sentiment": """SYSTEM: You are an expert in analyzing customer sentiment and emotional context in
            retail customer service communications.

            INSTRUCTIONS:
            • Analyze overall sentiment (positive, negative, neutral)
            • Identify emotional undertones
            • Detect urgency or stress indicators
            • Assess customer satisfaction level
            • Flag any concerning language
            • Recommend tone adjustments if needed
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            Email: {content}""",
            "example_justifier": """SYSTEM: You are an example matching expert for Toys"R"Us Canada. In 2 lines, explain why the following example responses are relevant
            to this email content. Be specific and concise.

            Email content: {content}
            Selected examples: {examples}""",
            "policy_justifier": """SYSTEM: You are a policy expert for Toys"R"Us Canada. In 2 lines, explain why the following policies are relevant
            to this email content. Be specific and concise.

            Email content: {content}
            Selected policies: {policies}"""
        }

    def process(self, content):
        # Get relevant example responses for the email content
        relevant_examples = self.response_retriever.get_relevant_response(content)
        relevant_policies = self.policy_retriever.get_relevant_policy(content)
        return prompt_llm(
            self.prompts[self.role].format(content=content, examples=relevant_examples, policies=relevant_policies),
            self.client
        )


class EmailProcessingSystem:
    def __init__(self, client):
        self.analyzer = EmailAgent("analyzer", client)
        self.drafter = EmailAgent("drafter", client)
        self.reviewer = EmailAgent("reviewer", client)
        self.example_justifier = EmailAgent("example_justifier", client)
        self.policy_justifier = EmailAgent("policy_justifier", client)
        # Initialize the knowledge retriever
        self.knowledge_retriever = KnowledgeRetriever()

    def process_email(self, email_content):
        # First check if we can directly answer from the knowledge base
        direct_answer = self.knowledge_retriever.get_direct_answer(email_content)
        relevant_knowledge = self.knowledge_retriever.get_relevant_knowledge(email_content) if not direct_answer else ""
        
        # If we have a direct answer or relevant knowledge, use it
        has_knowledge_answer = direct_answer or (relevant_knowledge and relevant_knowledge != "No relevant knowledge found.")
        
        # Step 1: Analyze email content
        print("\nAnalyzing email content...")
        analysis = self.analyzer.process(email_content)

        # Step 2: Analyze sentiment
        sentiment = prompt_llm(
            self.analyzer.prompts["sentiment"].format(content=email_content),
            self.analyzer.client
        )

        # Step 3: Draft response - use knowledge base if available
        print("\nDrafting response based on analysis...")
        if has_knowledge_answer:
            # Create a prompt for drafting a response using the knowledge base
            knowledge_content = direct_answer if direct_answer else relevant_knowledge
            
            # Create a more concise prompt for drafting based on knowledge
            draft_prompt = f"""SYSTEM: You are the Toys"R"Us Canada email assistant. Draft a complete, professional response to the following email using the knowledge provided. Include greeting and closing.

            ORIGINAL EMAIL:
            {email_content}

            KNOWLEDGE BASE INFORMATION:
            {knowledge_content}
            
            Make sure the response is complete with a clear beginning and end."""
            
            # Generate the draft using the LLM with increased token limit
            draft = prompt_llm(draft_prompt, self.analyzer.client, max_tokens=600)
        else:
            # If no knowledge is available, return the standard message
            draft = "Toys\"R\"Us Canada Email Assistant is unable to locate the relevant information. Please update the knowledge base to ensure data retrieval."

        # Get relevant policies and example responses for display
        relevant_policies = self.analyzer.policy_retriever.get_relevant_policy(email_content)
        relevant_examples = self.analyzer.response_retriever.get_relevant_response(email_content)

        # Add policy justification
        policy_justification = self.policy_justifier.process(
            f"Email: {email_content}\nPolicies: {relevant_policies}"
        )

        # Add example justification
        example_justification = self.example_justifier.process(
            f"Email: {email_content}\nExamples: {relevant_examples}"
        )

        # Step 4: Review response
        review = self.reviewer.process(draft)

        return {
            "status": "success",
            "analysis": analysis,
            "final_draft": draft,
            "review": review,
            "policies": relevant_policies,
            "examples": relevant_examples,
            "policy_justification": policy_justification,
            "example_justification": example_justification,
            "sentiment": sentiment,
            "knowledge_source": "Direct match" if direct_answer else ("Semantic search" if has_knowledge_answer else "Not found in knowledge base")
        }


# Sample emails for testing - updated for Toys"R"Us Canada context
sample_emails = [
    """Hello, I'm looking for the nearest Toys"R"Us store location. 
    Can you help me find one in my area? Thanks.""",
    
    """Hi, I purchased a toy last week and need to return it.
    What's your return policy and what do I need to bring?""",
    
    """I have a question about my gift card balance.
    How can I check how much money is left on it?""",
    
    """I'm trying to place an order online but having issues.
    What are the shipping options and costs?""",
    
    """Can you tell me about your loyalty program?
    I heard there's a way to earn rewards when shopping.""",
    
    """Hello, I received a damaged item in my online order.
    What's the process for getting it replaced?""",
    
    """I'm interested in creating a baby registry.
    How do I set one up and what are the benefits?""",
    
    """What's the current status of my online order?
    I haven't received any shipping updates.""",
    
    """Do you price match with other retailers?
    I found the same toy cheaper somewhere else.""",
    
    """I'm having trouble accessing my "R" Club account.
    How can I reset my password?""",
]


# Flask application
app = Flask(__name__)
app.secret_key = 'crna_email_system_secret_key'  # For session management


@app.route('/')
def index():
    return render_template('index.html', emails=sample_emails)


@app.route('/process', methods=['POST'])
def process():
    email_content = request.form.get('email')
    
    try:
        # Initialize Together client with API key from config
        client = Together(api_key=API_KEY)
        
        email_system = EmailProcessingSystem(client)
        result = email_system.process_email(email_content)
        
        # Store stats in session
        if 'approved_count' not in session:
            session['approved_count'] = 0
            session['disapproved_count'] = 0
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing email: {str(e)}"
        })


@app.route('/approve', methods=['POST'])
def approve():
    session['approved_count'] = session.get('approved_count', 0) + 1
    return jsonify({
        "status": "success", 
        "approved_count": session['approved_count'],
        "disapproved_count": session['disapproved_count']
    })


@app.route('/disapprove', methods=['POST'])
def disapprove():
    session['disapproved_count'] = session.get('disapproved_count', 0) + 1
    return jsonify({
        "status": "success", 
        "approved_count": session['approved_count'],
        "disapproved_count": session['disapproved_count']
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "approved_count": session.get('approved_count', 0),
        "disapproved_count": session.get('disapproved_count', 0)
    })


if __name__ == "__main__":
    # Use environment variables for port if available (needed for Hugging Face)
    port = int(os.environ.get('PORT', 5005))
    # Bind to 0.0.0.0 instead of 127.0.0.1 to be accessible from outside the container
    app.run(debug=False, host='0.0.0.0', port=port)

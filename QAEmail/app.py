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
# Get API tokens from config
API_KEY = config.get('together_ai_token')
HGToken = config.get('hf_token')

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
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=HGToken)
        self.qa_pairs = self._load_qa_pairs()
        # Precompute embeddings for questions to enable semantic search
        self.question_embeddings = {
            q: self.encoder.encode(q) for q, a in self.qa_pairs.items()
        }
        
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
        # Filter out common words
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

class EmailResponseRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=HGToken)
        # Sample email response examples with professional yet approachable tone for CRNA
        self.examples = {
"permit_verification": """
ORIGINAL EMAIL:
Hello, I need to verify if a nurse in our facility has a valid permit. How can I do this?

MY RESPONSE:
Hi there! You can easily verify a nurse's permit through our "Verify a Permit" feature on the CRNA website (nurses.ab.ca). It's quick and provides up-to-date information on registration status. Is there anything specific about the verification process you need help with?
""",
"registration_requirements": """
ORIGINAL EMAIL:
I'm interested in getting registered in Alberta. What are the requirements?

MY RESPONSE:
Hello! To register as a nurse in Alberta, you'll need to meet several core requirements including:
- Completion of registration exam
- Verified post-secondary education
- English language proficiency
- Currency of practice
- Continuing competence documentation
- Liability insurance

Would you like me to send you the detailed application pathway information?
""",
"internationally_educated": """
ORIGINAL EMAIL:
I completed my nursing education in the Philippines. How do I apply for a permit in Alberta?

MY RESPONSE:
Welcome! As an internationally educated nurse, you can start by creating an account in our online registrant portal, College Connect. Our pathway assessment tool will guide you through the specific requirements for your situation.

We've recently streamlined the process with updates like accepting multiple credential review services and recognizing NCLEX-RN exam results as evidence of education.

Would you like the direct link to get started?
""",
"continuing_competence": """
ORIGINAL EMAIL:
I need information about maintaining my nursing competence. What does CRNA require?

MY RESPONSE:
Hi there! The Continuing Competence Program outlines how to maintain and enhance your professional skills throughout your career. The program includes:
- Regular self-assessment
- Learning plan development
- Implementation of learning activities

You can access all resources through your College Connect account. Can I help you with a specific aspect of the program?
""",
        }
        # Pre-compute embeddings for examples
        self.example_embeddings = {
            k: self.encoder.encode(v) for k, v in self.examples.items()
        }

    def get_relevant_response(self, query, top_k=2):
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

class PolicyRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=HGToken)
        # CRNA policies based on Questions.txt
        self.policies = {
"registration": """
Registration Policy:
- Core registration requirements include demographic information, registration exam, and post-secondary education
- English language proficiency must be demonstrated through accepted tests or approved methods
- Currency of practice and continuing competence documentation required
- Good character, fitness to practice, jurisprudence completion, and liability insurance needed
""",
"standards": """
Professional Standards Policy:
- The CRNA standards outline minimum expectations for registered nurses and nurse practitioners
- Documentation Standards provide requirements for clear, accurate, and comprehensive client care records
- Medication Management Standards define safe medication practices requirements
- Advertising Standards ensure transparency and accuracy when promoting nursing services
""",
"complaint_process": """
Complaint Process Policy:
- Complaints can be submitted through the "Submit a Complaint" feature on the CRNA website
- Process includes submission, investigation, and resolution
- Support services are available for victims of sexual abuse and misconduct
- The CRNA is committed to transparency throughout the complaint process
""",
"education_programs": """
Education Program Policy:
- CRNA approves nursing education programs in Alberta for RN and NP practice
- The Nursing Education Program Approval Framework outlines expectations for post-secondary institutions
- Entry-Level Competencies define required knowledge, skills, and judgment for new practitioners
- CRNA provides resources and guidelines to nursing educators to ensure regulatory alignment
""",
        }
        # Pre-compute embeddings for policies
        self.policy_embeddings = {
            k: self.encoder.encode(v) for k, v in self.policies.items()
        }

    def get_relevant_policy(self, query, top_k=2):
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

class EmailAgent:
    def __init__(self, role, client):
        self.role = role
        self.client = client
        self.response_retriever = EmailResponseRetriever()
        self.policy_retriever = PolicyRetriever()

        self.prompts = {
            "analyzer": """SYSTEM: You are an expert email analyzer for the College of Registered Nurses of Alberta (CRNA).
            Your role is to break down emails into key components and provide clear, actionable insights.

            INSTRUCTIONS:
            • Extract main topics and key points from the email
            • Determine urgency level (Low, Medium, High)
            • List all required actions in bullet points
            • Analyze tone of the message (formal, informal, urgent, etc.)
            • Consider similar past responses
            • Highlight any regulatory compliance concerns
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            SIMILAR PAST RESPONSES:
            {examples}

            RELEVANT POLICIES:
            {policies}

            Email: {content}""",
            "drafter": """SYSTEM: You are a professional email response specialist for the College of Registered Nurses of Alberta (CRNA).
            Draft responses that align with our past successful responses while maintaining regulatory compliance.

            INSTRUCTIONS:
            • Address all key points from the original email
            • Use a professional, helpful tone like in our examples
            • Ensure compliance with nursing regulatory standards
            • Include clear next steps and action items
            • Reference specific CRNA resources or processes when appropriate
            • Add necessary disclaimers where applicable
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            SIMILAR PAST RESPONSES:
            {examples}

            RELEVANT POLICIES:
            {policies}

            Based on this analysis: {content}""",
            "reviewer": """SYSTEM: You are a senior email quality assurance specialist for the College of Registered Nurses of Alberta (CRNA).
            Ensure responses meet professional regulatory communication standards.

            INSTRUCTIONS:
            • Verify alignment with example responses
            • Check for regulatory compliance
            • Assess professional and helpful tone
            • Review completeness of response
            • Evaluate appropriate handling of sensitive information
            • Confirm all action items are clearly stated
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            SIMILAR PAST RESPONSES:
            {examples}

            RELEVANT POLICIES:
            {policies}

            Evaluate this draft response: {content}""",
            "sentiment": """SYSTEM: You are an expert in analyzing email sentiment and emotional context in
            nursing regulatory communications.

            INSTRUCTIONS:
            • Analyze overall sentiment (positive, negative, neutral)
            • Identify emotional undertones
            • Detect urgency or stress indicators
            • Assess sender satisfaction or concern level
            • Flag any concerning language
            • Recommend tone adjustments if needed
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            Email: {content}""",
            "example_justifier": """SYSTEM: You are an example matching expert. In 2 lines, explain why the following example responses are relevant
            to this email content. Be specific and concise.

            Email content: {content}
            Selected examples: {examples}""",
            "policy_justifier": """SYSTEM: You are a policy expert. In 2 lines, explain why the following CRNA policies are relevant
            to this email content. Be specific and concise.

            Email content: {content}
            Selected policies: {policies}""",
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
            draft_prompt = f"""SYSTEM: You are the CRNA email assistant. Draft a complete, professional response to the following email using the knowledge provided. Include greeting and closing.

            ORIGINAL EMAIL:
            {email_content}

            KNOWLEDGE BASE INFORMATION:
            {knowledge_content}
            
            Make sure the response is complete with a clear beginning and end."""
            
            # Generate the draft using the LLM with increased token limit
            draft = prompt_llm(draft_prompt, self.analyzer.client, max_tokens=600)
        else:
            # If no knowledge is available, return the standard message
            draft = "CRNA Email Assistant is unable to locate the relevant information. Please update the knowledge base to ensure data retrieval."

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


# Sample emails for testing - updated for CRNA context
sample_emails = [
    """Hello, I need to verify if a nurse at our facility has a valid permit. 
    What's the best way to check this information? Thank you.""",
    
    """Hi, I'm an internationally educated nurse who recently moved to Alberta.
    I'd like to know how to apply for a nursing permit and what documentation I'll need.""",
    
    """I have questions about the Continuing Competence Program requirements.
    Where can I find resources to help me complete the necessary documentation?""",
    
    """I'm considering becoming a Nurse Practitioner in Alberta.
    Can you tell me more about the Nurse Practitioner Primary Care Program (NPPCP)?""",
    
    """I need to submit a complaint regarding unprofessional conduct by a nurse.
    What is the proper process to follow, and what information should I include?""",
    
    """I'm a nursing educator at a post-secondary institution in Alberta. 
    We're developing a new nursing program and would like to know the CRNA approval process.""",
    
    """Hello, I'm a registered nurse and need to renew my permit soon. 
    Could you please explain the renewal process and what I need to prepare?""",
    
    """I'm looking for resources on medication management standards in Alberta. 
    Where can I find the CRNA's guidelines on this topic?""",
    
    """Good morning, I'm interested in learning more about the CRNA's stance on diversity, equity, and inclusion. 
    Do you have any resources or initiatives related to this area?""",
    
    """I'm having technical issues accessing my College Connect account. 
    Could you please advise on how to resolve login problems or reset my password?""",
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
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        print("Check that tokens are properly set in config.json")

import warnings
import os
import json
import requests

warnings.filterwarnings("ignore")

# Create a compatibility layer for different versions of the Together API
try:
    # Try importing the newer version first
    from together import Together
    TOGETHER_API_STYLE = "new"
except ImportError:
    try:
        # Fall back to the older version
        import together
        TOGETHER_API_STYLE = "old"
    except ImportError:
        # If neither works, we'll create a minimal fallback implementation
        TOGETHER_API_STYLE = "fallback"
        print("WARNING: together package not found, using fallback implementation")


class TogetherAPIWrapper:
    """Wrapper class to handle different versions of the Together API."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        
        # If no API key was provided, try to get it from environment variable
        if not self.api_key:
            self.api_key = os.environ.get("TOGETHER_API_KEY")
            
        # If still no API key, try to load from config.json
        if not self.api_key:
            try:
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as config_file:
                        config = json.load(config_file)
                        self.api_key = config.get("together_ai_token")
                        if self.api_key:
                            print("Successfully loaded API key from config.json")
            except Exception as e:
                print(f"Error loading API key from config.json: {str(e)}")
                
        # If we still don't have an API key, log a warning
        if not self.api_key:
            print("WARNING: No Together API key found in environment variables or config.json")
        
        if TOGETHER_API_STYLE == "new":
            self.client = Together(api_key=self.api_key) if self.api_key else None
        elif TOGETHER_API_STYLE == "old":
            together.api_key = self.api_key
            self.client = together
        else:
            # Fallback implementation using direct API calls
            self.client = None
    
    def chat_completion(self, model, messages, temperature=0.7, max_tokens=1024):
        """Handle chat completion based on available API version."""
        if not self.api_key:
            return "ERROR: No Together API key found. Please set TOGETHER_API_KEY environment variable or add it to config.json."
            
        if TOGETHER_API_STYLE == "new":
            if self.client:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                return "ERROR: Together client initialization failed."
        elif TOGETHER_API_STYLE == "old":
            response = together.Complete.create(
                prompt=self._convert_messages_to_prompt(messages),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['output']['choices'][0]['text']
        else:
            # Fallback implementation using direct API calls
            return self._fallback_api_call(model, messages, temperature, max_tokens)
    
    def _convert_messages_to_prompt(self, messages):
        """Convert chat messages to a prompt string for older API versions."""
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
        return prompt
    
    def _fallback_api_call(self, model, messages, temperature, max_tokens):
        """Direct API call as fallback."""
        if not self.api_key:
            return "ERROR: No Together API key provided. Please set TOGETHER_API_KEY environment variable."
            
        # Convert messages to the format expected by the API
        api_messages = []
        for msg in messages:
            if msg.get("role") and msg.get("content"):
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make the API call
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"API Error ({response.status_code}): {response.text}"
        except Exception as e:
            return f"Error calling Together API: {str(e)}"


# Initialize the API wrapper
together_api = TogetherAPIWrapper()


class SummarizerAgent:
    """Agent for summarizing code and documentation."""
    
    def __init__(self, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model = model
    
    def process(self, content):
        """Summarize the given content."""
        try:
            # Skip empty content
            if not content or len(content.strip()) < 10:
                return "Content too short to summarize."
                
            messages = [
                {"role": "system", "content": "You are a technical documentation specialist. Summarize the following code or documentation in a clear, concise manner. Focus on the main functionality, purpose, and key components."},
                {"role": "user", "content": f"Please summarize the following content:\n\n{content}"}
            ]
            
            return together_api.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
        except Exception as e:
            return f"Error generating summary: {str(e)}"


class InsightAgent:
    """Agent for generating insights from code summaries."""
    
    def __init__(self, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model = model
    
    def process_text(self, summaries):
        """Generate insights from a list of summaries."""
        try:
            if not summaries:
                return "No summaries provided for insight generation."
                
            # Combine summaries into a single text
            combined_summary = "\n\n".join(summaries)
            
            messages = [
                {"role": "system", "content": "You are a software architecture expert. Analyze the following code summaries and provide key insights about the overall system, architecture patterns, potential issues, and recommendations."},
                {"role": "user", "content": f"Based on these summaries of a codebase, what insights can you provide?\n\n{combined_summary}"}
            ]
            
            return together_api.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
        except Exception as e:
            return f"Error generating insights: {str(e)}"


class RecommenderAgent:
    """Agent for recommending next steps based on insights."""
    
    def __init__(self, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model = model
    
    def process(self, insights, summaries, user_goal, persona):
        """Generate recommendations based on insights and user context."""
        try:
            messages = [
                {"role": "system", "content": f"You are a technical advisor for a {persona}. Provide actionable recommendations based on code analysis and insights."},
                {"role": "user", "content": f"Goal: {user_goal}\n\nInsights about the codebase:\n{insights}\n\nBased on this information, what specific recommendations would you make?"}
            ]
            
            return together_api.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def suggest_next_query(self, insights, summaries, user_goal, persona):
        """Suggest the next query or area to explore."""
        try:
            messages = [
                {"role": "system", "content": "You are a technical researcher. Suggest the next area to explore or query to make based on current findings."},
                {"role": "user", "content": f"Based on these insights about a codebase:\n{insights}\n\nWhat would be the most valuable next area to explore for a {persona} with the goal: {user_goal}?"}
            ]
            
            return together_api.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=256
            )
        except Exception as e:
            return f"Error suggesting next query: {str(e)}"


class QuestionGeneratorAgent:
    """Agent for generating relevant questions about a codebase or resource."""
    
    def __init__(self, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model = model
    
    def generate_questions(self, content, category, source):
        """Generate interesting questions about the content."""
        try:
            messages = [
                {"role": "system", "content": "You are a technical interviewer. Generate 3-5 insightful questions that would help someone understand this resource better."},
                {"role": "user", "content": f"Please generate 3-5 insightful questions about this {category} from {source}:\n{content}"}
            ]
            
            return together_api.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
        except Exception as e:
            return f"Error generating questions: {str(e)}"


class CLISetupAgent:
    """Agent for generating CLI setup instructions for repositories."""
    
    def __init__(self, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model = model
    
    def generate_setup_instructions(self, repo_content, repo_metadata):
        """Generate CLI setup instructions based on repository content."""
        try:
            # Extract key files for setup instructions
            setup_files = {}
            for filename, content in repo_content.items():
                if any(filename.endswith(ext) for ext in [
                    'requirements.txt', 'setup.py', 'package.json', 'Dockerfile',
                    'docker-compose.yml', 'Makefile', 'README.md', '.env.example'
                ]):
                    setup_files[filename] = content
            
            file_list = "\n".join([f"- {filename}" for filename in repo_content.keys()])
            setup_file_content = "\n\n".join([f"### {filename}\n```\n{content}\n```" for filename, content in setup_files.items()])
            
            messages = [
                {"role": "system", "content": "You are a DevOps specialist. Generate step-by-step CLI instructions for setting up and running this repository locally."},
                {"role": "user", "content": f"Repository: {repo_metadata.get('name', 'Unknown')}\nDescription: {repo_metadata.get('description', 'No description')}\n\nFiles in repository:\n{file_list}\n\nKey setup files:\n{setup_file_content}\n\nProvide detailed CLI instructions for setting up and running this repository locally."}
            ]
            
            return together_api.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
        except Exception as e:
            return f"Error generating setup instructions: {str(e)}"


class WebsiteContentSummarizer:
    """Agent for summarizing website content."""
    
    def __init__(self, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model = model
    
    def summarize_page(self, url, content):
        """Summarize a single webpage's content."""
        try:
            if not content or len(content.strip()) < 50:
                return "Page has insufficient content to summarize."
                
            # Limit content length to avoid token limits
            if len(content) > 10000:
                content = content[:10000] + "... (content truncated)"
                
            messages = [
                {"role": "system", "content": "You are a web content analyst. Summarize the key information from this webpage in a concise manner."},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{content}\n\nPlease provide a concise summary of this webpage:"}
            ]
            
            return together_api.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
        except Exception as e:
            return f"Error summarizing page: {str(e)}"


class QuestionAnswerer:
    """Agent for answering questions based on website content."""
    
    def __init__(self, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model = model
    
    def answer_question(self, question, content, source_urls):
        """Answer a question based on website content."""
        try:
            if not content:
                return "No relevant content found to answer this question."
                
            # Combine content sections and format sources
            combined_content = "\n\n".join(content)
            sources_text = "\n".join([f"- {url}" for url in source_urls[:3]])
            
            messages = [
                {"role": "system", "content": "You are a web research assistant. Answer the user's question based on the provided website content. Be concise, accurate, and cite your sources when possible."},
                {"role": "user", "content": f"Question: {question}\n\nRelevant Website Content:\n{combined_content}\n\nSources:\n{sources_text}\n\nPlease answer the question based only on this information:"}
            ]
            
            return together_api.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
        except Exception as e:
            return f"Error answering question: {str(e)}"


class WebsiteSummarizer:
    """Agent for creating overall website summaries."""
    
    def __init__(self, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model = model
    
    def create_website_overview(self, pages_content):
        """Create an overview of an entire website."""
        try:
            # Get a list of URLs
            urls = list(pages_content.keys())
            
            # Get a sample of page content (first 5 pages)
            sample_content = []
            for url in urls[:5]:
                content = pages_content[url]
                if len(content) > 1000:
                    content = content[:1000] + "... (truncated)"
                sample_content.append(f"URL: {url}\n\n{content}")
            
            combined_sample = "\n\n---\n\n".join(sample_content)
            
            messages = [
                {"role": "system", "content": "You are a web analyst. Create a comprehensive overview of this website based on the sample pages provided."},
                {"role": "user", "content": f"This website has {len(urls)} pages. Here are samples from the first few pages:\n\n{combined_sample}\n\nPlease provide a comprehensive overview of what this website is about:"}
            ]
            
            return together_api.chat_completion(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
        except Exception as e:
            return f"Error creating website overview: {str(e)}"

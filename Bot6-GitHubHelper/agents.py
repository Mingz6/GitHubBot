import warnings
import os
import json

warnings.filterwarnings("ignore")
from together import Together

# Load configuration from config.json or environment variables
def load_config():
    # Try to get API key from environment variables (for Hugging Face deployment)
    # Check both possible environment variable names
    together_api_key = os.environ.get("TOGETHER_API_KEY", "")
    
    # If not found with TOGETHER_API_KEY, try together_ai_token (the name used in your HF secret)
    if not together_api_key:
        together_api_key = os.environ.get("together_ai_token", "")
    
    # If still not found in environment, try to load from config.json (for local development)
    if not together_api_key:
        try:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    together_api_key = config.get("together_ai_token", "")
                    model_name = config.get("model", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
                    return {"together_ai_token": together_api_key, "model": model_name}
        except Exception as e:
            print(f"Error loading config.json: {str(e)}")
    
    # Print debug information
    print(f"API key found: {'Yes' if together_api_key else 'No'}")
    
    # Return a config dictionary with the API key from environment variable
    return {
        "together_ai_token": together_api_key,
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Using the model from your config.json
    }

# Get API key and model from config
config = load_config()
your_api_key = config["together_ai_token"]
model = config.get("model", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")  # Use default if not in config

# Initialize client only if we have an API key
client = Together(api_key=your_api_key) if your_api_key else None

def prompt_llm(prompt, show_cost=False):
    # This function allows us to prompt an LLM via the Together API
    if not client:
        return "Error: Together API client not initialized. Please check your API key."

    # Calculate the number of tokens
    tokens = len(prompt.split())

    # Calculate and print estimated cost for each model
    if show_cost:
        print(f"\nNumber of tokens: {tokens}")
        cost = (0.1 / 1_000_000) * tokens
        print(f"Estimated cost for {model}: ${cost:.10f}\n")

    # Make the API call
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Together API: {str(e)}"


class SummarizerAgent:
    def __init__(self):
        self.client = client

    def process(self, content):
        prompt = """SYSTEM: You are an expert code summarizer. 
        Your task is to condense the provided code into a clear, informative summary of exactly 4 lines.

        INSTRUCTIONS:
        • Identify and include only the most important functionality
        • Explain what the code does and its purpose
        • Ensure the summary is exactly 4 lines long
        • Use concise, clear language
        • Show output only - provide just the summary
        • Do not include any other text or comments, show only the summary
        * do not say "Here is a 4-line summary: " show the summary directly and nothing else
        
        Code to summarize: {content}
        
        Provide a 4-line summary:"""

        return prompt_llm(prompt.format(content=content))


class InsightAgent:
    def __init__(self):
        self.client = client

    def process_text(self, summaries):
        # Process a list of summary texts directly
        all_summaries = "\n\n".join(summaries)
        return self._generate_insights(all_summaries)

    def _generate_insights(self, all_summaries):
        prompt = """SYSTEM: You are an expert code analyst who can identify key insights from code summaries.

        INSTRUCTIONS:
        • Review the provided code summaries
        • Identify 3 key insights that represent the most important takeaways
        • Consider code structure, patterns, and best practices
        • Format your response as exactly 3 bullet points
        • Each bullet point must be a single sentence
        • Be concise, clear, and informative
        • Do not include any introductory or concluding text
        • Show only the 3 bullet points, nothing else
        
        Summaries to analyze:
        {summaries}
        
        Provide exactly 3 bullet point insights:"""

        return prompt_llm(prompt.format(summaries=all_summaries))


class RecommenderAgent:
    def __init__(self):
        self.client = client

    def process(self, insights, summaries, user_goal, persona=""):
        prompt = """SYSTEM: You are an expert code consultant who provides actionable recommendations.

        INSTRUCTIONS:
        • Review the provided insights and summaries about the code
        • Consider the user's specific goal: {user_goal}
        • Consider the user's persona: {persona}
        • Recommend exactly 2 specific, actionable steps the user can take to improve or work with this codebase
        • Each recommendation should be practical, specific, and directly related to the goal
        • Format your response as exactly 2 bullet points
        • Each bullet point should be 1-2 sentences
        • Be concise, clear, and actionable
        • Do not include any introductory or concluding text
        • Show only the 2 bullet point recommendations, nothing else
        
        Insights:
        {insights}
        
        Additional context from summaries:
        {summaries}
        
        User's goal: {user_goal}
        User's persona: {persona}
        
        Provide exactly 2 actionable recommendations:"""

        return prompt_llm(
            prompt.format(
                insights=insights,
                summaries="\n\n".join(summaries),
                user_goal=user_goal,
                persona=persona if persona else "General user",
            )
        )

    def suggest_next_query(self, insights, summaries, user_goal, persona=""):
        """Generate a suggested next search query based on insights, summaries, and user goal."""
        prompt = f"""
        Based on the following insights and summaries about code, and considering the user's goal and persona,
        suggest ONE specific area of the codebase the user should explore next.
        
        INSIGHTS:
        {insights}
        
        SUMMARIES:
        {summaries}
        
        USER'S GOAL:
        {user_goal}
        
        USER'S PERSONA:
        {persona if persona else "General user"}
        
        Suggest a specific, focused area or component (5-10 words) that would help the user find additional 
        information to achieve their goal. This should guide their next exploration of the repository.
        
        NEXT EXPLORATION AREA:
        """

        return prompt_llm(prompt)


class QuestionGeneratorAgent:
    def __init__(self):
        self.client = client
        
    def generate_questions(self, content, category, source):
        prompt_template_path = os.path.join(os.path.dirname(__file__), "PromptTemplate.json")
        with open(prompt_template_path, "r") as f:
            import json
            prompt_data = json.load(f)
            template = prompt_data["prompt_template"]
            
        prompt = template.format(
            content=content,
            category=category,
            source=source
        )
        
        return prompt_llm(prompt)


class CLISetupAgent:
    def __init__(self):
        self.client = client
        
    def generate_setup_instructions(self, repo_content, repo_metadata):
        """Generate step-by-step CLI instructions to set up the environment for a repository."""
        language = repo_metadata.get("language", "")
        repo_name = repo_metadata.get("name", "")
        repo_url = repo_metadata.get("url", "")
        
        # Collect all potential setup files in the repo
        setup_files = {}
        common_setup_files = [
            "requirements.txt", "package.json", "setup.py", "Dockerfile", 
            "docker-compose.yml", ".env.example", "Makefile", "README.md"
        ]
        
        for filename, content in repo_content.items():
            if filename in common_setup_files or filename.endswith((".yml", ".yaml", ".sh", ".bat")):
                setup_files[filename] = content
        
        # Default setup steps if no specific files are found
        default_steps = f"""
1. Clone the repository:
   ```
   git clone {repo_url}
   cd {repo_name}
   ```

2. Check the repository structure:
   ```
   ls -la
   ```

3. Read the README file for specific instructions:
   ```
   cat README.md
   ```
        """
                
        # If we have no setup files, provide basic instructions
        if not setup_files:
            return default_steps
        
        # Create a prompt with all the relevant information
        prompt = f"""
        SYSTEM: You are an expert DevOps engineer who provides clear CLI setup instructions.
        
        INSTRUCTIONS:
        • Generate step-by-step CLI instructions to set up a development environment for the given repository
        • The repository is named "{repo_name}" and primarily uses {language if language else "unknown language"}
        • Include commands for cloning, installing dependencies, and basic configuration
        • Format your response as a numbered list with clear command-line instructions
        • Include comments explaining what each command does
        • Focus on practical, executable commands that work on both macOS/Linux and Windows where possible
        • If different platforms require different commands, clearly indicate which is for which
        • Mention any prerequisites that need to be installed (like Python, Node.js, Docker, etc.)
        
        REPOSITORY INFORMATION:
        Name: {repo_name}
        Primary Language: {language if language else "Not specified"}
        URL: {repo_url}
        
        RELEVANT SETUP FILES:
        {chr(10).join([f"--- {name} ---{chr(10)}{content[:300]}..." for name, content in setup_files.items()])}
        
        Provide a step-by-step CLI setup guide with exactly 5-10 commands:
        """
        
        try:
            result = prompt_llm(prompt)
            # Check if result is empty or invalid
            if not result or len(result.strip()) < 10:
                return default_steps
            return result
        except Exception as e:
            print(f"Error generating CLI setup: {str(e)}")
            return default_steps


class ChatbotAgent:
    """Agent for answering questions about GitHub repositories."""
    
    def __init__(self):
        self.client = client
    
    def answer_question(self, question, repo_content, repo_metadata, summaries=None, insights=None):
        """
        Answer a question about a GitHub repository based on its content and analysis.
        
        Args:
            question: The user's question about the repository
            repo_content: Dictionary of repository files and their content
            repo_metadata: Repository metadata like name, description, etc.
            summaries: Optional dictionary of file summaries
            insights: Optional insights about the repository
            
        Returns:
            A string containing the answer to the question
        """
        # Extract key repository information
        repo_name = repo_metadata.get("name", "Unknown repository")
        repo_description = repo_metadata.get("description", "No description available")
        repo_language = repo_metadata.get("language", "Unknown")
        
        # Create a context from the repository information
        context = f"Repository: {repo_name}\nDescription: {repo_description}\nLanguage: {repo_language}\n\n"
        
        # Add insights if available
        if insights:
            context += f"Key insights:\n{insights}\n\n"
            
        # Add summaries if available
        if summaries:
            context += "File summaries:\n"
            for filename, summary in summaries.items():
                context += f"- {filename}: {summary}\n"
            context += "\n"
            
        # Select relevant files for the question to avoid token limit issues
        relevant_files = self._select_relevant_files(question, repo_content, max_files=5)
        
        # Add content of relevant files
        if relevant_files:
            context += "Relevant files:\n"
            for filename, content in relevant_files.items():
                # Truncate long files
                if len(content) > 1000:
                    context += f"--- {filename} (truncated) ---\n{content[:1000]}...\n\n"
                else:
                    context += f"--- {filename} ---\n{content}\n\n"
        
        # Create the prompt for the LLM
        prompt = f"""SYSTEM: You are a GitHub repository expert assistant. You provide accurate, helpful answers
        about code repositories based on their content, structure, and analysis. Draw upon the
        provided context to answer the question. If you don't know the answer, say so honestly.
        
        CONTEXT INFORMATION:
        {context}
        
        USER QUESTION:
        {question}
        
        Provide a clear, concise answer to the question based only on the information provided above.
        Include code snippets or commands when relevant. Be specific and informative.
        """
        
        return prompt_llm(prompt)
        
    def _select_relevant_files(self, question, repo_content, max_files=5):
        """Select files from the repository that are most relevant to the question."""
        
        # If there are only a few files, return all of them
        if len(repo_content) <= max_files:
            return repo_content
            
        # For more files, select the most relevant ones based on the question
        relevant_files = {}
        
        # Create a prompt to identify relevant file types for the question
        file_selection_prompt = f"""SYSTEM: You are a code repository expert. Given a question about a repository,
        identify what types of files would be most relevant to answer it.
        
        QUESTION: {question}
        
        Based on this question, list ONLY 3-5 file patterns or extensions that would be most relevant
        for answering it. For example: 'README.md', '.py', 'package.json', 'Dockerfile', etc.
        Just list the patterns, one per line, without any explanation or additional text.
        """
        
        # Get relevant file patterns
        try:
            file_patterns_response = prompt_llm(file_selection_prompt)
            file_patterns = [pattern.strip().lower() for pattern in file_patterns_response.split('\n') if pattern.strip()]
            
            # Filter files based on patterns
            for filename, content in repo_content.items():
                filename_lower = filename.lower()
                
                # Check if file matches any of the patterns
                if any(pattern in filename_lower for pattern in file_patterns):
                    relevant_files[filename] = content
                    
                # Stop when we reach the maximum number of files
                if len(relevant_files) >= max_files:
                    break
            
            # If we didn't find enough files with patterns, add important files
            if len(relevant_files) < max_files:
                important_files = ['readme.md', 'setup.py', 'requirements.txt', 'package.json', 'dockerfile']
                
                for filename, content in repo_content.items():
                    if filename.lower() in important_files and filename not in relevant_files:
                        relevant_files[filename] = content
                        
                    # Stop when we reach the maximum number of files
                    if len(relevant_files) >= max_files:
                        break
                        
            # If we still don't have enough files, add some random ones
            remaining_slots = max_files - len(relevant_files)
            if remaining_slots > 0:
                for filename, content in repo_content.items():
                    if filename not in relevant_files:
                        relevant_files[filename] = content
                        remaining_slots -= 1
                        
                    if remaining_slots <= 0:
                        break
                        
        except Exception as e:
            print(f"Error selecting relevant files: {str(e)}")
            # Fallback: just take the first max_files
            relevant_files = dict(list(repo_content.items())[:max_files])
            
        return relevant_files


class PRReviewAgent:
    """Agent for reviewing GitHub Pull Requests and providing professional code feedback."""
    
    def __init__(self):
        self.client = client
    
    def review_pr(self, pr_details, target_branch_code):
        """
        Review a GitHub pull request and provide professional code suggestions.
        
        Args:
            pr_details: Dictionary containing PR files, metadata, and changes
            target_branch_code: Dictionary of target branch files and their content
            
        Returns:
            A dictionary containing code suggestions and optimization recommendations
        """
        # Extract PR information
        pr_title = pr_details.get("title", "Untitled PR")
        pr_description = pr_details.get("description", "No description")
        changed_files = pr_details.get("changed_files", [])
        
        # Prepare context for the review
        context = f"Pull Request: {pr_title}\nDescription: {pr_description}\n\n"
        
        # Add changed files info
        if changed_files:
            context += "Files changed in this PR:\n"
            for file_info in changed_files:
                filename = file_info.get("filename", "unknown")
                changes = file_info.get("patch", "No changes available")
                context += f"--- {filename} ---\n{changes}\n\n"
        
        # Add target branch context for the files that were changed
        relevant_target_files = {}
        for file_info in changed_files:
            filename = file_info.get("filename", "")
            if filename in target_branch_code:
                relevant_target_files[filename] = target_branch_code[filename]
        
        if relevant_target_files:
            context += "Relevant files in target branch:\n"
            for filename, content in relevant_target_files.items():
                # Truncate long files
                if len(content) > 1000:
                    truncated_content = content[:1000] + "..."
                    context += f"--- {filename} (truncated) ---\n{truncated_content}\n\n"
                else:
                    context += f"--- {filename} ---\n{content}\n\n"
        
        # Generate code review
        code_review_prompt = f"""SYSTEM: You are a senior software developer reviewing a GitHub Pull Request.
        Provide professional, constructive feedback on the code changes. Focus on:
        
        1. Code style and adherence to best practices
        2. Potential bugs or issues
        3. Architecture and design considerations
        4. Performance implications
        
        CONTEXT INFORMATION:
        {context}
        
        Provide your code review in the following format:
        
        ## Overall Assessment
        [A brief 2-3 sentence assessment of the PR]
        
        ## Code Quality Suggestions
        - [Specific suggestion 1 with code example if applicable]
        - [Specific suggestion 2 with code example if applicable]
        - [Add more if necessary, at least 3 suggestions]
        
        ## Optimization Opportunities
        - [Specific optimization 1 with code example if applicable]
        - [Specific optimization 2 with code example if applicable]
        - [Add more if necessary, at least 2 suggestions]
        
        Your review should be professional, specific, and actionable. Provide code examples where appropriate.
        """
        
        review_result = prompt_llm(code_review_prompt)
        
        return {
            "review": review_result
        }

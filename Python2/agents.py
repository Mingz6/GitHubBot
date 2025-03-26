import warnings
import os
import json

warnings.filterwarnings("ignore")
from together import Together

# Load configuration from config.json
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

# Get API key from config
config = load_config()
your_api_key = config["together_ai_token"]
client = Together(api_key=your_api_key)


def prompt_llm(prompt, show_cost=False):
    # This function allows us to prompt an LLM via the Together API

    # model
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

    # Calculate the number of tokens
    tokens = len(prompt.split())

    # Calculate and print estimated cost for each model
    if show_cost:
        print(f"\nNumber of tokens: {tokens}")
        cost = (0.1 / 1_000_000) * tokens
        print(f"Estimated cost for {model}: ${cost:.10f}\n")

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


class SummarizerAgent:
    def __init__(self):
        self.client = Together(api_key=your_api_key)

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
        self.client = Together(api_key=your_api_key)

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
        self.client = Together(api_key=your_api_key)

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
        self.client = Together(api_key=your_api_key)
        
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
        self.client = Together(api_key=your_api_key)
        
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


class WebsiteContentSummarizer:
    def __init__(self):
        self.client = Together(api_key=your_api_key)

    def process(self, content):
        prompt = """SYSTEM: You are an expert website content summarizer. 
        Your task is to condense the provided website content into a clear, informative summary of exactly 4 lines.

        INSTRUCTIONS:
        • Identify and include only the most important information
        • Explain what the content is about and its main topics
        • Ensure the summary is exactly 4 lines long
        • Use concise, clear language
        • Show output only - provide just the summary
        • Do not include any other text or comments, show only the summary
        
        Website content to summarize: {content}
        
        Provide a 4-line summary:"""

        return prompt_llm(prompt.format(content=content))


class QuestionAnswerer:
    def __init__(self):
        self.client = Together(api_key=your_api_key)

    def answer_question(self, question, content, source_urls=None):
        """
        Generate an answer to the user's question based on website content.
        
        Args:
            question: The user's question
            content: Relevant content from the website
            source_urls: List of URLs where the information was found
        """
        if not content or content.strip() == "":
            return "Sorry, no information found on the website that answers your question."
            
        sources_text = ""
        if source_urls and len(source_urls) > 0:
            sources_text = "\n\nSources:\n" + "\n".join([f"- {url}" for url in source_urls[:3]])
            
        prompt = """SYSTEM: You are a helpful assistant that answers questions based on website content.

        INSTRUCTIONS:
        • Answer the user's question based ONLY on the provided website content
        • If the content doesn't contain information to answer the question, respond with EXACTLY: "Sorry, no information found on the website that answers your question."
        • Be concise but thorough in your answer
        • Use a neutral, informative tone
        • If there are sources, include them at the end of your answer
        • If there are multiple perspectives or options in the content, present them fairly
        • DO NOT make up or infer information that's not in the content
        
        USER QUESTION:
        {question}
        
        WEBSITE CONTENT:
        {content}
        
        Provide a helpful answer:"""

        answer = prompt_llm(prompt.format(question=question, content=content))
        
        if sources_text and "Sorry, no information found" not in answer:
            answer += sources_text
            
        return answer


class WebsiteSummarizer:
    def __init__(self):
        self.client = Together(api_key=your_api_key)

    def create_website_overview(self, content_dict):
        """
        Generate an overview summary of the entire website content.
        
        Args:
            content_dict: Dictionary with page URLs as keys and their content as values
        """
        # Combine content from all pages
        combined_content = []
        for url, page_data in content_dict.items():
            if isinstance(page_data, dict) and "title" in page_data and "content" in page_data:
                combined_content.append(f"Page: {page_data['title']}\n{page_data['content'][:500]}...")
        
        all_content = "\n\n".join(combined_content)
        
        # If content is too large, truncate it
        if len(all_content) > 8000:
            all_content = all_content[:8000] + "..."
            
        prompt = """SYSTEM: You are an expert website analyzer.

        INSTRUCTIONS:
        • Create a concise overview of the entire website based on the content provided
        • Focus on identifying the main topics, purpose, and type of website
        • Limit your response to 3-5 paragraphs
        • Be objective and informative
        • Include the main categories or sections of the website if identifiable
        
        WEBSITE CONTENT SAMPLES:
        {content}
        
        Provide a website overview:"""

        return prompt_llm(prompt.format(content=all_content))

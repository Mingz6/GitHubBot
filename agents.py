import warnings
import os

warnings.filterwarnings("ignore")
from together import Together

# Get Client
your_api_key = "9806a2601560024637df1e4acd804862faa67e08637db6598d920b64eebba43e"
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
        with open("/Users/mingz/code/GitHubBot/OldCode/PromptTemplate.json", "r") as f:
            import json
            prompt_data = json.load(f)
            template = prompt_data["prompt_template"]
            
        prompt = template.format(
            content=content,
            category=category,
            source=source
        )
        
        return prompt_llm(prompt)

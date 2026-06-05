import time
import logging
from typing import Generator
from groq import Groq, APIConnectionError, RateLimitError
from app.domain.interfaces import ILLMService
from app.core.config import settings

logger = logging.getLogger(__name__)


class GroqAdapter(ILLMService):
    def __init__(self, api_key: str = None):
        # Initialize client with provided key or fall back to core config settings
        self.api_key = api_key or settings.GROQ_API_KEY
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model = "llama-3.1-8b-instant"

    def generate_answer(self, context: str, query: str, system_prompt: str = None, provider: str = "groq") -> Generator[str, None, None]:
        """
        Generate an answer using the Groq API with streaming enabled.
        Supports tool calling for Python code execution and web document downloading.
        """
        if not self.client:
            raise ValueError("GROQ_API_KEY must be set in .env before using Groq.")

        # If no system prompt is provided, use default
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant. Answer the user's query using the provided context. "
                "Use Markdown formatting for your responses. If math/formula explanation is needed, "
                "use LaTeX format: $...$ for inline formulas and $$...$$ for block formulas."
            )
        
        user_content = f"Context:\n{context}\n\nQuery:\n{query}"

        # Define tool definitions for agentic workflows
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "Runs Python code securely. Use this for mathematical calculations, data analysis on CSV data, and rendering matplotlib charts/plots.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The exact valid Python code string to run."
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_automation_download",
                    "description": "Launches a headless browser using Playwright to navigate to a secure URL, searches for PDF links matching search_query, and downloads the file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The target website URL."
                            },
                            "search_query": {
                                "type": "string",
                                "description": "Search term or link text to click."
                            }
                        },
                        "required": ["url", "search_query"]
                    }
                }
            }
        ]

        max_retries = 3
        backoff_delay = 2.0  # Initial sleep duration in seconds

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    model=self.model,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                )
                
                tool_calls = []
                
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    
                    # Accumulate tool calls stream
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            while len(tool_calls) <= idx:
                                tool_calls.append({"id": "", "name": "", "arguments": ""})
                            
                            if tc.id:
                                tool_calls[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls[idx]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls[idx]["arguments"] += tc.function.arguments
                                    
                    elif delta.content:
                        yield delta.content
                
                # Yield reconstructed tool calls using special XML wrapper tags
                for tc in tool_calls:
                    if tc["name"]:
                        yield f"<tool_call:{tc['name']}>{tc['arguments']}</tool_call>"
                
                return
                
            except (RateLimitError, APIConnectionError) as e:
                logger.warning(
                    f"Groq API connection or rate limit error on attempt {attempt}: {e}. "
                    f"Retrying in {backoff_delay}s..."
                )
                if attempt == max_retries:
                    raise e
                
                time.sleep(backoff_delay)
                backoff_delay *= 2

import logging
import json
import requests
from typing import Generator
from app.domain.interfaces import ILLMService
from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiAdapter(ILLMService):
    def __init__(self, api_key: str = None):
        """
        Initialize the Gemini LLM service using REST API streaming.
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = "gemini-2.0-flash"

    def generate_answer(self, context: str, query: str, system_prompt: str = None, provider: str = "gemini") -> Generator[str, None, None]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env before using Gemini.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?key={self.api_key}"
        
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant. Answer the user's query using the provided context. "
                "Use Markdown formatting for your responses. If math/formula explanation is needed, "
                "use LaTeX format: $...$ for inline formulas and $$...$$ for block formulas."
            )
            
        user_content = f"Context:\n{context}\n\nQuery:\n{query}"
        
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "temperature": 0.2
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            logger.info("Sending request to Gemini API fallback...")
            r = requests.post(url, headers=headers, json=body, stream=True, timeout=15)
            r.raise_for_status()
            
            for line in r.iter_lines():
                if not line:
                    continue
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith(','):
                    decoded_line = decoded_line[1:].strip()
                if decoded_line.startswith('[') or decoded_line == ']':
                    continue
                try:
                    chunk_data = json.loads(decoded_line)
                    candidates = chunk_data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            if text:
                                yield text
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Gemini API streaming error: {e}", exc_info=True)
            raise e

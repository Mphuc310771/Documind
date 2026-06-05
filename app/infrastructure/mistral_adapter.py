import logging
import json
import requests
from typing import Generator
from app.domain.interfaces import ILLMService
from app.core.config import settings

logger = logging.getLogger(__name__)


class MistralAdapter(ILLMService):
    def __init__(self, api_key: str = None, model: str = None):
        """
        Mistral AI LLM service adapter.
        Uses OpenAI compatible chat completion endpoint.
        """
        self.api_key = api_key or settings.MISTRAL_API_KEY
        self.model = model or "mistral-large-latest"

    def generate_answer(self, context: str, query: str, system_prompt: str = None, provider: str = "mistral") -> Generator[str, None, None]:
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY must be set in .env before using Mistral.")

        url = "https://api.mistral.ai/v1/chat/completions"
        
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant. Answer the user's query using the provided context. "
                "Use Markdown formatting for your responses. If math/formula explanation is needed, "
                "use LaTeX format: $...$ for inline formulas and $$...$$ for block formulas."
            )
            
        user_content = f"Context:\n{context}\n\nQuery:\n{query}"
        
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "stream": True
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            logger.info(f"Sending request to Mistral API (model: {self.model})...")
            r = requests.post(url, headers=headers, json=body, stream=True, timeout=20)
            r.raise_for_status()
            
            for line in r.iter_lines():
                if not line:
                    continue
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Mistral API streaming error: {e}", exc_info=True)
            raise e

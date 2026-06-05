from abc import ABC, abstractmethod
from typing import Generator

class ILLMService(ABC):
    @abstractmethod
    def generate_answer(self, context: str, query: str, system_prompt: str = None, provider: str = "auto") -> Generator[str, None, None]:
        """
        Generate an answer stream based on the provided context and user query.
        Yields individual text tokens as they become available.
        """
        pass

import logging
from typing import Generator
from app.domain.interfaces import ILLMService
from app.infrastructure.groq_adapter import GroqAdapter
from app.infrastructure.gemini_adapter import GeminiAdapter

from app.infrastructure.openrouter_adapter import OpenRouterAdapter
from app.infrastructure.sambanova_adapter import SambaNovaAdapter
from app.infrastructure.mistral_adapter import MistralAdapter

logger = logging.getLogger(__name__)


class FallbackLLMService(ILLMService):
    def __init__(self, groq_adapter: GroqAdapter, gemini_adapter: GeminiAdapter, 
                 openrouter_adapter: OpenRouterAdapter = None,
                 sambanova_adapter: SambaNovaAdapter = None,
                 mistral_adapter: MistralAdapter = None):
        """
        LLM service wrapper implementing the Strategy/Decorator patterns to support
        explicit provider selection as well as automatic failover.
        """
        self.groq = groq_adapter
        self.gemini = gemini_adapter
        self.openrouter = openrouter_adapter
        self.sambanova = sambanova_adapter
        self.mistral = mistral_adapter

    def generate_answer(self, context: str, query: str, system_prompt: str = None, provider: str = "auto") -> Generator[str, None, None]:
        """
        Route request to the selected provider or run the auto-failover pipeline.
        """
        # If user explicitly chooses a provider, route to it directly
        if provider != "auto":
            if provider == "groq":
                yield from self.groq.generate_answer(context, query, system_prompt, provider)
            elif provider == "gemini":
                yield from self.gemini.generate_answer(context, query, system_prompt, provider)
            elif provider == "openrouter" and self.openrouter:
                yield from self.openrouter.generate_answer(context, query, system_prompt, provider)
            elif provider == "sambanova" and self.sambanova:
                yield from self.sambanova.generate_answer(context, query, system_prompt, provider)
            elif provider == "mistral" and self.mistral:
                yield from self.mistral.generate_answer(context, query, system_prompt, provider)
            else:
                raise ValueError(f"Provider '{provider}' không khả dụng hoặc chưa được cấu hình")
            return

        # Auto-failover pipeline (Mistral -> SambaNova -> Gemini -> Groq -> OpenRouter)
        pipeline = [
            ("Mistral", self.mistral, "mistral"),
            ("SambaNova", self.sambanova, "sambanova"),
            ("Gemini", self.gemini, "gemini"),
            ("Groq", self.groq, "groq"),
            ("OpenRouter", self.openrouter, "openrouter")
        ]
        
        # Filter out disabled / None adapters
        active_pipeline = [p for p in pipeline if p[1] is not None]
        
        for idx, (name, adapter, provider_id) in enumerate(active_pipeline):
            try:
                logger.info(f"Attempting LLM generation via {name}...")
                stream = adapter.generate_answer(context=context, query=query, system_prompt=system_prompt, provider=provider_id)
                
                iterator = iter(stream)
                try:
                    first_token = next(iterator)
                    yield first_token
                    for token in iterator:
                        yield token
                    # Successfully completed generation, exit the pipeline
                    return
                except StopIteration:
                    return
                except Exception as e:
                    logger.warning(f"{name} failed during generation: {e}")
                    if idx < len(active_pipeline) - 1:
                        next_name = active_pipeline[idx + 1][0]
                        logger.warning(f"Automatically falling back from {name} to {next_name}...")
                    else:
                        raise e
            except Exception as e:
                logger.warning(f"Failed to initialize generation via {name}: {e}")
                if idx < len(active_pipeline) - 1:
                    next_name = active_pipeline[idx + 1][0]
                    logger.warning(f"Automatically falling back from {name} to {next_name}...")
                else:
                    logger.error(f"All LLMs in failover pipeline failed. Last error from {name}: {e}", exc_info=True)
                    raise RuntimeError(f"Tất cả các hệ thống AI chính và dự phòng đều gặp sự cố. Lỗi: {str(e)}") from e


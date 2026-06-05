import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    SAMBANOVA_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None

    # ----- Optional authentication (multi-tenant). Disabled by default. -----
    AUTH_ENABLED: bool = False
    JWT_SECRET: str = "change-me-in-production-please"
    JWT_EXPIRE_HOURS: int = 168  # 7 days

    # ----- Optional background task queue (Celery + Redis). Disabled by default. -----
    CELERY_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    WHISPER_MODEL: str = "base"

    # ----- Screen-capture "contextual memory" daemon. Privacy-sensitive. -----
    SCREEN_CAPTURE_ENABLED: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()

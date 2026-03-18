"""
Centralised configuration via Pydantic Settings.
All values come from environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM Settings
    openai_api_key: str
    openai_model: str = "gpt-4.1-nano"

    # External API
    countries_api_base: str = "https://restcountries.com/v3.1"

    # HTTP
    api_timeout: int = 10          # seconds for REST Countries calls
    api_max_retries: int = 1       # retries on timeout

    # Cache
    cache_ttl_seconds: int = 300   # 5-minute TTL on country data

    # Context window
    max_context_tokens: int = 3000  # trim messages above this

    # Observability / LangSmith
    log_level: str = "INFO"

    # LangSmith tracing (optional, set to enable full trace dashboard)
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "false"   
    langchain_project: str = "country-info-agent"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()

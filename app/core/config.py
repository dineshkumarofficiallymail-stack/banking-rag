"""Centralised, typed configuration. Everything comes from environment variables / .env.
Nothing is hard-coded here; `get_settings()` is cached so it's read once per process."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "banking-rag"
    env: str = "dev"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    allow_registration: bool = True

    # Infra
    database_url: str
    redis_url: str | None = None

    # Auth
    jwt_secret: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    admin_email: str | None = None
    admin_password: str | None = None

    # LLM
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-4-5"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Embeddings
    embedding_provider: str = "fastembed"  # fastembed | fake
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    fastembed_cache_path: str = ".cache/fastembed"

    # RAG
    top_k: int = 6
    max_context_chars: int = 12_000
    min_similarity: float = 0.35
    rate_limit_per_minute: int = 30
    agent_max_steps: int = 4
    max_upload_mb: int = 25
    cache_ttl_seconds: int = 3600

    # Testing / CI
    offline_mode: bool = False

    @property
    def is_prod(self) -> bool:
        return self.env.lower() == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

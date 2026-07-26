from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Look for a local .env two levels up from this file (backend/.env),
# but don't error if it's not found — on Vercel, config comes from
# real environment variables set in the dashboard, not this file.
_LOCAL_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    APP_NAME: str = "Pakistan Ecology Graph RAG"
    APP_VERSION: str = "1.0.0"

    # LLM (Groq)
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Embeddings (Jina) — used for entity-linking, not bulk vector storage
    JINA_API_KEY: str
    JINA_MODEL: str = "jina-embeddings-v3"

    # Neo4j AuraDB
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str

    model_config = SettingsConfigDict(
        env_file=_LOCAL_ENV_FILE if _LOCAL_ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

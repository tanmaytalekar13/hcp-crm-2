from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"          # replaces gemma2-9b-it (dead) -> llama-3.1-8b-instant (now also deprecated)
    groq_fallback_model: str = "openai/gpt-oss-120b"  # replaces llama-3.3-70b-versatile (now deprecated)
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/hcp_crm"
    frontend_origin: str = "http://localhost:5173"

    class Config:
        env_file = ".env"


settings = Settings()
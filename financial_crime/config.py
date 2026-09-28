from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "sqlite:///./data/investigation.db"
    redis_url: str = "redis://localhost:6379/0"
    kafka_bootstrap_servers: str = "localhost:19092"
    llm_provider: str = "disabled"
    llm_fallback_provider: str = "disabled"
    llm_allow_evidence_export: bool = False
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    llm_timeout_seconds: float = Field(default=20, gt=0, le=120)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_max_input_tokens: int = Field(default=3000, ge=100, le=12000)
    llm_max_output_tokens: int = Field(default=1200, ge=1, le=4000)
    risk_review_threshold: float = Field(default=0.8, ge=0, le=1)
    model_artifact_dir: str = "artifacts/models"
    model_artifact_path: str = "artifacts/models/xgboost.joblib"
    data_dir: str = "data"
    api_cors_origins: str = "http://localhost:5173"
    otel_otlp_endpoint: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    """
    Project settings and configuration management.
    Uses pydantic-settings for validation and type safety.
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    openai_api_key: str | None = Field(default=os.getenv("OPENAI_API_KEY"))
    
    # Project Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    vector_db_path: str = str(data_dir / "chroma")
    
    # RAG Configuration
    collection_name: str = "rag_production"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4-turbo-preview"
    
    # Log Configuration
    log_level: str = "INFO"

settings = Settings()

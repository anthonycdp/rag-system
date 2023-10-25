"""Configuration settings for the RAG system."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # Embedding Configuration
    embedding_provider: Literal["openai", "huggingface"] = Field(
        default="openai", description="Embedding provider to use"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )

    # LLM Configuration
    llm_provider: Literal["openai"] = Field(
        default="openai", description="LLM provider to use"
    )
    llm_model: str = Field(
        default="gpt-4o-mini", description="LLM model name"
    )
    llm_temperature: float = Field(
        default=0.0, description="Temperature for LLM responses"
    )

    # Vector Store Configuration
    vector_store_type: Literal["chromadb"] = Field(
        default="chromadb", description="Vector store type"
    )
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        description="ChromaDB persistence directory",
    )
    chroma_collection_name: str = Field(
        default="rag_documents",
        description="ChromaDB collection name",
    )

    # RAG Configuration
    default_chunk_size: int = Field(
        default=512, description="Default chunk size for document splitting"
    )
    default_chunk_overlap: int = Field(
        default=50, description="Default chunk overlap"
    )
    default_top_k: int = Field(
        default=4, description="Default number of documents to retrieve"
    )

    # Evaluation Configuration
    evaluation_llm: str = Field(
        default="gpt-4o-mini", description="LLM for evaluation"
    )

    # Guardrails Configuration
    hallucination_threshold: float = Field(
        default=0.7,
        description="Threshold for hallucination detection (0-1)",
    )
    enable_hallucination_detection: bool = Field(
        default=True, description="Enable hallucination detection"
    )

    # Data Paths
    raw_data_dir: Path = Field(
        default=Path("./data/raw"),
        description="Directory for raw documents",
    )
    processed_data_dir: Path = Field(
        default=Path("./data/processed"),
        description="Directory for processed documents",
    )

    @field_validator("chroma_persist_dir", "raw_data_dir", "processed_data_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Ensure path is a Path object."""
        return Path(v) if isinstance(v, str) else v

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

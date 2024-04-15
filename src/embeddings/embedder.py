"""Embedding models for RAG systems."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


class EmbeddingProvider(str, Enum):
    """Available embedding providers."""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model_name: str = "text-embedding-3-small"
    # OpenAI specific
    openai_api_key: str | None = None
    # HuggingFace specific
    model_kwargs: dict[str, Any] = {}
    encode_kwargs: dict[str, Any] = {}


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Get the underlying LangChain embeddings instance.

        Returns:
            LangChain Embeddings instance.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            Number of dimensions in the embedding vector.
        """
        pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding model wrapper."""

    # Known dimensions for OpenAI models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize OpenAI embedder.

        Args:
            config: Embedding configuration.
        """
        self.config = config
        self._embeddings: Embeddings | None = None

    def get_embeddings(self) -> Embeddings:
        """Get OpenAI embeddings instance.

        Returns:
            OpenAI Embeddings instance.
        """
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings

            kwargs: dict[str, Any] = {}
            if self.config.openai_api_key:
                kwargs["openai_api_key"] = self.config.openai_api_key

            self._embeddings = OpenAIEmbeddings(
                model=self.config.model_name,
                **kwargs,
            )

        return self._embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension based on model.

        Returns:
            Dimension of the embedding vectors.
        """
        return self.MODEL_DIMENSIONS.get(self.config.model_name, 1536)


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace embedding model wrapper."""

    # Known dimensions for common models
    MODEL_DIMENSIONS = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/multi-qa-MiniLM-L6-dot-v1": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize HuggingFace embedder.

        Args:
            config: Embedding configuration.
        """
        self.config = config
        self._embeddings: Embeddings | None = None

    def get_embeddings(self) -> Embeddings:
        """Get HuggingFace embeddings instance.

        Returns:
            HuggingFaceEmbeddings instance.
        """
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                model_kwargs=self.config.model_kwargs,
                encode_kwargs=self.config.encode_kwargs,
            )

        return self._embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension based on model.

        Returns:
            Dimension of the embedding vectors.
        """
        return self.MODEL_DIMENSIONS.get(self.config.model_name, 768)


def create_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    """Factory function to create appropriate embedder.

    Args:
        config: Embedding configuration.

    Returns:
        Embedder instance.

    Raises:
        ValueError: If provider is not supported.
    """
    embedder_map: dict[EmbeddingProvider, type[BaseEmbedder]] = {
        EmbeddingProvider.OPENAI: OpenAIEmbedder,
        EmbeddingProvider.HUGGINGFACE: HuggingFaceEmbedder,
    }

    if config.provider not in embedder_map:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")

    return embedder_map[config.provider](config)


def get_default_embedder(
    provider: str = "openai",
    model_name: str | None = None,
    api_key: str | None = None,
) -> tuple[BaseEmbedder, Embeddings]:
    """Get a default embedder with sensible defaults.

    Args:
        provider: Embedding provider name.
        model_name: Optional model name override.
        api_key: Optional API key for OpenAI.

    Returns:
        Tuple of (embedder, embeddings instance).
    """
    provider_enum = EmbeddingProvider(provider.lower())

    # Set default model name based on provider
    if model_name is None:
        model_name = (
            "text-embedding-3-small"
            if provider_enum == EmbeddingProvider.OPENAI
            else "sentence-transformers/all-MiniLM-L6-v2"
        )

    config = EmbeddingConfig(
        provider=provider_enum,
        model_name=model_name,
        openai_api_key=api_key,
    )

    embedder = create_embedder(config)
    return embedder, embedder.get_embeddings()

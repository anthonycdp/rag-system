"""Document chunking strategies for RAG systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
)

from src.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEPARATORS,
    SENTENCE_AWARE_SEPARATORS,
)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    TOKEN = "token"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    separators: list[str] | None = None
    length_function: Any = len
    keep_separator: bool = True
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    tokens_per_chunk: int | None = None


@dataclass
class ChunkingResult:
    """Result of chunking operation."""

    chunks: list[Document] = field(default_factory=list)
    total_chunks: int = 0
    total_characters: int = 0
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize the chunker.

        Args:
            config: Chunking configuration.
        """
        self.config = config

    @abstractmethod
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List of chunked documents.
        """
        pass

    def get_statistics(self, chunks: list[Document]) -> ChunkingResult:
        """Calculate statistics for chunked documents.

        Args:
            chunks: List of chunked documents.

        Returns:
            ChunkingResult with statistics.
        """
        if not chunks:
            return ChunkingResult()

        sizes = [len(chunk.page_content) for chunk in chunks]

        return ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            total_characters=sum(sizes),
            avg_chunk_size=sum(sizes) / len(sizes),
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
        )


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter with customizable separators."""

    def __init__(self, config: ChunkingConfig) -> None:
        super().__init__(config)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators or DEFAULT_SEPARATORS,
            length_function=config.length_function,
            keep_separator=config.keep_separator,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents using recursive character splitting.

        Args:
            documents: List of documents to split.

        Returns:
            List of chunked documents with preserved metadata.
        """
        chunks = self.splitter.split_documents(documents)

        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks


class TokenChunker(BaseChunker):
    """Token-based text splitter using HuggingFace tokenizers."""

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize token chunker.

        Args:
            config: Chunking configuration.
        """
        super().__init__(config)
        self.splitter = SentenceTransformersTokenTextSplitter(
            model_name=config.model_name,
            tokens_per_chunk=config.tokens_per_chunk or config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents based on token count.

        Args:
            documents: List of documents to split.

        Returns:
            List of chunked documents.
        """
        chunks = self.splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking that splits at sentence boundaries."""

    def __init__(self, config: ChunkingConfig) -> None:
        super().__init__(config)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=SENTENCE_AWARE_SEPARATORS,
            length_function=config.length_function,
            keep_separator=False,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents semantically at sentence boundaries.

        Args:
            documents: List of documents to split.

        Returns:
            List of chunked documents.
        """
        chunks = self.splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk.metadata["chunking_strategy"] = "semantic"

        return chunks


def create_chunker(config: ChunkingConfig) -> BaseChunker:
    """Factory function to create appropriate chunker.

    Args:
        config: Chunking configuration.

    Returns:
        Chunker instance.

    Raises:
        ValueError: If chunking strategy is not supported.
    """
    chunker_map: dict[ChunkingStrategy, type[BaseChunker]] = {
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.TOKEN: TokenChunker,
    }

    if config.strategy not in chunker_map:
        raise ValueError(f"Unsupported chunking strategy: {config.strategy}")

    return chunker_map[config.strategy](config)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
) -> tuple[list[Document], ChunkingResult]:
    """Convenience function to chunk documents.

    Args:
        documents: List of documents to chunk.
        chunk_size: Target size for each chunk.
        chunk_overlap: Overlap between chunks.
        strategy: Chunking strategy to use.

    Returns:
        Tuple of (chunked documents, chunking statistics).
    """
    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunker = create_chunker(config)
    chunks = chunker.split_documents(documents)
    stats = chunker.get_statistics(chunks)

    return chunks, stats

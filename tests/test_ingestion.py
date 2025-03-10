"""Tests for document ingestion module."""

import pytest
from langchain_core.documents import Document

from src.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    DocumentLoader,
    RecursiveChunker,
    chunk_documents,
    load_sample_documents,
)


class TestDocumentLoader:
    """Tests for DocumentLoader class."""

    def test_load_sample_documents(self) -> None:
        """Test loading sample documents."""
        documents = load_sample_documents()

        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(doc.page_content for doc in documents)
        assert all(doc.metadata.get("source") for doc in documents)


class TestChunking:
    """Tests for document chunking."""

    def test_recursive_chunker(self) -> None:
        """Test recursive character chunker."""
        documents = [
            Document(
                page_content="This is a test document. " * 50,
                metadata={"source": "test.txt"},
            )
        ]

        config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
        )

        chunker = RecursiveChunker(config)
        chunks = chunker.split_documents(documents)

        assert len(chunks) > 1
        assert all(chunk.page_content for chunk in chunks)

    def test_chunk_documents_convenience(self) -> None:
        """Test convenience function for chunking."""
        documents = [
            Document(
                page_content="Test content. " * 100,
                metadata={"source": "test.txt"},
            )
        ]

        chunks, stats = chunk_documents(
            documents,
            chunk_size=200,
            chunk_overlap=50,
        )

        assert len(chunks) > 0
        assert stats.total_chunks == len(chunks)
        assert stats.total_characters > 0
        assert stats.avg_chunk_size > 0

    def test_chunk_metadata_preserved(self) -> None:
        """Test that metadata is preserved during chunking."""
        documents = [
            Document(
                page_content="Test content with some text. " * 20,
                metadata={"source": "test.txt", "custom": "value"},
            )
        ]

        chunks, _ = chunk_documents(documents, chunk_size=100, chunk_overlap=20)

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["custom"] == "value"
            assert "chunk_index" in chunk.metadata

    def test_empty_documents(self) -> None:
        """Test chunking empty document list."""
        chunks, stats = chunk_documents([])

        assert len(chunks) == 0
        assert stats.total_chunks == 0


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ChunkingConfig()

        assert config.strategy == ChunkingStrategy.RECURSIVE
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=1024,
            chunk_overlap=100,
        )

        assert config.strategy == ChunkingStrategy.SEMANTIC
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100

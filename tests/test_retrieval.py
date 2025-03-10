"""Tests for retrieval module."""

import pytest
from langchain_core.documents import Document

from src.retrieval import (
    RetrievalResult,
    RetrieverConfig,
    SearchType,
)


class TestRetrieverConfig:
    """Tests for RetrieverConfig."""

    def test_default_config(self) -> None:
        """Test default retriever configuration."""
        config = RetrieverConfig()

        assert config.search_type == SearchType.SIMILARITY
        assert config.top_k == 4
        assert config.fetch_k == 20
        assert config.lambda_mult == 0.5

    def test_custom_config(self) -> None:
        """Test custom retriever configuration."""
        config = RetrieverConfig(
            search_type=SearchType.MMR,
            top_k=6,
            fetch_k=30,
            lambda_mult=0.7,
        )

        assert config.search_type == SearchType.MMR
        assert config.top_k == 6
        assert config.fetch_k == 30
        assert config.lambda_mult == 0.7


class TestRetrievalResult:
    """Tests for RetrievalResult."""

    def test_empty_result(self) -> None:
        """Test empty retrieval result."""
        result = RetrievalResult()

        assert len(result.documents) == 0
        assert len(result.scores) == 0
        assert result.total_retrieved == 0

    def test_result_with_documents(self) -> None:
        """Test retrieval result with documents."""
        docs = [
            Document(page_content="Doc 1", metadata={"id": 1}),
            Document(page_content="Doc 2", metadata={"id": 2}),
        ]

        result = RetrievalResult(
            documents=docs,
            scores=[0.9, 0.8],
            query="test query",
            total_retrieved=2,
        )

        assert len(result.documents) == 2
        assert len(result.scores) == 2
        assert result.query == "test query"
        assert result.total_retrieved == 2


class TestSearchType:
    """Tests for SearchType enum."""

    def test_search_types(self) -> None:
        """Test available search types."""
        assert SearchType.SIMILARITY.value == "similarity"
        assert SearchType.MMR.value == "mmr"
        assert SearchType.SIMILARITY_SCORE.value == "similarity_score_threshold"

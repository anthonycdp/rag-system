"""Tests for RAG pipeline module."""

import pytest

from src.pipeline import RAGPipelineConfig, RAGResult


class TestRAGPipelineConfig:
    """Tests for RAGPipelineConfig."""

    def test_default_config(self) -> None:
        """Test default pipeline configuration."""
        config = RAGPipelineConfig()

        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.top_k == 4
        assert config.llm_model == "gpt-4o-mini"

    def test_custom_config(self) -> None:
        """Test custom pipeline configuration."""
        config = RAGPipelineConfig(
            embedding_provider="huggingface",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=1024,
            top_k=6,
            llm_model="gpt-4",
        )

        assert config.embedding_provider == "huggingface"
        assert config.chunk_size == 1024
        assert config.top_k == 6
        assert config.llm_model == "gpt-4"


class TestRAGResult:
    """Tests for RAGResult."""

    def test_empty_result(self) -> None:
        """Test empty RAG result."""
        result = RAGResult(query="test", answer="")

        assert result.query == "test"
        assert result.answer == ""
        assert len(result.sources) == 0
        assert len(result.retrieval_scores) == 0

    def test_result_with_data(self) -> None:
        """Test RAG result with data."""
        from langchain_core.documents import Document

        result = RAGResult(
            query="What is Python?",
            answer="Python is a programming language.",
            sources=[
                Document(page_content="Python docs", metadata={}),
            ],
            retrieval_scores=[0.95],
            generation_tokens=100,
            latency_seconds=1.5,
        )

        assert result.query == "What is Python?"
        assert "programming language" in result.answer
        assert len(result.sources) == 1
        assert result.generation_tokens == 100


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""

    def test_metric_types(self) -> None:
        """Test available metric types."""
        from src.evaluation import MetricType

        assert MetricType.FAITHFULNESS.value == "faithfulness"
        assert MetricType.ANSWER_RELEVANCY.value == "answer_relevancy"
        assert MetricType.CONTEXT_RELEVANCY.value == "context_relevancy"

    def test_metric_result(self) -> None:
        """Test metric result creation."""
        from src.evaluation import MetricResult

        result = MetricResult(
            metric_name="faithfulness",
            score=0.85,
            explanation="Answer is mostly grounded",
            details={"claims_count": 5},
        )

        assert result.metric_name == "faithfulness"
        assert result.score == 0.85
        assert "grounded" in result.explanation


class TestGuardrails:
    """Tests for guardrails module."""

    def test_hallucination_level(self) -> None:
        """Test hallucination levels."""
        from src.guardrails import HallucinationLevel

        assert HallucinationLevel.NONE.value == "none"
        assert HallucinationLevel.LOW.value == "low"
        assert HallucinationLevel.MEDIUM.value == "medium"
        assert HallucinationLevel.HIGH.value == "high"
        assert HallucinationLevel.CRITICAL.value == "critical"

    def test_hallucination_config(self) -> None:
        """Test hallucination configuration."""
        from src.guardrails import HallucinationConfig

        config = HallucinationConfig(
            model_name="gpt-4o-mini",
            threshold=0.8,
            enable_suggestions=True,
        )

        assert config.model_name == "gpt-4o-mini"
        assert config.threshold == 0.8
        assert config.enable_suggestions is True

    def test_hallucination_result(self) -> None:
        """Test hallucination result."""
        from src.guardrails import HallucinationLevel, HallucinationResult

        result = HallucinationResult(
            is_hallucination=False,
            level=HallucinationLevel.LOW,
            overall_score=0.85,
            ungrounded_claims=[],
        )

        assert result.is_hallucination is False
        assert result.level == HallucinationLevel.LOW
        assert result.overall_score == 0.85

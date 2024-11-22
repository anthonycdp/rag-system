"""Evaluation metrics for RAG systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from src.constants import DEFAULT_LLM_MODEL
from src.utils import create_llm, extract_json_from_response, format_context_text


class MetricType(str, Enum):
    """Types of evaluation metrics."""

    # Retrieval metrics
    PRECISION = "precision"
    RECALL = "recall"
    MRR = "mrr"
    NDCG = "ndcg"

    # Generation metrics
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_RELEVANCY = "context_relevancy"
    CORRECTNESS = "correctness"
    GROUNDEDNESS = "groundedness"


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""

    metric_name: str
    score: float
    explanation: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class MetricConfig(BaseModel):
    """Configuration for metrics."""

    llm_model: str = DEFAULT_LLM_MODEL
    api_key: str | None = None


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(self, config: MetricConfig | None = None) -> None:
        self.config = config or MetricConfig()
        self._llm: BaseChatModel | None = None

    def _get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = create_llm(
                model_name=self.config.llm_model,
                api_key=self.config.api_key,
            )
        return self._llm

    def _evaluate_with_llm(self, prompt: str) -> MetricResult:
        """Evaluate using LLM and parse JSON response.

        Args:
            prompt: The prompt to send to the LLM.
            metric_name: Name of the metric for error reporting.

        Returns:
            Parsed MetricResult.

        Raises:
            ValueError: If JSON parsing fails.
        """
        llm = self._get_llm()
        response = llm.invoke(prompt)
        content = response.content

        try:
            result = extract_json_from_response(content)
            return MetricResult(
                metric_name=self.metric_name,
                score=float(result.get("score", 0.0)),
                explanation=result.get("explanation", ""),
                details={k: v for k, v in result.items() if k not in ["score", "explanation"]},
            )
        except ValueError as e:
            return MetricResult(
                metric_name=self.metric_name,
                score=0.0,
                explanation=f"Failed to parse evaluation: {e}",
                details={"raw_response": content},
            )

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return the metric name."""
        pass

    @abstractmethod
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[Document],
        ground_truth: str | None = None,
    ) -> MetricResult:
        """Evaluate the metric."""
        pass


class FaithfulnessMetric(BaseMetric):
    """Evaluates if the answer is faithful to the retrieved context."""

    EVALUATION_PROMPT = """Evaluate if the following answer is faithful to the provided context.

Context:
{context}

Answer: {answer}

Instructions:
1. Check if all claims in the answer are supported by the context
2. Identify any hallucinations or claims not grounded in context
3. Score from 0.0 (completely unfaithful) to 1.0 (fully faithful)

Provide your response as JSON:
{{"score": <float 0-1>, "explanation": "<explanation>", "ungrounded_claims": ["claim1", ...]}}"""

    @property
    def metric_name(self) -> str:
        return MetricType.FAITHFULNESS.value

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[Document],
        ground_truth: str | None = None,
    ) -> MetricResult:
        context_text = format_context_text(contexts)
        prompt = self.EVALUATION_PROMPT.format(context=context_text, answer=answer)
        return self._evaluate_with_llm(prompt)


class AnswerRelevancyMetric(BaseMetric):
    """Evaluates how relevant the answer is to the query."""

    EVALUATION_PROMPT = """Evaluate how relevant the following answer is to the given question.

Question: {query}

Answer: {answer}

Instructions:
1. Assess if the answer directly addresses the question
2. Check if the answer is complete and informative
3. Score from 0.0 (completely irrelevant) to 1.0 (highly relevant)

Provide your response as JSON:
{{"score": <float 0-1>, "explanation": "<explanation>", "addressed_aspects": ["aspect1", ...], "missed_aspects": ["aspect1", ...]}}"""

    @property
    def metric_name(self) -> str:
        return MetricType.ANSWER_RELEVANCY.value

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[Document],
        ground_truth: str | None = None,
    ) -> MetricResult:
        prompt = self.EVALUATION_PROMPT.format(query=query, answer=answer)
        return self._evaluate_with_llm(prompt)


class ContextRelevancyMetric(BaseMetric):
    """Evaluates how relevant the retrieved context is to the query."""

    EVALUATION_PROMPT = """Evaluate how relevant the retrieved context is to the given question.

Question: {query}

Retrieved Context:
{context}

Instructions:
1. Assess if the context contains information useful for answering the question
2. Check if irrelevant or redundant information is present
3. Score from 0.0 (completely irrelevant) to 1.0 (highly relevant)

Provide your response as JSON:
{{"score": <float 0-1>, "explanation": "<explanation>", "relevant_parts": ["part1", ...], "irrelevant_parts": ["part1", ...]}}"""

    @property
    def metric_name(self) -> str:
        return MetricType.CONTEXT_RELEVANCY.value

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[Document],
        ground_truth: str | None = None,
    ) -> MetricResult:
        context_text = format_context_text(contexts)
        prompt = self.EVALUATION_PROMPT.format(query=query, context=context_text)
        return self._evaluate_with_llm(prompt)


class GroundednessMetric(BaseMetric):
    """Evaluates if statements are grounded in the context."""

    EVALUATION_PROMPT = """Evaluate how well the following statements are grounded in the provided context.

Context:
{context}

Statements to evaluate:
{statements}

For each statement, determine if it can be inferred from the context.
Score from 0.0 (not grounded) to 1.0 (fully grounded).

Provide your response as JSON:
{{"score": <float 0-1>, "explanation": "<explanation>", "grounded_statements": ["stmt1", ...], "ungrounded_statements": ["stmt1", ...]}}"""

    @property
    def metric_name(self) -> str:
        return MetricType.GROUNDEDNESS.value

    def _extract_statements(self, answer: str) -> list[str]:
        """Extract statements from answer."""
        import re
        sentences = re.split(r"[.!?]+", answer)
        return [s.strip() for s in sentences if s.strip()]

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[Document],
        ground_truth: str | None = None,
    ) -> MetricResult:
        statements = self._extract_statements(answer)

        if not statements:
            return MetricResult(
                metric_name=self.metric_name,
                score=0.0,
                explanation="No statements to evaluate",
            )

        context_text = format_context_text(contexts)
        statements_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))

        prompt = self.EVALUATION_PROMPT.format(
            context=context_text,
            statements=statements_text,
        )
        return self._evaluate_with_llm(prompt)


def create_metric(
    metric_type: MetricType,
    config: MetricConfig | None = None,
) -> BaseMetric:
    """Factory function to create metrics.

    Args:
        metric_type: Type of metric to create.
        config: Metric configuration.

    Returns:
        Metric instance.

    Raises:
        ValueError: If metric type is not supported.
    """
    metric_map: dict[MetricType, type[BaseMetric]] = {
        MetricType.FAITHFULNESS: FaithfulnessMetric,
        MetricType.ANSWER_RELEVANCY: AnswerRelevancyMetric,
        MetricType.CONTEXT_RELEVANCY: ContextRelevancyMetric,
        MetricType.GROUNDEDNESS: GroundednessMetric,
    }

    if metric_type not in metric_map:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    return metric_map[metric_type](config)

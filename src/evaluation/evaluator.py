"""RAG evaluation framework."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel

from src.evaluation.metrics import (
    BaseMetric,
    MetricConfig,
    MetricResult,
    MetricType,
    create_metric,
)
from src.pipeline import RAGPipeline, RAGResult


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""

    metrics: list[str] = ["faithfulness", "answer_relevancy", "context_relevancy"]
    llm_model: str = "gpt-4o-mini"
    output_dir: Path = Path("./data/evaluation_results")


@dataclass
class EvaluationSample:
    """Single evaluation sample."""

    query: str
    answer: str
    contexts: list[Document]
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single sample."""

    sample_id: int
    query: str
    answer: str
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    avg_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    timestamp: str
    total_samples: int
    avg_scores: dict[str, float] = field(default_factory=dict)
    std_scores: dict[str, float] = field(default_factory=dict)
    min_scores: dict[str, float] = field(default_factory=dict)
    max_scores: dict[str, float] = field(default_factory=dict)
    results: list[EvaluationResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dictionary representation of the report.
        """
        return {
            "timestamp": self.timestamp,
            "total_samples": self.total_samples,
            "avg_scores": self.avg_scores,
            "std_scores": self.std_scores,
            "min_scores": self.min_scores,
            "max_scores": self.max_scores,
            "results": [
                {
                    "sample_id": r.sample_id,
                    "query": r.query,
                    "answer": r.answer,
                    "avg_score": r.avg_score,
                    "metrics": {
                        k: {"score": v.score, "explanation": v.explanation}
                        for k, v in r.metrics.items()
                    },
                }
                for r in self.results
            ],
            "config": self.config,
        }

    def save(self, path: Path | None = None) -> None:
        """Save report to JSON file.

        Args:
            path: Output path. Uses default if not provided.
        """
        import statistics

        if path is None:
            path = Path(f"./data/evaluation_results/eval_{self.timestamp}.json")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EvaluationReport":
        """Load report from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            EvaluationReport instance.
        """
        with open(path) as f:
            data = json.load(f)

        return cls(
            timestamp=data["timestamp"],
            total_samples=data["total_samples"],
            avg_scores=data["avg_scores"],
            std_scores=data["std_scores"],
            min_scores=data["min_scores"],
            max_scores=data["max_scores"],
            config=data.get("config", {}),
        )


class RAGEvaluator:
    """Evaluates RAG pipeline performance."""

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration.
            api_key: OpenAI API key.
        """
        self.config = config or EvaluationConfig()
        self.api_key = api_key

        # Initialize metrics
        metric_config = MetricConfig(
            llm_model=self.config.llm_model,
            api_key=api_key,
        )
        self._metrics: dict[str, BaseMetric] = {}
        for metric_name in self.config.metrics:
            metric_type = MetricType(metric_name)
            self._metrics[metric_name] = create_metric(metric_type, metric_config)

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        sample_id: int = 0,
    ) -> EvaluationResult:
        """Evaluate a single sample.

        Args:
            sample: Sample to evaluate.
            sample_id: Sample identifier.

        Returns:
            EvaluationResult with all metric scores.
        """
        metric_results: dict[str, MetricResult] = {}

        for name, metric in self._metrics.items():
            result = metric.evaluate(
                query=sample.query,
                answer=sample.answer,
                contexts=sample.contexts,
                ground_truth=sample.ground_truth,
            )
            metric_results[name] = result

        # Calculate average score
        scores = [r.score for r in metric_results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return EvaluationResult(
            sample_id=sample_id,
            query=sample.query,
            answer=sample.answer,
            metrics=metric_results,
            avg_score=avg_score,
            metadata=sample.metadata,
        )

    def evaluate_samples(
        self,
        samples: list[EvaluationSample],
    ) -> EvaluationReport:
        """Evaluate multiple samples and generate a report.

        Args:
            samples: List of samples to evaluate.

        Returns:
            EvaluationReport with aggregated results.
        """
        import statistics

        results: list[EvaluationResult] = []
        all_scores: dict[str, list[float]] = {name: [] for name in self._metrics}

        for i, sample in enumerate(samples):
            result = self.evaluate_sample(sample, sample_id=i)
            results.append(result)

            for name, metric_result in result.metrics.items():
                all_scores[name].append(metric_result.score)

        # Calculate aggregated statistics
        avg_scores: dict[str, float] = {}
        std_scores: dict[str, float] = {}
        min_scores: dict[str, float] = {}
        max_scores: dict[str, float] = {}

        for name, scores in all_scores.items():
            if scores:
                avg_scores[name] = statistics.mean(scores)
                std_scores[name] = statistics.stdev(scores) if len(scores) > 1 else 0.0
                min_scores[name] = min(scores)
                max_scores[name] = max(scores)

        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_samples=len(samples),
            avg_scores=avg_scores,
            std_scores=std_scores,
            min_scores=min_scores,
            max_scores=max_scores,
            results=results,
            config={
                "metrics": self.config.metrics,
                "llm_model": self.config.llm_model,
            },
        )

    def evaluate_pipeline(
        self,
        pipeline: RAGPipeline,
        queries: list[str],
        ground_truths: list[str] | None = None,
    ) -> EvaluationReport:
        """Evaluate a RAG pipeline on a set of queries.

        Args:
            pipeline: RAG pipeline to evaluate.
            queries: List of queries to evaluate.
            ground_truths: Optional ground truth answers.

        Returns:
            EvaluationReport with results.
        """
        samples: list[EvaluationSample] = []

        for i, query in enumerate(queries):
            # Run query through pipeline
            result = pipeline.query(query)

            sample = EvaluationSample(
                query=query,
                answer=result.answer,
                contexts=result.sources,
                ground_truth=ground_truths[i] if ground_truths else None,
                metadata={"latency": result.latency_seconds},
            )
            samples.append(sample)

        return self.evaluate_samples(samples)


def create_test_queries() -> list[tuple[str, str]]:
    """Create test queries with expected answers for Python domain.

    Returns:
        List of (query, ground_truth) tuples.
    """
    return [
        (
            "What are the basic data types in Python?",
            "Python has numeric types (int, float, complex), sequence types (str, list, tuple), "
            "mapping types (dict), set types (set, frozenset), and boolean type (bool).",
        ),
        (
            "How do you define a function in Python?",
            "Functions are defined using the 'def' keyword followed by the function name, "
            "parameters in parentheses, and a docstring. Example: def function_name(parameters):",
        ),
        (
            "What is the difference between a list and a tuple?",
            "Lists are mutable ordered collections while tuples are immutable ordered collections. "
            "Tuples can be used as dictionary keys but lists cannot.",
        ),
        (
            "How does exception handling work in Python?",
            "Python uses try-except blocks for exception handling. Code that might raise an "
            "exception goes in the try block, and specific exceptions are caught in except blocks. "
            "A finally block can be used for cleanup code.",
        ),
        (
            "What are the different ways to handle concurrency in Python?",
            "Python offers three main approaches: threading (for I/O-bound tasks), "
            "multiprocessing (for CPU-bound tasks with true parallelism), and asyncio "
            "(for asynchronous I/O with cooperative multitasking).",
        ),
        (
            "How do you read and write files in Python?",
            "Use the 'with' statement and open() function. For reading: with open('file.txt', 'r') as f: "
            "content = f.read(). For writing: with open('file.txt', 'w') as f: f.write('content').",
        ),
        (
            "What are dunder methods in Python?",
            "Dunder (double underscore) methods are special methods like __init__, __str__, __repr__, "
            "__len__, __getitem__, __iter__ that define how objects behave with built-in operations.",
        ),
        (
            "What is the time complexity of dictionary operations?",
            "Dictionary operations like lookup, insert, and delete have O(1) average time complexity.",
        ),
    ]

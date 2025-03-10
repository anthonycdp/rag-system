#!/usr/bin/env python3
"""Script to tune RAG hyperparameters for optimal retrieval."""

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.constants import (
    QUICK_MODE_QUERY_COUNT,
    SCORE_EXCELLENT,
    SCORE_GOOD,
    FAITHFULNESS_WEIGHT,
    ANSWER_RELEVANCY_WEIGHT,
    CONTEXT_RELEVANCY_WEIGHT,
    MAX_RESULTS_IN_TABLE,
)

console = Console()


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    chunk_sizes: list[int] = None
    chunk_overlaps: list[int] = None
    top_k_values: list[int] = None

    def __post_init__(self):
        if self.chunk_sizes is None:
            self.chunk_sizes = [256, 512, 768]
        if self.chunk_overlaps is None:
            self.chunk_overlaps = [25, 50, 100]
        if self.top_k_values is None:
            self.top_k_values = [2, 4, 6]


@dataclass
class ExperimentResult:
    """Result of a single hyperparameter experiment."""

    chunk_size: int
    chunk_overlap: int
    top_k: int
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_relevancy: float = 0.0
    overall_score: float = 0.0
    latency_avg: float = 0.0


def run_single_experiment(
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    api_key: str | None = None,
    queries: list[str] | None = None,
) -> ExperimentResult:
    """Run a single hyperparameter experiment.

    Args:
        chunk_size: Chunk size to test.
        chunk_overlap: Chunk overlap to test.
        top_k: Number of documents to retrieve.
        api_key: OpenAI API key.
        queries: Test queries.

    Returns:
        ExperimentResult with metrics.
    """
    from src.evaluation import EvaluationConfig, RAGEvaluator, create_test_queries
    from src.ingestion import load_sample_documents
    from src.pipeline import RAGPipeline, RAGPipelineConfig
    from src.retrieval import VectorStoreManager

    # Get test queries
    if queries is None:
        test_data = create_test_queries()
        queries = [q[0] for q in test_data][:QUICK_MODE_QUERY_COUNT]

    # Create pipeline config
    config = RAGPipelineConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )

    # Create pipeline
    pipeline = RAGPipeline(config=config, api_key=api_key or settings.openai_api_key)

    # Clear any existing vector store and re-ingest
    pipeline.clear_vector_store()

    # Ingest documents with current chunking params
    documents = load_sample_documents()
    pipeline.ingest_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Evaluate
    eval_config = EvaluationConfig(
        metrics=["faithfulness", "answer_relevancy", "context_relevancy"],
        llm_model=settings.evaluation_llm,
    )
    evaluator = RAGEvaluator(config=eval_config, api_key=api_key or settings.openai_api_key)

    # Run queries and collect latencies
    latencies = []
    for query in queries:
        result = pipeline.query(query)
        latencies.append(result.latency_seconds)

    # Evaluate on subset of queries
    report = evaluator.evaluate_pipeline(pipeline=pipeline, queries=queries)

    # Calculate overall score
    avg_faithfulness = report.avg_scores.get("faithfulness", 0.0)
    avg_answer_relevancy = report.avg_scores.get("answer_relevancy", 0.0)
    avg_context_relevancy = report.avg_scores.get("context_relevancy", 0.0)

    # Weighted overall score using centralized weights
    overall_score = (
        FAITHFULNESS_WEIGHT * avg_faithfulness +
        ANSWER_RELEVANCY_WEIGHT * avg_answer_relevancy +
        CONTEXT_RELEVANCY_WEIGHT * avg_context_relevancy
    )

    result = ExperimentResult(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        avg_faithfulness=avg_faithfulness,
        avg_answer_relevancy=avg_answer_relevancy,
        avg_context_relevancy=avg_context_relevancy,
        overall_score=overall_score,
        latency_avg=sum(latencies) / len(latencies) if latencies else 0.0,
    )

    # Clean up
    pipeline.clear_vector_store()

    return result


def run_tuning(
    config: TuningConfig | None = None,
    api_key: str | None = None,
    output_path: Path | None = None,
    quick_mode: bool = False,
) -> list[ExperimentResult]:
    """Run hyperparameter tuning experiments.

    Args:
        config: Tuning configuration.
        api_key: OpenAI API key.
        output_path: Path to save results.
        quick_mode: Use fewer combinations for faster results.

    Returns:
        List of experiment results.
    """
    if config is None:
        if quick_mode:
            config = TuningConfig(
                chunk_sizes=[256, 512],
                chunk_overlaps=[50],
                top_k_values=[4],
            )
        else:
            config = TuningConfig()

    # Generate all combinations
    combinations = list(product(
        config.chunk_sizes,
        config.chunk_overlaps,
        config.top_k_values,
    ))

    console.print(f"[bold blue]Hyperparameter Tuning[/bold blue]\n")
    console.print(f"Testing {len(combinations)} parameter combinations\n")

    # Show configuration
    config_table = Table(title="Tuning Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Values")
    config_table.add_row("Chunk Sizes", ", ".join(map(str, config.chunk_sizes)))
    config_table.add_row("Chunk Overlaps", ", ".join(map(str, config.chunk_overlaps)))
    config_table.add_row("Top-K Values", ", ".join(map(str, config.top_k_values)))
    console.print(config_table)

    results: list[ExperimentResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Running experiments...",
            total=len(combinations),
        )

        for chunk_size, chunk_overlap, top_k in combinations:
            progress.update(
                task,
                description=f"Testing chunk_size={chunk_size}, overlap={chunk_overlap}, k={top_k}",
                advance=1,
            )

            try:
                result = run_single_experiment(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    api_key=api_key,
                )
                results.append(result)
            except Exception as e:
                console.print(f"[red]Error in experiment: {e}[/red]")
                continue

    # Sort by overall score
    results.sort(key=lambda r: r.overall_score, reverse=True)

    # Display results
    console.print("\n[bold green]Results[/bold green]\n")

    results_table = Table(title="Experiment Results (Sorted by Overall Score)")
    results_table.add_column("Rank", justify="right", style="dim")
    results_table.add_column("Chunk Size", justify="right")
    results_table.add_column("Overlap", justify="right")
    results_table.add_column("Top-K", justify="right")
    results_table.add_column("Faithfulness", justify="right")
    results_table.add_column("Answer Rel.", justify="right")
    results_table.add_column("Context Rel.", justify="right")
    results_table.add_column("Overall", justify="right", style="bold")
    results_table.add_column("Latency (s)", justify="right")

    for i, r in enumerate(results[:MAX_RESULTS_IN_TABLE], 1):
        # Color code overall score
        if r.overall_score >= SCORE_EXCELLENT:
            score_style = "green"
        elif r.overall_score >= SCORE_GOOD:
            score_style = "yellow"
        else:
            score_style = "red"

        results_table.add_row(
            str(i),
            str(r.chunk_size),
            str(r.chunk_overlap),
            str(r.top_k),
            f"{r.avg_faithfulness:.3f}",
            f"{r.avg_answer_relevancy:.3f}",
            f"{r.avg_context_relevancy:.3f}",
            f"[{score_style}]{r.overall_score:.3f}[/{score_style}]",
            f"{r.latency_avg:.2f}",
        )

    console.print(results_table)

    # Best configuration
    if results:
        best = results[0]
        console.print("\n[bold green]Best Configuration[/bold green]\n")
        console.print(f"  Chunk Size: [cyan]{best.chunk_size}[/cyan]")
        console.print(f"  Chunk Overlap: [cyan]{best.chunk_overlap}[/cyan]")
        console.print(f"  Top-K: [cyan]{best.top_k}[/cyan]")
        console.print(f"  Overall Score: [green]{best.overall_score:.3f}[/green]")
        console.print(f"  Avg Latency: {best.latency_avg:.2f}s")

    # Save results
    if output_path is None:
        output_path = Path("./data/evaluation_results/tuning_results.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "best_config": {
            "chunk_size": best.chunk_size,
            "chunk_overlap": best.chunk_overlap,
            "top_k": best.top_k,
            "overall_score": best.overall_score,
        } if results else None,
        "all_results": [
            {
                "chunk_size": r.chunk_size,
                "chunk_overlap": r.chunk_overlap,
                "top_k": r.top_k,
                "avg_faithfulness": r.avg_faithfulness,
                "avg_answer_relevancy": r.avg_answer_relevancy,
                "avg_context_relevancy": r.avg_context_relevancy,
                "overall_score": r.overall_score,
                "latency_avg": r.latency_avg,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to {output_path}")

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tune RAG hyperparameters for optimal retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick tuning (fewer combinations)
  python tune_hyperparameters.py --quick

  # Run full tuning
  python tune_hyperparameters.py

  # Custom configuration
  python tune_hyperparameters.py --chunk-sizes 256 512 --overlaps 50 --top-k 4 6
""",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tuning with fewer combinations",
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Chunk sizes to test",
    )
    parser.add_argument(
        "--overlaps",
        nargs="+",
        type=int,
        default=None,
        help="Chunk overlaps to test",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=None,
        help="Top-K values to test",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save results",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to env var)",
    )

    args = parser.parse_args()

    # Build config if any params specified
    config = None
    if args.chunk_sizes or args.overlaps or args.top_k:
        config = TuningConfig(
            chunk_sizes=args.chunk_sizes or [512],
            chunk_overlaps=args.overlaps or [50],
            top_k_values=args.top_k or [4],
        )

    # Ensure directories exist
    settings.ensure_directories()

    try:
        run_tuning(
            config=config,
            api_key=args.api_key,
            output_path=args.output,
            quick_mode=args.quick,
        )
        console.print("\n[bold green]✓ Tuning complete![/bold green]")

    except Exception as e:
        console.print(f"\n[red]Error during tuning: {e}[/red]")
        raise


if __name__ == "__main__":
    main()

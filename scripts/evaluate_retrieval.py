#!/usr/bin/env python3
"""Script to evaluate RAG retrieval performance."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.constants import (
    MAX_RESULTS_DISPLAYED,
    QUERY_PREVIEW_LENGTH,
    SCORE_EXCELLENT,
    SCORE_GOOD,
)

console = Console()


def run_evaluation(
    api_key: str | None = None,
    output_path: Path | None = None,
    use_sample_queries: bool = True,
) -> dict:
    """Run RAG evaluation.

    Args:
        api_key: OpenAI API key.
        output_path: Path to save results.
        use_sample_queries: Whether to use sample queries.

    Returns:
        Evaluation results dictionary.
    """
    from src.evaluation import EvaluationConfig, RAGEvaluator, create_test_queries
    from src.pipeline import RAGPipeline

    # Initialize pipeline
    console.print("[bold blue]Initializing RAG Pipeline[/bold blue]\n")

    pipeline = RAGPipeline(api_key=api_key or settings.openai_api_key)

    try:
        # Load existing vector store
        pipeline._initialize_retriever()
        console.print("[green]✓[/green] Loaded vector store")
    except FileNotFoundError:
        console.print("[red]Vector store not found! Run ingest_documents.py first.[/red]")
        return {}

    # Get test queries
    if use_sample_queries:
        test_data = create_test_queries()
        queries = [q[0] for q in test_data]
        ground_truths = [q[1] for q in test_data]
    else:
        # Could load from file here
        console.print("[red]No custom queries provided. Using sample queries.[/red]")
        test_data = create_test_queries()
        queries = [q[0] for q in test_data]
        ground_truths = [q[1] for q in test_data]

    console.print(f"\n[bold]Running evaluation on {len(queries)} queries...[/bold]\n")

    # Initialize evaluator
    eval_config = EvaluationConfig(
        metrics=["faithfulness", "answer_relevancy", "context_relevancy"],
        llm_model=settings.evaluation_llm,
    )
    evaluator = RAGEvaluator(config=eval_config, api_key=api_key or settings.openai_api_key)

    # Run evaluation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=None)
        report = evaluator.evaluate_pipeline(
            pipeline=pipeline,
            queries=queries,
            ground_truths=ground_truths,
        )
        progress.update(task, description="Evaluation complete")

    # Display results
    console.print("\n[bold green]Evaluation Results[/bold green]\n")

    # Summary table
    summary_table = Table(title="Metric Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Avg Score", justify="right")
    summary_table.add_column("Std Dev", justify="right")
    summary_table.add_column("Min", justify="right")
    summary_table.add_column("Max", justify="right")

    for metric, avg in report.avg_scores.items():
        std = report.std_scores.get(metric, 0.0)
        min_val = report.min_scores.get(metric, 0.0)
        max_val = report.max_scores.get(metric, 0.0)

        # Color code based on score
        if avg >= SCORE_EXCELLENT:
            score_style = "green"
        elif avg >= SCORE_GOOD:
            score_style = "yellow"
        else:
            score_style = "red"

        summary_table.add_row(
            metric,
            f"[{score_style}]{avg:.3f}[/{score_style}]",
            f"{std:.3f}",
            f"{min_val:.3f}",
            f"{max_val:.3f}",
        )

    console.print(summary_table)

    # Overall score
    overall_avg = sum(report.avg_scores.values()) / len(report.avg_scores) if report.avg_scores else 0
    console.print(f"\n[bold]Overall Average Score:[/bold] {overall_avg:.3f}")

    # Per-query results
    console.print("\n[bold]Per-Query Results[/bold]\n")

    query_table = Table()
    query_table.add_column("#", justify="right", style="dim")
    query_table.add_column("Query", max_width=QUERY_PREVIEW_LENGTH)
    query_table.add_column("Avg Score", justify="right")

    for result in report.results[:MAX_RESULTS_DISPLAYED]:
        query_preview = result.query[:QUERY_PREVIEW_LENGTH] + "..." if len(result.query) > QUERY_PREVIEW_LENGTH else result.query
        score_str = f"{result.avg_score:.3f}"
        query_table.add_row(str(result.sample_id), query_preview, score_str)

    if len(report.results) > MAX_RESULTS_DISPLAYED:
        query_table.add_row("...", f"({len(report.results) - MAX_RESULTS_DISPLAYED} more)", "")

    console.print(query_table)

    # Save results
    if output_path:
        report.save(output_path)
        console.print(f"\n[green]✓[/green] Results saved to {output_path}")
    else:
        # Save to default location
        default_path = Path(f"./data/evaluation_results/eval_{report.timestamp.replace(':', '-')}.json")
        report.save(default_path)
        console.print(f"\n[green]✓[/green] Results saved to {default_path}")

    return report.to_dict()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with sample queries
  python evaluate_retrieval.py

  # Save results to specific path
  python evaluate_retrieval.py --output ./results/eval.json
""",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to env var)",
    )

    args = parser.parse_args()

    # Ensure directories exist
    settings.ensure_directories()

    try:
        run_evaluation(
            api_key=args.api_key,
            output_path=args.output,
        )
        console.print("\n[bold green]✓ Evaluation complete![/bold green]")

    except Exception as e:
        console.print(f"\n[red]Error during evaluation: {e}[/red]")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive demo of the RAG system."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.constants import (
    CLAIM_PREVIEW_LENGTH,
    MAX_SOURCES_SHOWN,
    PREVIEW_LENGTH,
    SCORE_EXCELLENT,
    SCORE_GOOD,
    SEPARATOR_WIDTH,
)

console = Console()


def format_rag_result(result) -> dict:
    """Format RAG result for display.

    Args:
        result: RAG result object.

    Returns:
        Formatted result dictionary.
    """
    return {
        "query": result.query,
        "answer": result.answer,
        "sources": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("retrieval_score", 0),
            }
            for doc in result.sources
        ],
        "latency_seconds": result.latency_seconds,
        "tokens": result.generation_tokens,
    }


def display_result(result: dict, show_sources: bool = True, show_guardrails: bool = True) -> None:
    """Display RAG result with rich formatting.

    Args:
        result: RAG result dictionary.
        show_sources: Whether to show source documents.
        show_guardrails: Whether to show guardrail results.
    """
    # Display answer
    console.print("\n" + "=" * SEPARATOR_WIDTH + "\n")
    console.print(Panel(
        Markdown(result["answer"]),
        title="[bold green]Answer[/bold green]",
        border_style="green",
    ))

    # Display sources
    if show_sources and result.get("sources"):
        console.print("\n[bold cyan]Sources:[/bold cyan]")

        for i, source in enumerate(result["sources"][:MAX_SOURCES_SHOWN], 1):
            content = source["content"][:PREVIEW_LENGTH] + "..." if len(source["content"]) > PREVIEW_LENGTH else source["content"]
            score = source.get("score", 0)

            console.print(f"\n[dim]Source {i}[/dim] (relevance: {score:.3f})")
            console.print(Panel(content, border_style="dim"))

    # Display guardrail results
    if show_guardrails and result.get("guardrails"):
        guardrails = result["guardrails"]

        if guardrails.get("passed"):
            console.print("\n[green]✓ Guardrails passed[/green]")
        else:
            console.print("\n[yellow]⚠ Guardrail warnings:[/yellow]")
            for warning in guardrails.get("warnings", []):
                console.print(f"  - {warning}")

            if guardrails.get("checks", {}).get("hallucination"):
                hall_check = guardrails["checks"]["hallucination"]
                console.print(f"\n  Hallucination score: {hall_check['score']:.2f}")
                if hall_check.get("ungrounded_claims"):
                    console.print("  Ungrounded claims:")
                    for claim in hall_check["ungrounded_claims"][:MAX_SOURCES_SHOWN]:
                        console.print(f"    - {claim[:CLAIM_PREVIEW_LENGTH]}...")

    # Display metadata
    console.print(f"\n[dim]Latency: {result['latency_seconds']:.2f}s | Tokens: {result.get('tokens', 'N/A')}[/dim]")


def run_interactive_demo(
    api_key: str | None = None,
    show_sources: bool = True,
    show_guardrails: bool = True,
) -> None:
    """Run interactive RAG demo.

    Args:
        api_key: OpenAI API key.
        show_sources: Show source documents.
        show_guardrails: Show guardrail results.
    """
    from src.guardrails import GuardrailsManager, HallucinationConfig
    from src.pipeline import RAGPipeline

    # Initialize pipeline
    console.print("[bold blue]Initializing RAG System...[/bold blue]\n")

    pipeline = RAGPipeline(api_key=api_key or settings.openai_api_key)

    try:
        pipeline._initialize_retriever()
        console.print("[green]✓[/green] Vector store loaded")
    except FileNotFoundError:
        console.print("[yellow]Vector store not found. Ingesting sample documents...[/yellow]")
        num_chunks = pipeline.ingest_sample_documents()
        console.print(f"[green]✓[/green] Ingested {num_chunks} chunks")

    # Initialize guardrails
    guardrails_manager = GuardrailsManager(
        hallucination_config=HallucinationConfig(
            api_key=api_key or settings.openai_api_key,
            threshold=settings.hallucination_threshold,
        ),
        enable_hallucination_detection=show_guardrails,
    )
    console.print("[green]✓[/green] Guardrails initialized")

    # Show configuration
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Embedding: {settings.embedding_model}")
    console.print(f"  LLM: {settings.llm_model}")
    console.print(f"  Top-K: {settings.default_top_k}")
    console.print(f"  Hallucination threshold: {settings.hallucination_threshold}")

    console.print("\n" + "=" * SEPARATOR_WIDTH)
    console.print(Panel(
        "Welcome to the RAG System Demo!\n\n"
        "This system has been trained on Python programming documentation.\n"
        "Ask questions about Python basics, data types, functions, OOP, etc.\n\n"
        "Commands:\n"
        "  [cyan]quit[/cyan] - Exit the demo\n"
        "  [cyan]toggle sources[/cyan] - Toggle source display\n"
        "  [cyan]toggle guardrails[/cyan] - Toggle guardrail display",
        title="[bold]RAG Demo[/bold]",
        border_style="blue",
    ))
    console.print("=" * SEPARATOR_WIDTH + "\n")

    # Main loop
    while True:
        try:
            query = Prompt.ask("\n[bold cyan]Your question[/bold cyan]").strip()

            if not query:
                continue

            if query.lower() == "quit":
                console.print("\n[bold]Goodbye![/bold]")
                break

            if query.lower() == "toggle sources":
                show_sources = not show_sources
                console.print(f"Source display: {'on' if show_sources else 'off'}")
                continue

            if query.lower() == "toggle guardrails":
                show_guardrails = not show_guardrails
                console.print(f"Guardrail display: {'on' if show_guardrails else 'off'}")
                continue

            # Run query
            console.print("\n[dim]Processing...[/dim]")
            result = pipeline.query(query)

            # Check guardrails
            guardrails_result = None
            if show_guardrails:
                guardrails_result = guardrails_manager.check_response(
                    answer=result.answer,
                    contexts=result.sources,
                )

            # Format and display result
            formatted_result = format_rag_result(result)
            formatted_result["guardrails"] = guardrails_result

            display_result(
                formatted_result,
                show_sources=show_sources,
                show_guardrails=show_guardrails,
            )

        except KeyboardInterrupt:
            console.print("\n[bold]Goodbye![/bold]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def run_single_query(
    query: str,
    api_key: str | None = None,
    show_sources: bool = True,
    show_guardrails: bool = True,
) -> None:
    """Run a single query through the RAG system.

    Args:
        query: Query to run.
        api_key: OpenAI API key.
        show_sources: Show source documents.
        show_guardrails: Show guardrail results.
    """
    from src.guardrails import GuardrailsManager, HallucinationConfig
    from src.pipeline import RAGPipeline

    # Initialize pipeline
    pipeline = RAGPipeline(api_key=api_key or settings.openai_api_key)

    try:
        pipeline._initialize_retriever()
    except FileNotFoundError:
        console.print("[yellow]Vector store not found. Ingesting sample documents...[/yellow]")
        pipeline.ingest_sample_documents()

    # Initialize guardrails
    guardrails_manager = GuardrailsManager(
        hallucination_config=HallucinationConfig(
            api_key=api_key or settings.openai_api_key,
        ),
        enable_hallucination_detection=show_guardrails,
    )

    # Run query
    console.print(f"\n[bold]Query:[/bold] {query}\n")
    result = pipeline.query(query)

    # Check guardrails
    guardrails_result = None
    if show_guardrails:
        guardrails_result = guardrails_manager.check_response(
            answer=result.answer,
            contexts=result.sources,
        )

    # Format and display
    formatted_result = format_rag_result(result)
    formatted_result["guardrails"] = guardrails_result

    display_result(
        formatted_result,
        show_sources=show_sources,
        show_guardrails=show_guardrails,
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive RAG system demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive demo
  python run_demo.py

  # Run single query
  python run_demo.py --query "What are Python data types?"

  # Run without guardrails
  python run_demo.py --no-guardrails
""",
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Run a single query instead of interactive mode",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source documents",
    )
    parser.add_argument(
        "--no-guardrails",
        action="store_true",
        help="Don't run guardrail checks",
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
        if args.query:
            run_single_query(
                query=args.query,
                api_key=args.api_key,
                show_sources=not args.no_sources,
                show_guardrails=not args.no_guardrails,
            )
        else:
            run_interactive_demo(
                api_key=args.api_key,
                show_sources=not args.no_sources,
                show_guardrails=not args.no_guardrails,
            )

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()

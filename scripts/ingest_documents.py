#!/usr/bin/env python3
"""Script to ingest documents into the RAG system."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.constants import DOCUMENT_PREVIEW_LENGTH, MAX_DOCS_IN_TABLE
from src.ingestion import DocumentLoader, chunk_documents, load_sample_documents
from src.retrieval import VectorStoreManager

console = Console()


def ingest_sample_documents(api_key: str | None = None) -> None:
    """Ingest sample Python documentation.

    Args:
        api_key: OpenAI API key.
    """
    console.print("[bold blue]Ingesting Sample Python Documentation[/bold blue]\n")

    # Load sample documents
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading sample documents...", total=None)
        documents = load_sample_documents()
        progress.update(task, description=f"Loaded {len(documents)} documents")

    console.print(f"[green]✓[/green] Loaded {len(documents)} sample documents")

    # Chunk documents
    chunk_size = settings.default_chunk_size
    chunk_overlap = settings.default_chunk_overlap

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking documents...", total=None)
        chunks, stats = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        progress.update(task, description=f"Created {stats.total_chunks} chunks")

    console.print(f"[green]✓[/green] Created {stats.total_chunks} chunks")
    console.print(f"  - Avg chunk size: {stats.avg_chunk_size:.0f} chars")
    console.print(f"  - Size range: {stats.min_chunk_size} - {stats.max_chunk_size} chars")

    # Create embeddings and vector store
    from src.embeddings import get_default_embedder

    embedder, embeddings = get_default_embedder(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        api_key=api_key or settings.openai_api_key,
    )

    console.print(f"\n[bold]Embedding Model:[/bold] {settings.embedding_model}")
    console.print(f"[bold]Embedding Dimension:[/bold] {embedder.dimension}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating vector store...", total=None)
        vector_store = VectorStoreManager(embeddings)
        vector_store.create_vectorstore(chunks)
        progress.update(task, description="Vector store created")

    console.print(f"[green]✓[/green] Vector store created at {settings.chroma_persist_dir}")
    console.print(f"[green]✓[/green] Total documents indexed: {vector_store.document_count}")


def ingest_from_directory(
    directory: Path,
    api_key: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    recursive: bool = True,
) -> None:
    """Ingest documents from a directory.

    Args:
        directory: Directory containing documents.
        api_key: OpenAI API key.
        chunk_size: Override default chunk size.
        chunk_overlap: Override default chunk overlap.
        recursive: Search directories recursively.
    """
    console.print(f"[bold blue]Ingesting Documents from {directory}[/bold blue]\n")

    # Load documents
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading documents...", total=None)
        loader = DocumentLoader()
        documents = loader.load_from_source(directory, recursive=recursive)
        progress.update(task, description=f"Loaded {len(documents)} documents")

    console.print(f"[green]✓[/green] Loaded {len(documents)} documents")

    if not documents:
        console.print("[red]No documents found![/red]")
        return

    # Show document sources
    table = Table(title="Loaded Documents")
    table.add_column("Source", style="cyan")
    table.add_column("Content Preview", style="dim")

    for doc in documents[:MAX_DOCS_IN_TABLE]:
        source = doc.metadata.get("source", "Unknown")
        preview = doc.page_content[:DOCUMENT_PREVIEW_LENGTH] + "..." if len(doc.page_content) > DOCUMENT_PREVIEW_LENGTH else doc.page_content
        table.add_row(source, preview)

    if len(documents) > MAX_DOCS_IN_TABLE:
        table.add_row(f"... and {len(documents) - MAX_DOCS_IN_TABLE} more", "")

    console.print(table)

    # Chunk documents
    _chunk_size = chunk_size or settings.default_chunk_size
    _chunk_overlap = chunk_overlap or settings.default_chunk_overlap

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking documents...", total=None)
        chunks, stats = chunk_documents(
            documents,
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
        )
        progress.update(task, description=f"Created {stats.total_chunks} chunks")

    console.print(f"[green]✓[/green] Created {stats.total_chunks} chunks")

    # Create vector store
    from src.embeddings import get_default_embedder

    embedder, embeddings = get_default_embedder(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        api_key=api_key or settings.openai_api_key,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating vector store...", total=None)
        vector_store = VectorStoreManager(embeddings)
        vector_store.create_vectorstore(chunks)
        progress.update(task, description="Vector store created")

    console.print(f"[green]✓[/green] Vector store created at {settings.chroma_persist_dir}")
    console.print(f"[green]✓[/green] Total documents indexed: {vector_store.document_count}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest sample Python documentation
  python ingest_documents.py --sample

  # Ingest from a directory
  python ingest_documents.py --dir ./data/raw

  # Custom chunking parameters
  python ingest_documents.py --dir ./data/raw --chunk-size 256 --overlap 25
""",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample",
        action="store_true",
        help="Ingest sample Python documentation",
    )
    group.add_argument(
        "--dir",
        type=Path,
        help="Directory containing documents to ingest",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override default chunk size",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Override default chunk overlap",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search directories recursively",
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
        if args.sample:
            ingest_sample_documents(api_key=args.api_key)
        else:
            ingest_from_directory(
                directory=args.dir,
                api_key=args.api_key,
                chunk_size=args.chunk_size,
                chunk_overlap=args.overlap,
                recursive=not args.no_recursive,
            )

        console.print("\n[bold green]✓ Ingestion complete![/bold green]")

    except Exception as e:
        console.print(f"\n[red]Error during ingestion: {e}[/red]")
        raise


if __name__ == "__main__":
    main()

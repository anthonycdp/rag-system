"""Formatting utilities for RAG system."""

from langchain_core.documents import Document


def format_context_text(documents: list[Document]) -> str:
    """Format documents into a single context string.

    Args:
        documents: List of documents to format.

    Returns:
        Formatted context string with document separators.
    """
    return "\n\n".join(doc.page_content for doc in documents)


def format_documents_list(
    documents: list[Document],
    prefix: str = "[Document {i}]",
    include_source: bool = True,
) -> str:
    """Format documents as a numbered list with optional source info.

    Args:
        documents: List of documents to format.
        prefix: Format string for each document (use {i} for index).
        include_source: Whether to include source metadata.

    Returns:
        Formatted string with all documents.
    """
    parts = []
    for i, doc in enumerate(documents, 1):
        doc_prefix = prefix.format(i=i)
        if include_source and "source" in doc.metadata:
            doc_prefix += f" (Source: {doc.metadata['source']})"
        parts.append(f"{doc_prefix}\n{doc.page_content}")
    return "\n\n".join(parts)

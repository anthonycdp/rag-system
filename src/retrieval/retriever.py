"""Retriever implementations for RAG systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel

from src.constants import (
    DEFAULT_FETCH_K,
    DEFAULT_LAMBDA_MULT,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_QUERY_VARIATIONS,
)
from src.retrieval.vector_store import VectorStoreManager


class SearchType(str, Enum):
    """Types of search for retrieval."""

    SIMILARITY = "similarity"
    MMR = "mmr"
    SIMILARITY_SCORE = "similarity_score_threshold"


class RetrieverConfig(BaseModel):
    """Configuration for retrievers."""

    search_type: SearchType = SearchType.SIMILARITY
    top_k: int = DEFAULT_TOP_K
    fetch_k: int = DEFAULT_FETCH_K
    lambda_mult: float = DEFAULT_LAMBDA_MULT
    score_threshold: float = DEFAULT_SCORE_THRESHOLD


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    documents: list[Document] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    query: str = ""
    total_retrieved: int = 0


class BaseRetrieverWrapper(ABC):
    """Abstract base class for retriever wrappers."""

    @abstractmethod
    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve documents for a query.

        Args:
            query: Query string.

        Returns:
            RetrievalResult with documents and scores.
        """
        pass

    @abstractmethod
    def get_langchain_retriever(self) -> BaseRetriever:
        """Get the underlying LangChain retriever.

        Returns:
            LangChain BaseRetriever instance.
        """
        pass


class VectorStoreRetrieverWrapper(BaseRetrieverWrapper):
    """Wrapper for vector store retriever."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        config: RetrieverConfig | None = None,
    ) -> None:
        """Initialize retriever wrapper.

        Args:
            vector_store_manager: Vector store manager instance.
            config: Retriever configuration.
        """
        self.vector_store_manager = vector_store_manager
        self.config = config or RetrieverConfig()
        self._retriever: VectorStoreRetriever | None = None

    def _build_search_kwargs(self) -> dict[str, Any]:
        """Build search kwargs from config.

        Returns:
            Dictionary of search parameters.
        """
        kwargs: dict[str, Any] = {"k": self.config.top_k}

        if self.config.search_type == SearchType.MMR:
            kwargs["fetch_k"] = self.config.fetch_k
            kwargs["lambda_mult"] = self.config.lambda_mult
        elif self.config.search_type == SearchType.SIMILARITY_SCORE:
            kwargs["score_threshold"] = self.config.score_threshold

        return kwargs

    def get_langchain_retriever(self) -> BaseRetriever:
        """Get LangChain retriever instance.

        Returns:
            VectorStoreRetriever instance.
        """
        if self._retriever is None:
            search_kwargs = self._build_search_kwargs()

            self._retriever = self.vector_store_manager.get_retriever(
                search_kwargs=search_kwargs
            )

        return self._retriever

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve documents for a query.

        Args:
            query: Query string.

        Returns:
            RetrievalResult with documents and scores.
        """
        # Get results with scores
        results = self.vector_store_manager.similarity_search_with_score(
            query, k=self.config.top_k
        )

        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]

        return RetrievalResult(
            documents=documents,
            scores=scores,
            query=query,
            total_retrieved=len(documents),
        )


class MultiQueryRetrieverWrapper(BaseRetrieverWrapper):
    """Wrapper that generates multiple queries for better retrieval."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        config: RetrieverConfig | None = None,
        llm: Any = None,
    ) -> None:
        """Initialize multi-query retriever.

        Args:
            vector_store_manager: Vector store manager instance.
            config: Retriever configuration.
            llm: LLM for query generation.
        """
        self.vector_store_manager = vector_store_manager
        self.config = config or RetrieverConfig()
        self.llm = llm
        self._base_retriever: VectorStoreRetrieverWrapper | None = None

    def _generate_queries(self, original_query: str) -> list[str]:
        """Generate multiple query variations.

        Args:
            original_query: Original query string.

        Returns:
            List of query variations.

        Raises:
            ValueError: If LLM is not configured for multi-query generation.
        """
        if self.llm is None:
            raise ValueError(
                "Multi-query retrieval requires an LLM. "
                "Either provide an LLM or use standard retrieval."
            )

        prompt = f"""Generate 3 different versions of the following query to
retrieve relevant documents from a vector database. Each version should
use different wording but maintain the same semantic meaning.

Original query: {original_query}

Provide only the queries, one per line, without numbering or extra text."""

        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        queries = [q.strip() for q in content.strip().split("\n") if q.strip()]
        queries.append(original_query)

        return queries[:MAX_QUERY_VARIATIONS]

    def get_langchain_retriever(self) -> BaseRetriever:
        """Get LangChain retriever instance.

        Returns:
            BaseRetriever instance.
        """
        if self._base_retriever is None:
            self._base_retriever = VectorStoreRetrieverWrapper(
                self.vector_store_manager, self.config
            )

        return self._base_retriever.get_langchain_retriever()

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve documents using multiple query variations.

        Args:
            query: Query string.

        Returns:
            Combined RetrievalResult.
        """
        queries = self._generate_queries(query)

        all_documents: list[Document] = []
        seen_content: set[str] = set()

        for q in queries:
            result = self.vector_store_manager.similarity_search_with_score(
                q, k=self.config.top_k
            )

            for doc, score in result:
                # Deduplicate by content
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    doc.metadata["retrieval_score"] = score
                    doc.metadata["source_query"] = q
                    all_documents.append(doc)

        # Sort by score and take top-k
        all_documents.sort(
            key=lambda d: d.metadata.get("retrieval_score", 0), reverse=True
        )
        top_docs = all_documents[: self.config.top_k * 2]

        return RetrievalResult(
            documents=top_docs,
            scores=[d.metadata.get("retrieval_score", 0) for d in top_docs],
            query=query,
            total_retrieved=len(top_docs),
        )


def create_retriever(
    vector_store_manager: VectorStoreManager,
    config: RetrieverConfig | None = None,
    use_multi_query: bool = False,
    llm: Any = None,
) -> BaseRetrieverWrapper:
    """Factory function to create retriever.

    Args:
        vector_store_manager: Vector store manager instance.
        config: Retriever configuration.
        use_multi_query: Whether to use multi-query retrieval.
        llm: LLM for multi-query generation.

    Returns:
        Retriever wrapper instance.
    """
    if use_multi_query:
        return MultiQueryRetrieverWrapper(
            vector_store_manager, config, llm
        )

    return VectorStoreRetrieverWrapper(vector_store_manager, config)

"""Vector store management for RAG systems."""

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from config.settings import settings


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    persist_directory: Path = settings.chroma_persist_dir
    collection_name: str = settings.chroma_collection_name


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""

    def __init__(
        self,
        embeddings: Embeddings,
        config: VectorStoreConfig | None = None,
    ) -> None:
        """Initialize vector store manager.

        Args:
            embeddings: Embeddings model to use.
            config: Vector store configuration.
        """
        self.embeddings = embeddings
        self.config = config or VectorStoreConfig()
        self._vectorstore: Chroma | None = None

    def create_vectorstore(self, documents: list[Document]) -> Chroma:
        """Create a new vector store from documents.

        Args:
            documents: List of documents to index.

        Returns:
            Chroma vector store instance.
        """
        # Ensure directory exists
        self.config.persist_directory.mkdir(parents=True, exist_ok=True)

        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.config.persist_directory),
            collection_name=self.config.collection_name,
        )

        return self._vectorstore

    def load_vectorstore(self) -> Chroma:
        """Load existing vector store from disk.

        Returns:
            Chroma vector store instance.

        Raises:
            FileNotFoundError: If vector store doesn't exist.
        """
        if not self.config.persist_directory.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.config.persist_directory}"
            )

        self._vectorstore = Chroma(
            persist_directory=str(self.config.persist_directory),
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
        )

        return self._vectorstore

    def get_or_create_vectorstore(self, documents: list[Document] | None = None) -> Chroma:
        """Get existing vector store or create new one.

        Args:
            documents: Documents to index if creating new store.

        Returns:
            Chroma vector store instance.
        """
        if self._vectorstore is not None:
            return self._vectorstore

        try:
            return self.load_vectorstore()
        except FileNotFoundError:
            if documents:
                return self.create_vectorstore(documents)
            raise

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to existing vector store.

        Args:
            documents: Documents to add.

        Returns:
            List of document IDs.
        """
        if self._vectorstore is None:
            self._vectorstore = self.load_vectorstore()

        return self._vectorstore.add_documents(documents)

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        if self._vectorstore is not None:
            self._vectorstore.delete_collection()
            self._vectorstore = None

    def get_retriever(self, search_kwargs: dict[str, Any] | None = None) -> Any:
        """Get a retriever from the vector store.

        Args:
            search_kwargs: Search parameters (e.g., k for top-k).

        Returns:
            Vector store retriever.
        """
        if self._vectorstore is None:
            self._vectorstore = self.load_vectorstore()

        default_kwargs = {"k": settings.default_top_k}
        if search_kwargs:
            default_kwargs.update(search_kwargs)

        return self._vectorstore.as_retriever(
            search_kwargs=default_kwargs,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Perform similarity search.

        Args:
            query: Query string.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of similar documents.
        """
        if self._vectorstore is None:
            self._vectorstore = self.load_vectorstore()

        return self._vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Perform similarity search with scores.

        Args:
            query: Query string.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of (document, score) tuples.
        """
        if self._vectorstore is None:
            self._vectorstore = self.load_vectorstore()

        return self._vectorstore.similarity_search_with_score(query, k=k, filter=filter)

    @property
    def document_count(self) -> int:
        """Get number of documents in the store.

        Returns:
            Number of documents.
        """
        if self._vectorstore is None:
            self._vectorstore = self.load_vectorstore()

        # Use the collection name to get count from the underlying Chroma client
        # This avoids accessing private _collection attribute
        try:
            return len(self._vectorstore.get()["ids"])
        except (AttributeError, KeyError):
            # Fallback: estimate from similarity search
            results = self._vectorstore.similarity_search("", k=1)
            return len(results) if results else 0

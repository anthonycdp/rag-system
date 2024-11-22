"""Complete RAG Pipeline implementation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel

from config.settings import settings
from src.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
)
from src.embeddings import EmbeddingConfig, create_embedder
from src.generation import GenerationResult, ResponseGenerator
from src.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    DocumentLoader,
    chunk_documents,
    load_sample_documents,
)
from src.retrieval import (
    RetrievalResult,
    RetrieverConfig,
    VectorStoreManager,
    create_retriever,
)


class RAGPipelineConfig(BaseModel):
    """Configuration for the complete RAG pipeline."""

    embedding_provider: str = "openai"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY
    top_k: int = DEFAULT_TOP_K
    use_multi_query: bool = False
    llm_model: str = DEFAULT_LLM_MODEL
    temperature: float = DEFAULT_TEMPERATURE


@dataclass
class RAGResult:
    """Complete result from RAG pipeline."""

    query: str
    answer: str
    sources: list[Document] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    generation_tokens: int = 0
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Complete RAG pipeline combining ingestion, retrieval, and generation."""

    def __init__(
        self,
        config: RAGPipelineConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            config: Pipeline configuration.
            api_key: OpenAI API key.
        """
        self.config = config or RAGPipelineConfig()
        self.api_key = api_key or settings.openai_api_key

        # Initialize components
        self._embedder = None
        self._embeddings = None
        self._vector_store_manager: VectorStoreManager | None = None
        self._retriever = None
        self._generator: ResponseGenerator | None = None

    def _initialize_embeddings(self) -> None:
        """Initialize embedding model."""
        if self._embeddings is None:
            embedding_config = EmbeddingConfig(
                provider=self.config.embedding_provider,
                model_name=self.config.embedding_model,
                openai_api_key=self.api_key,
            )
            self._embedder = create_embedder(embedding_config)
            self._embeddings = self._embedder.get_embeddings()

    def _initialize_vector_store(self) -> VectorStoreManager:
        """Initialize vector store manager.

        Returns:
            VectorStoreManager instance.
        """
        if self._vector_store_manager is None:
            self._initialize_embeddings()
            self._vector_store_manager = VectorStoreManager(
                embeddings=self._embeddings,
            )
        return self._vector_store_manager

    def _initialize_generator(self) -> ResponseGenerator:
        """Initialize response generator.

        Returns:
            ResponseGenerator instance.
        """
        if self._generator is None:
            from src.generation import create_generator

            self._generator = create_generator(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                api_key=self.api_key,
            )
        return self._generator

    def _initialize_retriever(self) -> None:
        """Initialize retriever."""
        if self._retriever is None:
            vector_store = self._initialize_vector_store()
            retriever_config = RetrieverConfig(top_k=self.config.top_k)

            # Note: multi-query requires an LLM
            self._retriever = create_retriever(
                vector_store_manager=vector_store,
                config=retriever_config,
                use_multi_query=False,
            )

    def ingest_documents(
        self,
        documents: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> int:
        """Ingest documents into the vector store.

        Args:
            documents: Documents to ingest.
            chunk_size: Override chunk size.
            chunk_overlap: Override chunk overlap.

        Returns:
            Number of chunks created.
        """
        # Chunk documents
        strategy = ChunkingStrategy(self.config.chunking_strategy)
        chunks, stats = chunk_documents(
            documents,
            chunk_size=chunk_size or self.config.chunk_size,
            chunk_overlap=chunk_overlap or self.config.chunk_overlap,
            strategy=strategy,
        )

        # Create vector store
        vector_store = self._initialize_vector_store()
        vector_store.create_vectorstore(chunks)

        return stats.total_chunks

    def ingest_from_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
    ) -> int:
        """Load and ingest documents from a directory.

        Args:
            directory: Directory path.
            recursive: Whether to search recursively.

        Returns:
            Number of chunks created.
        """
        loader = DocumentLoader()
        documents = loader.load_from_source(Path(directory), recursive=recursive)

        return self.ingest_documents(documents)

    def ingest_sample_documents(self) -> int:
        """Ingest sample documents for demonstration.

        Returns:
            Number of chunks created.
        """
        documents = load_sample_documents()
        return self.ingest_documents(documents)

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> RAGResult:
        """Query the RAG pipeline.

        Args:
            question: User question.
            top_k: Override number of documents to retrieve.

        Returns:
            RAGResult with answer and sources.
        """
        import time

        start_time = time.time()

        # Ensure components are initialized
        self._initialize_retriever()
        generator = self._initialize_generator()

        # Retrieve documents
        if top_k:
            retriever_config = RetrieverConfig(top_k=top_k)
            vector_store = self._initialize_vector_store()
            retriever = create_retriever(vector_store, retriever_config)
            retrieval_result = retriever.retrieve(question)
        else:
            retrieval_result = self._retriever.retrieve(question)

        # Generate response
        generation_result = generator.generate(
            query=question,
            documents=retrieval_result.documents,
        )

        latency = time.time() - start_time

        return RAGResult(
            query=question,
            answer=generation_result.answer,
            sources=retrieval_result.documents,
            retrieval_scores=retrieval_result.scores,
            generation_tokens=generation_result.total_tokens,
            latency_seconds=latency,
            metadata={
                "model": self.config.llm_model,
                "top_k": top_k or self.config.top_k,
                "num_sources": len(retrieval_result.documents),
            },
        )

    async def aquery(
        self,
        question: str,
        top_k: int | None = None,
    ) -> RAGResult:
        """Asynchronously query the RAG pipeline.

        Args:
            question: User question.
            top_k: Override number of documents to retrieve.

        Returns:
            RAGResult with answer and sources.
        """
        import asyncio
        import time

        start_time = time.time()

        # Ensure components are initialized
        self._initialize_retriever()
        generator = self._initialize_generator()

        # Retrieve documents (sync for now)
        if top_k:
            retriever_config = RetrieverConfig(top_k=top_k)
            vector_store = self._initialize_vector_store()
            retriever = create_retriever(vector_store, retriever_config)
            retrieval_result = retriever.retrieve(question)
        else:
            retrieval_result = self._retriever.retrieve(question)

        # Generate response asynchronously
        generation_result = await generator.agenerate(
            query=question,
            documents=retrieval_result.documents,
        )

        latency = time.time() - start_time

        return RAGResult(
            query=question,
            answer=generation_result.answer,
            sources=retrieval_result.documents,
            retrieval_scores=retrieval_result.scores,
            generation_tokens=generation_result.total_tokens,
            latency_seconds=latency,
            metadata={
                "model": self.config.llm_model,
                "top_k": top_k or self.config.top_k,
                "num_sources": len(retrieval_result.documents),
            },
        )

    def get_retriever(self):
        """Get the underlying retriever for use with LangChain.

        Returns:
            LangChain retriever instance.
        """
        self._initialize_retriever()
        return self._retriever.get_langchain_retriever()

    def clear_vector_store(self) -> None:
        """Clear the vector store."""
        if self._vector_store_manager:
            self._vector_store_manager.delete_collection()
            self._vector_store_manager = None
            self._retriever = None


def create_pipeline(
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    top_k: int = 4,
    llm_model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> RAGPipeline:
    """Factory function to create a RAG pipeline.

    Args:
        embedding_provider: Embedding provider (openai/huggingface).
        embedding_model: Embedding model name.
        chunk_size: Document chunk size.
        chunk_overlap: Chunk overlap.
        top_k: Number of documents to retrieve.
        llm_model: LLM model for generation.
        api_key: OpenAI API key.

    Returns:
        Configured RAGPipeline instance.
    """
    config = RAGPipelineConfig(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        llm_model=llm_model,
    )

    return RAGPipeline(config=config, api_key=api_key)

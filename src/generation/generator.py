"""Response generation for RAG systems."""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)
from src.utils import create_llm, format_documents_list


class GeneratorConfig(BaseModel):
    """Configuration for response generation."""

    model_name: str = DEFAULT_LLM_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    api_key: str | None = None


@dataclass
class GenerationResult:
    """Result of generation."""

    answer: str = ""
    sources: list[Document] = field(default_factory=list)
    query: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_seconds: float = 0.0


# Default RAG prompt template
DEFAULT_RAG_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be concise but comprehensive
4. Cite specific parts of the context when possible

Context:
{context}

Question: {question}

Answer:"""


class ResponseGenerator:
    """Generates responses using LLM with retrieved context."""

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        prompt_template: str | None = None,
    ) -> None:
        """Initialize the response generator.

        Args:
            config: Generator configuration.
            prompt_template: Custom prompt template.
        """
        self.config = config or GeneratorConfig()
        self.prompt_template = prompt_template or DEFAULT_RAG_TEMPLATE
        self._llm: BaseChatModel | None = None

    def _get_llm(self) -> BaseChatModel:
        """Get or create the LLM instance.

        Returns:
            Chat model instance.
        """
        if self._llm is None:
            self._llm = create_llm(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key,
            )
        return self._llm

    def _format_context(self, documents: list[Document]) -> str:
        """Format documents into context string.

        Args:
            documents: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        return format_documents_list(documents)

    def generate(
        self,
        query: str,
        documents: list[Document],
    ) -> GenerationResult:
        """Generate a response given query and retrieved documents.

        Args:
            query: User query.
            documents: Retrieved documents.

        Returns:
            GenerationResult with answer and metadata.
        """
        import time

        start_time = time.time()

        llm = self._get_llm()
        context = self._format_context(documents)

        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = prompt | llm

        # Invoke the chain
        response = chain.invoke({
            "context": context,
            "question": query,
        })

        # Extract usage metadata if available
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata

        latency = time.time() - start_time

        return GenerationResult(
            answer=response.content,
            sources=documents,
            query=query,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            model=self.config.model_name,
            latency_seconds=latency,
        )

    async def agenerate(
        self,
        query: str,
        documents: list[Document],
    ) -> GenerationResult:
        """Asynchronously generate a response.

        Args:
            query: User query.
            documents: Retrieved documents.

        Returns:
            GenerationResult with answer and metadata.
        """
        import time

        start_time = time.time()

        llm = self._get_llm()
        context = self._format_context(documents)

        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = prompt | llm

        response = await chain.ainvoke({
            "context": context,
            "question": query,
        })

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata

        latency = time.time() - start_time

        return GenerationResult(
            answer=response.content,
            sources=documents,
            query=query,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            model=self.config.model_name,
            latency_seconds=latency,
        )


def create_generator(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: str | None = None,
    prompt_template: str | None = None,
) -> ResponseGenerator:
    """Factory function to create a response generator.

    Args:
        model_name: LLM model name.
        temperature: Temperature for generation.
        api_key: OpenAI API key.
        prompt_template: Custom prompt template.

    Returns:
        ResponseGenerator instance.
    """
    config = GeneratorConfig(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
    )

    return ResponseGenerator(config=config, prompt_template=prompt_template)

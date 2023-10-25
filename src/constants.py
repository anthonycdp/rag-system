"""Constants and default values for the RAG system."""


# =============================================================================
# Embedding Defaults
# =============================================================================
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# =============================================================================
# Chunking Defaults
# =============================================================================
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_CHUNKING_STRATEGY = "recursive"

# Separator patterns
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
SENTENCE_AWARE_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


# =============================================================================
# Retrieval Defaults
# =============================================================================
DEFAULT_TOP_K = 4
DEFAULT_FETCH_K = 20
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_LAMBDA_MULT = 0.5

# Multi-query retrieval
MAX_QUERY_VARIATIONS = 4


# =============================================================================
# Generation Defaults
# =============================================================================
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0


# =============================================================================
# LLM Defaults
# =============================================================================
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 1536


# =============================================================================
# Hallucination Detection
# =============================================================================
DEFAULT_HALLUCINATION_THRESHOLD = 0.7
MIN_CLAIM_LENGTH = 10
MAX_DISPLAY_CLAIMS = 3
MAX_CLAIMS_SHOWN = 2
MAX_RESPONSE_LENGTH = 2000
MAX_CONTEXT_RATIO = 2
CONTENT_PREVIEW_LENGTH = 100


# =============================================================================
# Evaluation Weights
# =============================================================================
FAITHFULNESS_WEIGHT = 0.4
ANSWER_RELEVANCY_WEIGHT = 0.3
CONTEXT_RELEVANCY_WEIGHT = 0.3


# =============================================================================
# Display/UI Constants
# =============================================================================
SEPARATOR_WIDTH = 60
PREVIEW_LENGTH = 200
QUERY_PREVIEW_LENGTH = 40
MAX_SOURCES_SHOWN = 3
MAX_CLAIMS_SHOWN = 2
MAX_RESULTS_DISPLAYED = 15
QUICK_MODE_QUERY_COUNT = 3
DOCUMENT_PREVIEW_LENGTH = 50
MAX_DOCS_IN_TABLE = 10
MAX_RESULTS_IN_TABLE = 10
MAX_TUNE_RESULTS_IN_TABLE = 10
CLAIM_PREVIEW_LENGTH = 100


# =============================================================================
# Score Thresholds (for display color-coding)
# =============================================================================
SCORE_EXCELLENT = 0.8
SCORE_GOOD = 0.6


# =============================================================================
# Embedding Dimensions (known models)
# =============================================================================
EMBEDDING_DIMENSIONS: dict[str, int] = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # HuggingFace
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/multi-qa-MiniLM-L6-dot-v1": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}

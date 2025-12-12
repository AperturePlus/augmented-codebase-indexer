"""
Embedding client module for Project ACI.

Provides async HTTP client for generating embeddings via API calls.
Supports batch processing and exponential backoff retry logic.
"""

from .client import OpenAIEmbeddingClient, create_embedding_client
from .errors import (
    BatchSizeError,
    EmbeddingClientError,
    NonRetryableError,
    RetryableError,
)
from .interface import EmbeddingClientInterface
from .retry import RetryConfig

__all__ = [
    "EmbeddingClientInterface",
    "OpenAIEmbeddingClient",
    "create_embedding_client",
    "EmbeddingClientError",
    "RetryableError",
    "NonRetryableError",
    "BatchSizeError",
    "RetryConfig",
]

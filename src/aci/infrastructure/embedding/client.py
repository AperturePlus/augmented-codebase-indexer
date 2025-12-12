"""OpenAI-compatible embedding client implementation."""

import logging
from typing import List, Optional

import httpx

from .errors import (
    BatchSizeError,
    NonRetryableError,
    RetryableError,
)
from .interface import EmbeddingClientInterface
from .response_parser import is_token_limit_error, parse_embedding_response
from .retry import RetryConfig, with_retry

logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient(EmbeddingClientInterface):
    """
    Embedding client for OpenAI-compatible APIs.

    Supports batch processing with configurable batch size and
    exponential backoff retry for rate limits and transient errors.
    Uses connection pooling for efficient HTTP connections (Req 6.5).
    
    Batch Fallback Behavior:
        When enabled (default), the client automatically handles token limit
        errors (HTTP 413) by reducing the batch size and retrying. This allows
        successful embedding generation even when some batches exceed the API's
        token limit. The batch size is halved on each retry until it reaches
        the configured minimum. If a single item exceeds the limit, a
        NonRetryableError is raised.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        batch_size: int = 100,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
        encoding_format: str = 'float'
    ):
        """
        Initialize the embedding client.

        Args:
            api_url: Base URL for the embedding API
            api_key: API key for authentication
            model: Model name to use for embeddings
            dimension: Expected embedding dimension
            batch_size: Maximum texts per API call
            timeout: Request timeout in seconds
            retry_config: Configuration for retry behavior
            encoding_format: Encoding format for embeddings
        """
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._timeout = timeout
        self._retry_config = retry_config or RetryConfig()
        self._encoding_format = encoding_format

        # Connection pooling - reuse HTTP client across requests (Req 6.5)
        self._client: Optional[httpx.AsyncClient] = None

        # Validate batch_size
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    @property
    def batch_size(self) -> int:
        """Return the configured batch size."""
        return self._batch_size

    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Automatically splits large batches into smaller chunks based on
        the configured batch_size and processes them sequentially.
        
        If batch fallback is enabled and a token limit error occurs,
        the batch size will be reduced and the failed batch retried.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors in the same order as input texts

        Raises:
            EmbeddingClientError: If embedding generation fails after retries
            NonRetryableError: If a single item exceeds token limits
        """
        if not texts:
            return []

        return await self._embed_with_fallback(texts, self._batch_size)

    async def _embed_with_fallback(
        self, texts: List[str], current_batch_size: int
    ) -> List[List[float]]:
        """
        Embed texts with automatic batch size fallback on token limit errors.
        
        When a BatchSizeError occurs, reduces the batch size by half and retries.
        Continues with the reduced batch size for remaining items.
        
        Args:
            texts: List of texts to embed
            current_batch_size: Current batch size to use
            
        Returns:
            List of embedding vectors in the same order as input texts
            
        Raises:
            NonRetryableError: If a single item exceeds token limits
            EmbeddingClientError: If embedding fails after all retries
        """
        all_embeddings: List[List[float]] = []
        config = self._retry_config
        
        i = 0
        while i < len(texts):
            batch = texts[i : i + current_batch_size]
            
            try:
                batch_embeddings = await self._embed_single_batch(batch)
                all_embeddings.extend(batch_embeddings)
                i += len(batch)
            except BatchSizeError as e:
                # Check if fallback is enabled
                if not config.enable_batch_fallback:
                    raise NonRetryableError(
                        f"Token limit exceeded and batch fallback is disabled: {e}"
                    ) from e
                
                # Check if we can reduce batch size further
                if current_batch_size <= config.min_batch_size:
                    # Single item exceeds token limit
                    logger.error(
                        f"Item at index {i} exceeds token limit, "
                        f"cannot reduce batch further (min_batch_size={config.min_batch_size})"
                    )
                    raise NonRetryableError(
                        f"Single item at index {i} exceeds token limit: {e}"
                    ) from e
                
                # Reduce batch size and retry
                new_batch_size = max(config.min_batch_size, current_batch_size // 2)
                logger.warning(
                    f"Token limit exceeded, reducing batch size from "
                    f"{current_batch_size} to {new_batch_size}"
                )
                current_batch_size = new_batch_size
                # Don't increment i - retry the same batch with smaller size
        
        return all_embeddings

    async def _embed_single_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a single batch of texts with retry logic.

        Args:
            texts: Batch of texts (size <= batch_size)

        Returns:
            List of embedding vectors
            
        Raises:
            BatchSizeError: If token limit is exceeded (for fallback handling)
            RetryableError: For transient errors after retries exhausted
            NonRetryableError: For non-recoverable errors
        """
        return await with_retry(lambda: self._call_api(texts), self._retry_config)

    async def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Make the actual API call to generate embeddings.

        Args:
            texts: Batch of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            BatchSizeError: For token limit exceeded errors
            RetryableError: For rate limits and transient errors
            NonRetryableError: For auth failures and invalid requests
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        batch_size = len(texts)

        payload = {
            "input": texts,
            "model": self._model,
            "encoding_format": self._encoding_format
        }

        # Use pooled client for connection reuse (Req 6.5)
        client = await self._get_client()
        try:
            response = await client.post(
                self._api_url,
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                return parse_embedding_response(response.json(), len(texts), self._dimension)
            elif is_token_limit_error(response.status_code, response.text):
                # Token limit exceeded - can retry with smaller batch
                raise BatchSizeError(
                    f"Token limit exceeded: {response.status_code} - {response.text} "
                    f"(url={self._api_url}, model={self._model}, batch={batch_size})"
                )
            elif response.status_code == 429:
                # Rate limited - retryable
                raise RetryableError(f"Rate limited: {response.status_code} - {response.text}")
            elif response.status_code in (500, 502, 503, 504):
                # Server errors - retryable
                raise RetryableError(f"Server error: {response.status_code} - {response.text}")
            elif response.status_code in (401, 403):
                # Auth errors - not retryable
                raise NonRetryableError(
                    f"Authentication failed: {response.status_code} - {response.text} "
                    f"(url={self._api_url}, model={self._model}, batch={batch_size})"
                )
            else:
                # Other errors - not retryable
                raise NonRetryableError(
                    f"API error: {response.status_code} - {response.text} "
                    f"(url={self._api_url}, model={self._model}, batch={batch_size})"
                )

        except httpx.TimeoutException as e:
            raise RetryableError(f"Request timeout: {e}")
        except httpx.ConnectError as e:
            raise RetryableError(f"Connection error: {e}")
        except httpx.RequestError as e:
            raise RetryableError(f"Request error: {e}")


def create_embedding_client(
    api_url: str,
    api_key: str,
    model: str = "text-embedding-3-small",
    dimension: int = 1536,
    batch_size: int = 100,
    timeout: float = 30.0,
    max_retries: int = 3,
    enable_batch_fallback: bool = True,
    min_batch_size: int = 1,
) -> EmbeddingClientInterface:
    """
    Factory function to create an embedding client.

    Args:
        api_url: Base URL for the embedding API
        api_key: API key for authentication
        model: Model name to use for embeddings
        dimension: Expected embedding dimension
        batch_size: Maximum texts per API call
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        enable_batch_fallback: Whether to reduce batch size on token limit errors
        min_batch_size: Minimum batch size when reducing due to token limits

    Returns:
        Configured EmbeddingClientInterface instance
    """
    retry_config = RetryConfig(
        max_retries=max_retries,
        enable_batch_fallback=enable_batch_fallback,
        min_batch_size=min_batch_size,
    )

    return OpenAIEmbeddingClient(
        api_url=api_url,
        api_key=api_key,
        model=model,
        dimension=dimension,
        batch_size=batch_size,
        timeout=timeout,
        retry_config=retry_config,
    )

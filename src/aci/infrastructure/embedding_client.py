"""
Embedding Client for Project ACI.

Provides async HTTP client for generating embeddings via API calls.
Supports batch processing and exponential backoff retry logic.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class EmbeddingClientError(Exception):
    """Base exception for embedding client errors."""

    pass


class RetryableError(EmbeddingClientError):
    """Error that can be retried (rate limits, temporary failures)."""

    pass


class NonRetryableError(EmbeddingClientError):
    """Error that should not be retried (auth failures, invalid requests)."""

    pass


class BatchSizeError(EmbeddingClientError):
    """Error indicating batch size is too large (token limit exceeded).
    
    This error is raised when the embedding API returns a 413 status code
    or a 400 status code with a token limit error message. The embedding
    client will attempt to reduce the batch size and retry when this error
    is encountered.
    """

    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts for transient errors.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        exponential_base: Base for exponential backoff calculation.
        enable_batch_fallback: Whether to reduce batch size on token limit errors.
        min_batch_size: Minimum batch size when reducing due to token limits.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    enable_batch_fallback: bool = True
    min_batch_size: int = 1


class EmbeddingClientInterface(ABC):
    """Abstract interface for embedding clients."""

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one per input text
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        pass


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
            retry_config: Configuration for retry behavior, including:
                - max_retries: Maximum retry attempts for transient errors
                - enable_batch_fallback: Whether to reduce batch size on token limits
                - min_batch_size: Minimum batch size during fallback
        """
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._timeout = timeout
        self._retry_config = retry_config or RetryConfig()

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
        return await self._with_retry(lambda: self._call_api(texts))

    def _is_token_limit_error(self, status_code: int, response_text: str) -> bool:
        """
        Check if the error response indicates a token limit exceeded error.
        
        Detects token limit errors from various API providers:
        - HTTP 413 status code (Request Entity Too Large)
        - HTTP 400 with token limit patterns in response body
        - SiliconFlow error code 20042
        
        Args:
            status_code: HTTP status code from the response
            response_text: Response body text
            
        Returns:
            True if this is a token limit error, False otherwise
        """
        # HTTP 413 is always a token limit error
        if status_code == 413:
            return True
        
        # Check for token limit patterns in 400 responses
        if status_code == 400:
            response_lower = response_text.lower()
            # Check for common token limit error patterns
            if "token" in response_lower:
                if any(pattern in response_lower for pattern in [
                    "limit", "8192", "exceed", "maximum", "many"
                ]):
                    return True
            # Check for SiliconFlow specific error code
            if "20042" in response_text:
                return True
        
        return False

    async def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Make the actual API call to generate embeddings.

        Args:
            texts: Batch of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            BatchSizeError: For token limit exceeded errors (413 or detected patterns)
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
                return self._parse_response(response.json(), len(texts))
            elif self._is_token_limit_error(response.status_code, response.text):
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

    def _parse_response(self, response_data: dict, expected_count: int) -> List[List[float]]:
        """
        Parse the API response and extract embeddings.

        Args:
            response_data: JSON response from the API
            expected_count: Expected number of embeddings

        Returns:
            List of embedding vectors

        Raises:
            NonRetryableError: If response format is invalid
        """
        try:
            data = response_data.get("data", [])

            if len(data) != expected_count:
                raise NonRetryableError(f"Expected {expected_count} embeddings, got {len(data)}")

            # Sort by index to ensure correct order
            sorted_data = sorted(data, key=lambda x: x.get("index", 0))

            embeddings = []
            for item in sorted_data:
                embedding = item.get("embedding", [])
                if len(embedding) != self._dimension:
                    logger.warning(
                        f"Embedding dimension mismatch: expected {self._dimension}, "
                        f"got {len(embedding)}"
                    )
                embeddings.append(embedding)

            return embeddings

        except (KeyError, TypeError) as e:
            raise NonRetryableError(f"Invalid response format: {e}")

    async def _with_retry(self, operation) -> List[List[float]]:
        """
        Execute an operation with exponential backoff retry.

        Args:
            operation: Async callable to execute

        Returns:
            Result of the operation

        Raises:
            EmbeddingClientError: If all retries are exhausted
        """
        config = self._retry_config
        last_error: Optional[Exception] = None

        for attempt in range(config.max_retries + 1):
            try:
                return await operation()
            except NonRetryableError:
                # Don't retry non-retryable errors
                raise
            except RetryableError as e:
                last_error = e

                if attempt == config.max_retries:
                    # Last attempt failed
                    logger.error(f"All {config.max_retries + 1} attempts failed. Last error: {e}")
                    raise EmbeddingClientError(
                        f"Failed after {config.max_retries + 1} attempts: {e}"
                    ) from e

                # Calculate delay with exponential backoff
                delay = min(
                    config.base_delay * (config.exponential_base**attempt),
                    config.max_delay,
                )

                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")

                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        raise EmbeddingClientError(f"Unexpected retry loop exit: {last_error}")


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
        enable_batch_fallback: Whether to reduce batch size on token limit errors.
            When True, the client will automatically reduce batch size and retry
            when encountering HTTP 413 or token limit errors.
        min_batch_size: Minimum batch size when reducing due to token limits.
            If a single item exceeds the token limit, a NonRetryableError is raised.

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

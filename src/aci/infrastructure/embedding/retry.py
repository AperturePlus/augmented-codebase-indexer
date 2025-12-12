"""Retry configuration and logic for embedding client."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from .errors import EmbeddingClientError, NonRetryableError, RetryableError

logger = logging.getLogger(__name__)


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


async def with_retry(operation, config: RetryConfig):
    """
    Execute an operation with exponential backoff retry.

    Args:
        operation: Async callable to execute
        config: Retry configuration

    Returns:
        Result of the operation

    Raises:
        EmbeddingClientError: If all retries are exhausted
    """
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

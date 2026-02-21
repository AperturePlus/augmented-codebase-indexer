"""Response parsing logic for embedding API."""

import logging

from .errors import NonRetryableError

logger = logging.getLogger(__name__)


def parse_embedding_response(
    response_data: dict, expected_count: int, expected_dimension: int
) -> list[list[float]]:
    """
    Parse the API response and extract embeddings.

    Args:
        response_data: JSON response from the API
        expected_count: Expected number of embeddings
        expected_dimension: Expected embedding dimension

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
            if len(embedding) != expected_dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected {expected_dimension}, "
                    f"got {len(embedding)}"
                )
            embeddings.append(embedding)

        return embeddings

    except (KeyError, TypeError) as e:
        raise NonRetryableError(f"Invalid response format: {e}") from e


def is_token_limit_error(status_code: int, response_text: str) -> bool:
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

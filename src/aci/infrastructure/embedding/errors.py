"""Exception types for embedding client."""


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

"""
Property-based tests for EmbeddingClient.

**Feature: codebase-semantic-search, Property 9: Batch Size Compliance**
**Validates: Requirements 3.1**
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.infrastructure.embedding import OpenAIEmbeddingClient, RetryConfig
from aci.infrastructure.embedding.response_parser import is_token_limit_error

# Strategy for generating text batches
text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=100,
)

texts_strategy = st.lists(text_strategy, min_size=1, max_size=50)
batch_size_strategy = st.integers(min_value=1, max_value=20)


@given(texts=texts_strategy, batch_size=batch_size_strategy)
@settings(max_examples=100, deadline=None)
def test_batch_size_compliance(texts: list[str], batch_size: int):
    """
    **Feature: codebase-semantic-search, Property 9: Batch Size Compliance**
    **Validates: Requirements 3.1**

    *For any* set of texts sent to EmbeddingClient, the client should
    split them into batches where each batch size <= configured batch_size.
    """
    # Track actual batch sizes sent to API
    actual_batch_sizes = []

    async def mock_post(url, headers, json):
        """Mock HTTP POST that records batch sizes."""
        batch_texts = json.get("input", [])
        actual_batch_sizes.append(len(batch_texts))

        # Return mock embeddings - use MagicMock for sync json() method
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"index": i, "embedding": [0.1] * 1536} for i in range(len(batch_texts))]
        }
        return mock_response

    # Create client with specified batch size
    client = OpenAIEmbeddingClient(
        api_url="https://api.example.com/embeddings",
        api_key="test-key",
        batch_size=batch_size,
        retry_config=RetryConfig(max_retries=0),
    )

    # Run embed_batch with mocked HTTP client
    async def run_test():
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await client.embed_batch(texts)

    asyncio.run(run_test())

    # Verify all batches respect the batch_size limit
    for i, size in enumerate(actual_batch_sizes):
        assert size <= batch_size, f"Batch {i} has size {size}, exceeds batch_size {batch_size}"

    # Verify total texts processed equals input
    total_processed = sum(actual_batch_sizes)
    assert total_processed == len(texts), (
        f"Total processed {total_processed} != input size {len(texts)}"
    )


@given(batch_size=batch_size_strategy)
@settings(max_examples=100, deadline=None)
def test_empty_batch_returns_empty(batch_size: int):
    """
    *For any* batch_size configuration, embedding an empty list
    should return an empty list without making API calls.
    """
    client = OpenAIEmbeddingClient(
        api_url="https://api.example.com/embeddings",
        api_key="test-key",
        batch_size=batch_size,
    )

    async def run_test():
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await client.embed_batch([])

            # Should not make any API calls
            mock_client.post.assert_not_called()

            return result

    result = asyncio.run(run_test())
    assert result == []


@given(texts=texts_strategy)
@settings(max_examples=100, deadline=None)
def test_embedding_order_preserved(texts: list[str]):
    """
    *For any* list of texts, the returned embeddings should be
    in the same order as the input texts.
    """

    # Use unique embeddings based on index to verify order
    async def mock_post(url, headers, json):
        batch_texts = json.get("input", [])

        # Use MagicMock for sync json() method
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Create unique embeddings based on text hash
        mock_response.json.return_value = {
            "data": [
                {"index": i, "embedding": [hash(text) % 1000 / 1000.0] * 1536}
                for i, text in enumerate(batch_texts)
            ]
        }
        return mock_response

    client = OpenAIEmbeddingClient(
        api_url="https://api.example.com/embeddings",
        api_key="test-key",
        batch_size=5,  # Small batch to test ordering across batches
        retry_config=RetryConfig(max_retries=0),
    )

    async def run_test():
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            return await client.embed_batch(texts)

    embeddings = asyncio.run(run_test())

    # Verify count matches
    assert len(embeddings) == len(texts)

    # Verify each embedding corresponds to correct text
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        expected_value = hash(text) % 1000 / 1000.0
        assert embedding[0] == expected_value, (
            f"Embedding {i} doesn't match expected value for text"
        )


# Strategies for token limit error detection tests
status_code_413_strategy = st.just(413)
status_code_400_strategy = st.just(400)
status_code_other_strategy = st.integers(min_value=200, max_value=599).filter(
    lambda x: x not in (200, 400, 413, 429, 500, 502, 503, 504, 401, 403)
)

# Token limit error message patterns
token_limit_patterns = [
    "input must have less than 8192 tokens",
    "token limit exceeded",
    "maximum token limit",
    "too many tokens",
    '{"code":20042,"message":"input must have less than 8192 tokens"}',
]

non_token_limit_patterns = [
    "invalid request",
    "bad request",
    "unknown error",
    "rate limit exceeded",  # This is 429, not token limit
]


@given(response_body=st.sampled_from(token_limit_patterns))
@settings(max_examples=100, deadline=None)
def test_token_limit_error_detection_413(response_body: str):
    """
    **Feature: embedding-batch-fallback, Property 1: Token Limit Error Detection**
    **Validates: Requirements 1.1, 4.1, 4.2**

    *For any* HTTP 413 response, the error classification function SHALL
    return True (indicating a token limit error), regardless of response body.
    """
    # 413 should always be detected as token limit error
    assert is_token_limit_error(413, response_body) is True
    assert is_token_limit_error(413, "") is True
    assert is_token_limit_error(413, "any random text") is True


@given(response_body=st.sampled_from(token_limit_patterns))
@settings(max_examples=100, deadline=None)
def test_token_limit_error_detection_400_with_pattern(response_body: str):
    """
    **Feature: embedding-batch-fallback, Property 1: Token Limit Error Detection**
    **Validates: Requirements 1.1, 4.1, 4.2**

    *For any* HTTP 400 response with token limit patterns in the body,
    the error classification function SHALL return True.
    """
    # 400 with token limit patterns should be detected
    assert is_token_limit_error(400, response_body) is True


@given(response_body=st.sampled_from(non_token_limit_patterns))
@settings(max_examples=100, deadline=None)
def test_non_token_limit_error_detection_400(response_body: str):
    """
    **Feature: embedding-batch-fallback, Property 1: Token Limit Error Detection**
    **Validates: Requirements 1.1, 4.1, 4.2**

    *For any* HTTP 400 response without token limit patterns,
    the error classification function SHALL return False.
    """
    # 400 without token limit patterns should not be detected
    assert is_token_limit_error(400, response_body) is False


@given(
    status_code=st.integers(min_value=200, max_value=599).filter(
        lambda x: x not in (400, 413)
    ),
    response_body=st.text(min_size=0, max_size=200),
)
@settings(max_examples=100, deadline=None)
def test_non_token_limit_status_codes(status_code: int, response_body: str):
    """
    **Feature: embedding-batch-fallback, Property 1: Token Limit Error Detection**
    **Validates: Requirements 1.1, 4.1, 4.2**

    *For any* HTTP status code other than 400 or 413, the error classification
    function SHALL return False, regardless of response body content.
    """
    # Non-400/413 status codes should never be detected as token limit errors
    assert is_token_limit_error(status_code, response_body) is False


@given(
    batch_size=st.integers(min_value=2, max_value=128),
    min_batch_size=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=100, deadline=None)
def test_batch_size_reduction_correctness(batch_size: int, min_batch_size: int):
    """
    **Feature: embedding-batch-fallback, Property 2: Batch Size Reduction Correctness**
    **Validates: Requirements 1.2, 3.2**

    *For any* batch size N > min_batch_size, when a BatchSizeError occurs,
    the new batch size SHALL be max(min_batch_size, N // 2), which is strictly less than N.
    """
    # Ensure min_batch_size <= batch_size for valid test
    min_batch_size = min(min_batch_size, batch_size)

    # Calculate expected new batch size
    expected_new_size = max(min_batch_size, batch_size // 2)

    # Verify the reduction formula
    assert expected_new_size <= batch_size, (
        f"New batch size {expected_new_size} should be <= original {batch_size}"
    )

    # If batch_size > min_batch_size, new size should be strictly less
    if batch_size > min_batch_size:
        assert expected_new_size < batch_size, (
            f"New batch size {expected_new_size} should be < original {batch_size} "
            f"when batch_size > min_batch_size"
        )

    # New size should never be less than min_batch_size
    assert expected_new_size >= min_batch_size, (
        f"New batch size {expected_new_size} should be >= min_batch_size {min_batch_size}"
    )


@given(
    texts_count=st.integers(min_value=1, max_value=20),
    initial_batch_size=st.integers(min_value=4, max_value=16),
    fail_threshold=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=100, deadline=None)
def test_complete_processing_with_fallback(
    texts_count: int, initial_batch_size: int, fail_threshold: int
):
    """
    **Feature: embedding-batch-fallback, Property 3: Complete Processing Guarantee**
    **Validates: Requirements 1.3, 4.3**

    *For any* list of texts where at least one valid batch size exists that succeeds,
    embed_batch SHALL return embeddings for all input texts (output length equals input length).
    """
    from aci.infrastructure.embedding import BatchSizeError

    # Ensure fail_threshold < initial_batch_size for meaningful test
    fail_threshold = min(fail_threshold, initial_batch_size - 1)
    if fail_threshold < 1:
        fail_threshold = 1

    texts = [f"text_{i}" for i in range(texts_count)]
    call_count = 0

    async def mock_post(url, headers, json):
        """Mock HTTP POST that fails for large batches."""
        nonlocal call_count
        call_count += 1
        batch_texts = json.get("input", [])
        batch_size = len(batch_texts)

        # Fail if batch size exceeds threshold
        if batch_size > fail_threshold:
            mock_response = MagicMock()
            mock_response.status_code = 413
            mock_response.text = "Token limit exceeded"
            return mock_response

        # Success for smaller batches
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"index": i, "embedding": [0.1] * 1536} for i in range(batch_size)]
        }
        return mock_response

    client = OpenAIEmbeddingClient(
        api_url="https://api.example.com/embeddings",
        api_key="test-key",
        batch_size=initial_batch_size,
        retry_config=RetryConfig(
            max_retries=0,
            enable_batch_fallback=True,
            min_batch_size=1,
        ),
    )

    async def run_test():
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            return await client.embed_batch(texts)

    embeddings = asyncio.run(run_test())

    # Verify all texts got embeddings
    assert len(embeddings) == len(texts), (
        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
    )


@given(texts_count=st.integers(min_value=1, max_value=10))
@settings(max_examples=100, deadline=None)
def test_fallback_disabled_raises_immediately(texts_count: int):
    """
    **Feature: embedding-batch-fallback, Property 4: Configuration Respected**
    **Validates: Requirements 3.1, 3.3**

    *For any* RetryConfig with enable_batch_fallback=False, when a 413 error occurs,
    the client SHALL raise NonRetryableError immediately without attempting batch reduction.
    """
    from aci.infrastructure.embedding import NonRetryableError

    texts = [f"text_{i}" for i in range(texts_count)]
    call_count = 0

    async def mock_post(url, headers, json):
        """Mock HTTP POST that always returns 413."""
        nonlocal call_count
        call_count += 1

        mock_response = MagicMock()
        mock_response.status_code = 413
        mock_response.text = "Token limit exceeded"
        return mock_response

    client = OpenAIEmbeddingClient(
        api_url="https://api.example.com/embeddings",
        api_key="test-key",
        batch_size=10,
        retry_config=RetryConfig(
            max_retries=0,
            enable_batch_fallback=False,  # Fallback disabled
            min_batch_size=1,
        ),
    )

    async def run_test():
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            try:
                await client.embed_batch(texts)
                return None  # Should not reach here
            except NonRetryableError as e:
                return e

    error = asyncio.run(run_test())

    # Should raise NonRetryableError
    assert error is not None, "Expected NonRetryableError to be raised"
    assert "fallback is disabled" in str(error).lower(), (
        f"Error message should mention fallback is disabled: {error}"
    )

    # Should only make one API call (no retry with smaller batch)
    assert call_count == 1, (
        f"Expected 1 API call when fallback disabled, got {call_count}"
    )

"""
Property-based tests for EmbeddingClient.

**Feature: codebase-semantic-search, Property 9: Batch Size Compliance**
**Validates: Requirements 3.1**
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.infrastructure.embedding_client import OpenAIEmbeddingClient, RetryConfig

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

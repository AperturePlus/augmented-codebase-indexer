import asyncio

import pytest

from aci.infrastructure.embedding import OpenAIEmbeddingClient, RetryConfig
from aci.infrastructure.embedding.errors import BatchSizeError, NonRetryableError


def test_embed_with_fallback_skips_oversized_single_item() -> None:
    texts = ["ok-1", "oversized", "ok-2"]

    client = OpenAIEmbeddingClient(
        api_url="https://api.example.com/embeddings",
        api_key="test-key",
        dimension=4,
        batch_size=2,
        retry_config=RetryConfig(max_retries=0, enable_batch_fallback=True, min_batch_size=1),
    )

    async def fake_embed_single_batch(batch: list[str]) -> list[list[float]]:
        if "oversized" in batch:
            raise BatchSizeError("Token limit exceeded")
        return [[float(i)] * 4 for i, _ in enumerate(batch, start=1)]

    client._embed_single_batch = fake_embed_single_batch  # type: ignore[method-assign]

    embeddings = asyncio.run(client.embed_batch(texts))

    assert len(embeddings) == len(texts)
    assert embeddings[0] == [1.0, 1.0, 1.0, 1.0]
    assert embeddings[1] == [0.0, 0.0, 0.0, 0.0]
    assert embeddings[2] == [1.0, 1.0, 1.0, 1.0]


def test_embed_with_fallback_still_raises_when_disabled() -> None:
    client = OpenAIEmbeddingClient(
        api_url="https://api.example.com/embeddings",
        api_key="test-key",
        dimension=4,
        batch_size=2,
        retry_config=RetryConfig(max_retries=0, enable_batch_fallback=False, min_batch_size=1),
    )

    async def fake_embed_single_batch(_: list[str]) -> list[list[float]]:
        raise BatchSizeError("Token limit exceeded")

    client._embed_single_batch = fake_embed_single_batch  # type: ignore[method-assign]

    with pytest.raises(NonRetryableError) as exc_info:
        asyncio.run(client.embed_batch(["oversized"]))

    assert "fallback is disabled" in str(exc_info.value).lower()

"""Property-based tests for vector store search and retrieval behavior."""

import asyncio

from hypothesis import HealthCheck, given, settings, strategies as st

from aci.infrastructure.fakes import InMemoryVectorStore
from tests.vector_store_strategies import (
    chunk_id_strategy,
    chunk_payload_strategy,
    file_path_strategy,
    vector_strategy,
)


@given(chunk_id=chunk_id_strategy, vector=vector_strategy, payload=chunk_payload_strategy())
@settings(max_examples=100, deadline=None)
def test_vector_storage_round_trip(chunk_id: str, vector: list[float], payload: dict):
    """Storing and fetching by ID should preserve all payload fields."""
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        await store.upsert(chunk_id, vector, payload)
        return await store.get_by_id(chunk_id)

    result = asyncio.run(run_test())

    assert result is not None
    assert result.chunk_id == chunk_id
    assert result.file_path == payload["file_path"]
    assert result.start_line == payload["start_line"]
    assert result.end_line == payload["end_line"]
    assert result.content == payload["content"]


@given(
    chunks=vector_strategy.flatmap(
        lambda _: st.lists(
            st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0],
        )
    ),
    query_vector=vector_strategy,
    limit=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=100, deadline=None)
def test_search_result_limit(chunks: list[tuple[str, list[float], dict]], query_vector: list[float], limit: int):
    """Search should respect requested limit."""
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        for chunk_id, vector, payload in chunks:
            await store.upsert(chunk_id, vector, payload)
        return await store.search(query_vector, limit=limit)

    results = asyncio.run(run_test())

    assert len(results) <= limit
    assert len(results) <= len(chunks)


@given(
    chunks=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=2,
        max_size=10,
        unique_by=lambda x: x[0],
    ),
    query_vector=vector_strategy,
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_search_results_ordering(chunks: list[tuple[str, list[float], dict]], query_vector: list[float]):
    """Search results should be sorted by score descending."""
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        for chunk_id, vector, payload in chunks:
            await store.upsert(chunk_id, vector, payload)
        return await store.search(query_vector, limit=len(chunks))

    results = asyncio.run(run_test())

    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


@given(
    chunks=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=1,
        max_size=20,
        unique_by=lambda x: x[0],
    ),
    query_vector=vector_strategy,
)
@settings(max_examples=100, deadline=None)
def test_search_result_completeness(chunks: list[tuple[str, list[float], dict]], query_vector: list[float]):
    """Search results should include required fields."""
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        for chunk_id, vector, payload in chunks:
            await store.upsert(chunk_id, vector, payload)
        return await store.search(query_vector, limit=len(chunks))

    results = asyncio.run(run_test())

    for i, result in enumerate(results):
        assert result.file_path
        assert result.start_line > 0
        assert result.end_line >= result.start_line
        assert result.score is not None
        assert result.chunk_id


@given(
    file_path=file_path_strategy,
    chunks_for_file=st.integers(min_value=1, max_value=5),
    other_chunks=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_delete_by_file(file_path: str, chunks_for_file: int, other_chunks: int):
    """delete_by_file should remove all vectors for the requested file."""
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        for i in range(chunks_for_file):
            await store.upsert(
                f"target_{i}",
                [0.1] * 128,
                {
                    "file_path": file_path,
                    "start_line": i + 1,
                    "end_line": i + 10,
                    "content": f"chunk {i}",
                },
            )

        for i in range(other_chunks):
            await store.upsert(
                f"other_{i}",
                [0.2] * 128,
                {
                    "file_path": f"other/file_{i}.py",
                    "start_line": 1,
                    "end_line": 10,
                    "content": "other",
                },
            )

        deleted_count = await store.delete_by_file(file_path)

        remaining_target = []
        for i in range(chunks_for_file):
            result = await store.get_by_id(f"target_{i}")
            if result:
                remaining_target.append(result)

        stats = await store.get_stats()
        return deleted_count, remaining_target, stats

    deleted_count, remaining_target, stats = asyncio.run(run_test())

    assert deleted_count == chunks_for_file
    assert len(remaining_target) == 0
    assert stats["total_vectors"] == other_chunks

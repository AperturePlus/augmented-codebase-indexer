"""
Property-based tests for VectorStore.

**Feature: codebase-semantic-search, Property 10: Vector Storage Round-Trip**
**Validates: Requirements 3.3, 3.4**
"""

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aci.infrastructure.fakes import InMemoryVectorStore

# Strategies for generating test data
chunk_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=36,
)

file_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/._-"),
    min_size=1,
    max_size=100,
).filter(lambda x: x.strip() != "")

line_number_strategy = st.integers(min_value=1, max_value=10000)

content_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=0,
    max_size=500,
)

# Vector strategy - normalized floats
vector_strategy = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=128,
    max_size=128,
)


@st.composite
def chunk_payload_strategy(draw):
    """Generate a valid chunk payload."""
    start_line = draw(line_number_strategy)
    end_line = draw(st.integers(min_value=start_line, max_value=start_line + 500))

    return {
        "file_path": draw(file_path_strategy),
        "start_line": start_line,
        "end_line": end_line,
        "content": draw(content_strategy),
        "language": draw(st.sampled_from(["python", "javascript", "go", "unknown"])),
        "chunk_type": draw(st.sampled_from(["function", "class", "method", "fixed"])),
    }


@given(
    chunk_id=chunk_id_strategy,
    vector=vector_strategy,
    payload=chunk_payload_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_vector_storage_round_trip(chunk_id: str, vector: list[float], payload: dict):
    """
    **Feature: codebase-semantic-search, Property 10: Vector Storage Round-Trip**
    **Validates: Requirements 3.3, 3.4**

    *For any* CodeChunk stored in VectorStore, retrieving by chunk_id
    should return the exact same file_path, start_line, end_line, and content.
    """
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        # Store the chunk
        await store.upsert(chunk_id, vector, payload)

        # Retrieve by ID
        result = await store.get_by_id(chunk_id)

        return result

    result = asyncio.run(run_test())

    # Verify round-trip preserves all fields
    assert result is not None, f"Failed to retrieve chunk {chunk_id}"
    assert result.chunk_id == chunk_id
    assert result.file_path == payload["file_path"]
    assert result.start_line == payload["start_line"]
    assert result.end_line == payload["end_line"]
    assert result.content == payload["content"]


@given(
    chunks=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=1,
        max_size=20,
        unique_by=lambda x: x[0],  # Unique chunk IDs
    ),
    query_vector=vector_strategy,
    limit=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=100, deadline=None)
def test_search_result_limit(
    chunks: list[tuple[str, list[float], dict]],
    query_vector: list[float],
    limit: int,
):
    """
    **Feature: codebase-semantic-search, Property 13: Search Result Limit**
    **Validates: Requirements 4.4**

    *For any* search with limit K, the number of results should be <= K.
    """
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        # Store all chunks
        for chunk_id, vector, payload in chunks:
            await store.upsert(chunk_id, vector, payload)

        # Search with limit
        results = await store.search(query_vector, limit=limit)

        return results

    results = asyncio.run(run_test())

    # Verify result count respects limit
    assert len(results) <= limit, f"Got {len(results)} results, expected <= {limit}"
    # Also verify we don't get more than stored
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
def test_search_results_ordering(
    chunks: list[tuple[str, list[float], dict]],
    query_vector: list[float],
):
    """
    **Feature: codebase-semantic-search, Property 11: Search Results Ordering**
    **Validates: Requirements 4.2**

    *For any* search query, results should be sorted by score descending.
    """
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        for chunk_id, vector, payload in chunks:
            await store.upsert(chunk_id, vector, payload)

        results = await store.search(query_vector, limit=len(chunks))
        return results

    results = asyncio.run(run_test())

    # Verify descending order by score
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score, (
            f"Results not sorted: score[{i}]={results[i].score} < "
            f"score[{i + 1}]={results[i + 1].score}"
        )


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
def test_search_result_completeness(
    chunks: list[tuple[str, list[float], dict]],
    query_vector: list[float],
):
    """
    **Feature: codebase-semantic-search, Property 12: Search Result Completeness**
    **Validates: Requirements 4.3**

    *For any* SearchResult, file_path, start_line, end_line, and score
    should not be null or empty.
    """
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        for chunk_id, vector, payload in chunks:
            await store.upsert(chunk_id, vector, payload)

        results = await store.search(query_vector, limit=len(chunks))
        return results

    results = asyncio.run(run_test())

    for i, result in enumerate(results):
        assert result.file_path, f"Result {i} has empty file_path"
        assert result.start_line > 0, f"Result {i} has invalid start_line"
        assert result.end_line >= result.start_line, f"Result {i} has end_line < start_line"
        assert result.score is not None, f"Result {i} has null score"
        assert result.chunk_id, f"Result {i} has empty chunk_id"


@given(
    file_path=file_path_strategy,
    chunks_for_file=st.integers(min_value=1, max_value=5),
    other_chunks=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_delete_by_file(file_path: str, chunks_for_file: int, other_chunks: int):
    """
    *For any* file_path, delete_by_file should remove all chunks
    with that file_path and return the correct count.
    """
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        # Add chunks for target file
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

        # Add chunks for other files
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

        # Delete by file
        deleted_count = await store.delete_by_file(file_path)

        # Try to retrieve deleted chunks
        remaining_target = []
        for i in range(chunks_for_file):
            result = await store.get_by_id(f"target_{i}")
            if result:
                remaining_target.append(result)

        # Get stats
        stats = await store.get_stats()

        return deleted_count, remaining_target, stats

    deleted_count, remaining_target, stats = asyncio.run(run_test())

    # Verify correct count deleted
    assert deleted_count == chunks_for_file, (
        f"Expected {chunks_for_file} deleted, got {deleted_count}"
    )

    # Verify no target chunks remain
    assert len(remaining_target) == 0, "Some target chunks still exist"

    # Verify other chunks still exist
    assert stats["total_vectors"] == other_chunks

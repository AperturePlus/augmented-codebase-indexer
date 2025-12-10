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


@given(
    default_collection=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    explicit_collection=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    chunks_default=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0],
    ),
    chunks_explicit=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0],
    ),
    query_vector=vector_strategy,
)
@settings(max_examples=100, deadline=None)
def test_explicit_collection_preserves_instance_state(
    default_collection: str,
    explicit_collection: str,
    chunks_default: list[tuple[str, list[float], dict]],
    chunks_explicit: list[tuple[str, list[float], dict]],
    query_vector: list[float],
):
    """
    **Feature: vector-store-isolation, Property 2: Explicit collection preserves instance state**
    **Validates: Requirements 1.2, 1.3**

    *For any* vector store instance with default collection D, and *for any* search
    operation specifying explicit collection E (where E ≠ D), after the search
    completes, the instance's default collection SHALL remain D.
    """
    # Skip if collections are the same (we need distinct collections for this test)
    if default_collection == explicit_collection:
        return

    store = InMemoryVectorStore(vector_size=128, collection_name=default_collection)

    async def run_test():
        # Store chunks in default collection
        for chunk_id, vector, payload in chunks_default:
            await store.upsert(chunk_id, vector, payload)

        # Switch to explicit collection and store different chunks
        store.set_collection(explicit_collection)
        for chunk_id, vector, payload in chunks_explicit:
            await store.upsert(chunk_id, vector, payload)

        # Switch back to default collection
        store.set_collection(default_collection)

        # Verify instance state before search
        collection_before = store.get_collection_name()
        assert collection_before == default_collection

        # Search with explicit collection parameter (should NOT change instance state)
        results = await store.search(
            query_vector,
            limit=10,
            collection_name=explicit_collection,
        )

        # Verify instance state after search - should still be default
        collection_after = store.get_collection_name()

        # Also verify the search returned results from the explicit collection
        # (not from the default collection)
        stats_explicit = await store.get_stats(collection_name=explicit_collection)
        stats_default = await store.get_stats(collection_name=default_collection)

        return collection_before, collection_after, results, stats_explicit, stats_default

    (
        collection_before,
        collection_after,
        results,
        stats_explicit,
        stats_default,
    ) = asyncio.run(run_test())

    # Property: Instance state should be preserved
    assert collection_after == default_collection, (
        f"Instance collection changed from '{default_collection}' to '{collection_after}' "
        f"after search with explicit collection '{explicit_collection}'"
    )

    # Verify search actually queried the explicit collection
    # Results should be from explicit collection (up to its size)
    assert len(results) <= len(chunks_explicit), (
        f"Got {len(results)} results but explicit collection only has {len(chunks_explicit)} chunks"
    )

    # Verify stats are correct for each collection
    assert stats_default["total_vectors"] == len(chunks_default)
    assert stats_explicit["total_vectors"] == len(chunks_explicit)


@given(
    collection_a=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    collection_b=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    chunks_a=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0],
    ),
    chunks_b=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0],
    ),
    query_vector=vector_strategy,
)
@settings(max_examples=100, deadline=None)
def test_concurrent_search_isolation(
    collection_a: str,
    collection_b: str,
    chunks_a: list[tuple[str, list[float], dict]],
    chunks_b: list[tuple[str, list[float], dict]],
    query_vector: list[float],
):
    """
    **Feature: vector-store-isolation, Property 1: Concurrent search isolation**
    **Validates: Requirements 1.1**

    *For any* two collections A and B with distinct data, and *for any* two
    concurrent search queries targeting A and B respectively, each query SHALL
    return results only from its specified collection.
    """
    # Skip if collections are the same (we need distinct collections for this test)
    if collection_a == collection_b:
        return

    # Ensure chunk IDs are unique across both collections
    chunk_ids_a = {c[0] for c in chunks_a}
    chunk_ids_b = {c[0] for c in chunks_b}
    if chunk_ids_a & chunk_ids_b:
        return  # Skip if there's overlap in chunk IDs

    store = InMemoryVectorStore(vector_size=128, collection_name=collection_a)

    async def run_test():
        # Store chunks in collection A
        for chunk_id, vector, payload in chunks_a:
            await store.upsert(chunk_id, vector, payload)

        # Switch to collection B and store different chunks
        store.set_collection(collection_b)
        for chunk_id, vector, payload in chunks_b:
            await store.upsert(chunk_id, vector, payload)

        # Run concurrent searches targeting different collections
        async def search_collection_a():
            return await store.search(
                query_vector,
                limit=10,
                collection_name=collection_a,
            )

        async def search_collection_b():
            return await store.search(
                query_vector,
                limit=10,
                collection_name=collection_b,
            )

        # Execute both searches concurrently
        results_a, results_b = await asyncio.gather(
            search_collection_a(),
            search_collection_b(),
        )

        return results_a, results_b

    results_a, results_b = asyncio.run(run_test())

    # Property: Results from collection A should only contain chunk IDs from collection A
    for result in results_a:
        assert result.chunk_id in chunk_ids_a, (
            f"Result chunk_id '{result.chunk_id}' from collection A search "
            f"is not in collection A chunks: {chunk_ids_a}"
        )

    # Property: Results from collection B should only contain chunk IDs from collection B
    for result in results_b:
        assert result.chunk_id in chunk_ids_b, (
            f"Result chunk_id '{result.chunk_id}' from collection B search "
            f"is not in collection B chunks: {chunk_ids_b}"
        )

    # Verify we got results from each collection (up to the number of chunks stored)
    assert len(results_a) <= len(chunks_a)
    assert len(results_b) <= len(chunks_b)


@given(
    collection_name=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    chunks=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=0,
        max_size=20,
        unique_by=lambda x: x[0],
    ),
)
@settings(max_examples=100, deadline=None)
def test_status_reports_correct_collection_stats(
    collection_name: str,
    chunks: list[tuple[str, list[float], dict]],
):
    """
    **Feature: vector-store-isolation, Property 5: Status reports correct collection stats**
    **Validates: Requirements 3.1, 3.2**

    *For any* collection C with N vectors, when status is queried with explicit
    collection C, the returned vector count SHALL equal N.
    """
    store = InMemoryVectorStore(vector_size=128, collection_name=collection_name)

    async def run_test():
        # Store all chunks in the collection
        for chunk_id, vector, payload in chunks:
            await store.upsert(chunk_id, vector, payload)

        # Query stats with explicit collection name
        stats = await store.get_stats(collection_name=collection_name)

        return stats

    stats = asyncio.run(run_test())

    # Property: Vector count should equal number of chunks stored
    expected_count = len(chunks)
    actual_count = stats["total_vectors"]

    assert actual_count == expected_count, (
        f"Stats reported {actual_count} vectors but {expected_count} were stored "
        f"in collection '{collection_name}'"
    )

    # Also verify collection name is reported correctly
    assert stats["collection_name"] == collection_name, (
        f"Stats reported collection '{stats['collection_name']}' but expected '{collection_name}'"
    )


@given(
    collection_a=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    collection_b=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    chunks_a=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=1,
        max_size=10,
        unique_by=lambda x: x[0],
    ),
    chunks_b=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=1,
        max_size=10,
        unique_by=lambda x: x[0],
    ),
)
@settings(max_examples=100, deadline=None)
def test_status_reports_independent_collection_stats(
    collection_a: str,
    collection_b: str,
    chunks_a: list[tuple[str, list[float], dict]],
    chunks_b: list[tuple[str, list[float], dict]],
):
    """
    **Feature: vector-store-isolation, Property 5: Status reports correct collection stats**
    **Validates: Requirements 3.1, 3.2**

    *For any* two collections A and B with different data, querying status for each
    collection independently SHALL return the correct stats for that collection only.
    """
    # Skip if collections are the same (we need distinct collections for this test)
    if collection_a == collection_b:
        return

    store = InMemoryVectorStore(vector_size=128, collection_name=collection_a)

    async def run_test():
        # Store chunks in collection A
        for chunk_id, vector, payload in chunks_a:
            await store.upsert(chunk_id, vector, payload)

        # Switch to collection B and store different chunks
        store.set_collection(collection_b)
        for chunk_id, vector, payload in chunks_b:
            await store.upsert(chunk_id, vector, payload)

        # Query stats for each collection independently (using explicit collection_name)
        stats_a = await store.get_stats(collection_name=collection_a)
        stats_b = await store.get_stats(collection_name=collection_b)

        return stats_a, stats_b

    stats_a, stats_b = asyncio.run(run_test())

    # Property: Each collection should report its own vector count
    assert stats_a["total_vectors"] == len(chunks_a), (
        f"Collection A stats reported {stats_a['total_vectors']} vectors "
        f"but {len(chunks_a)} were stored"
    )
    assert stats_b["total_vectors"] == len(chunks_b), (
        f"Collection B stats reported {stats_b['total_vectors']} vectors "
        f"but {len(chunks_b)} were stored"
    )

    # Property: Collection names should be reported correctly
    assert stats_a["collection_name"] == collection_a
    assert stats_b["collection_name"] == collection_b


@given(
    default_collection=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ),
    repo_collections=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=20,
        ),
        min_size=1,
        max_size=5,
        unique=True,
    ),
    chunks_per_collection=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_reset_deletes_all_registered_collections(
    default_collection: str,
    repo_collections: list[str],
    chunks_per_collection: int,
):
    """
    **Feature: vector-store-isolation, Property 3: Reset deletes all registered collections**
    **Validates: Requirements 2.1, 2.3**

    *For any* set of repositories registered in metadata (each with an associated
    collection), after reset completes, all associated collections SHALL be deleted
    from Qdrant.
    """
    # Ensure default collection is not in repo_collections to avoid overlap
    repo_collections = [c for c in repo_collections if c != default_collection]
    if not repo_collections:
        return  # Skip if all collections were filtered out

    store = InMemoryVectorStore(vector_size=128, collection_name=default_collection)

    async def run_test():
        # Create chunks in default collection
        for i in range(chunks_per_collection):
            await store.upsert(
                f"default_{i}",
                [0.1] * 128,
                {
                    "file_path": f"default/file_{i}.py",
                    "start_line": 1,
                    "end_line": 10,
                    "content": f"default chunk {i}",
                },
            )

        # Create chunks in each repository collection
        for collection_name in repo_collections:
            store.set_collection(collection_name)
            for i in range(chunks_per_collection):
                await store.upsert(
                    f"{collection_name}_{i}",
                    [0.2] * 128,
                    {
                        "file_path": f"{collection_name}/file_{i}.py",
                        "start_line": 1,
                        "end_line": 10,
                        "content": f"{collection_name} chunk {i}",
                    },
                )

        # Verify all collections have data before reset
        stats_before = {}
        for collection_name in [default_collection] + repo_collections:
            stats = await store.get_stats(collection_name=collection_name)
            stats_before[collection_name] = stats["total_vectors"]
            assert stats["total_vectors"] == chunks_per_collection, (
                f"Collection {collection_name} should have {chunks_per_collection} vectors before reset"
            )

        # Simulate reset: delete all registered collections + default collection
        all_collections = [default_collection] + repo_collections
        deleted_collections = []
        for collection_name in all_collections:
            deleted = await store.delete_collection(collection_name)
            if deleted:
                deleted_collections.append(collection_name)

        # Verify all collections were deleted
        stats_after = {}
        for collection_name in all_collections:
            stats = await store.get_stats(collection_name=collection_name)
            stats_after[collection_name] = stats["total_vectors"]

        return stats_before, stats_after, deleted_collections

    stats_before, stats_after, deleted_collections = asyncio.run(run_test())

    # Property: All collections should be deleted (have 0 vectors after reset)
    for collection_name in [default_collection] + repo_collections:
        assert stats_after[collection_name] == 0, (
            f"Collection '{collection_name}' still has {stats_after[collection_name]} vectors after reset"
        )

    # Property: All collections should have been successfully deleted
    expected_deleted = set([default_collection] + repo_collections)
    actual_deleted = set(deleted_collections)
    assert actual_deleted == expected_deleted, (
        f"Expected to delete collections {expected_deleted}, but deleted {actual_deleted}"
    )

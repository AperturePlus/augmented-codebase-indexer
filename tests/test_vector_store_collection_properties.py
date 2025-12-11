"""Property-based tests for vector store collection isolation and lifecycle."""

import asyncio

from hypothesis import given, settings, strategies as st

from aci.infrastructure.fakes import InMemoryVectorStore
from tests.vector_store_strategies import chunk_id_strategy, chunk_payload_strategy, vector_strategy


@given(
    default_collection=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20),
    explicit_collection=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20),
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
    """Using explicit collection names should not mutate the instance's default collection."""
    if default_collection == explicit_collection:
        return

    store = InMemoryVectorStore(vector_size=128, collection_name=default_collection)

    async def run_test():
        for chunk_id, vector, payload in chunks_default:
            await store.upsert(chunk_id, vector, payload)

        store.set_collection(explicit_collection)
        for chunk_id, vector, payload in chunks_explicit:
            await store.upsert(chunk_id, vector, payload)

        store.set_collection(default_collection)
        collection_before = store.get_collection_name()
        assert collection_before == default_collection

        results = await store.search(query_vector, limit=10, collection_name=explicit_collection)
        collection_after = store.get_collection_name()

        stats_explicit = await store.get_stats(collection_name=explicit_collection)
        stats_default = await store.get_stats(collection_name=default_collection)
        return collection_after, results, stats_explicit, stats_default

    collection_after, results, stats_explicit, stats_default = asyncio.run(run_test())

    assert collection_after == default_collection
    assert len(results) <= len(chunks_explicit)
    assert stats_default["total_vectors"] == len(chunks_default)
    assert stats_explicit["total_vectors"] == len(chunks_explicit)


@given(
    collection_a=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20),
    collection_b=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20),
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
    """Concurrent searches should respect collection isolation."""
    if collection_a == collection_b:
        return

    chunk_ids_a = {c[0] for c in chunks_a}
    chunk_ids_b = {c[0] for c in chunks_b}
    if chunk_ids_a & chunk_ids_b:
        return

    store = InMemoryVectorStore(vector_size=128, collection_name=collection_a)

    async def run_test():
        for chunk_id, vector, payload in chunks_a:
            await store.upsert(chunk_id, vector, payload)

        store.set_collection(collection_b)
        for chunk_id, vector, payload in chunks_b:
            await store.upsert(chunk_id, vector, payload)

        async def search_collection(name: str):
            return await store.search(query_vector, limit=10, collection_name=name)

        return await asyncio.gather(search_collection(collection_a), search_collection(collection_b))

    results_a, results_b = asyncio.run(run_test())

    for result in results_a:
        assert result.chunk_id in chunk_ids_a
    for result in results_b:
        assert result.chunk_id in chunk_ids_b
    assert len(results_a) <= len(chunks_a)
    assert len(results_b) <= len(chunks_b)


@given(
    collection_name=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20),
    chunks=st.lists(
        st.tuples(chunk_id_strategy, vector_strategy, chunk_payload_strategy()),
        min_size=0,
        max_size=20,
        unique_by=lambda x: x[0],
    ),
)
@settings(max_examples=100, deadline=None)
def test_status_reports_correct_collection_stats(collection_name: str, chunks: list[tuple[str, list[float], dict]]):
    """Status should reflect stored vector counts for a collection."""
    store = InMemoryVectorStore(vector_size=128, collection_name=collection_name)

    async def run_test():
        for chunk_id, vector, payload in chunks:
            await store.upsert(chunk_id, vector, payload)
        return await store.get_stats(collection_name=collection_name)

    stats = asyncio.run(run_test())

    assert stats["total_vectors"] == len(chunks)
    assert stats["collection_name"] == collection_name


@given(
    collection_a=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20),
    collection_b=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20),
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
    """Status should report counts independently per collection."""
    if collection_a == collection_b:
        return

    store = InMemoryVectorStore(vector_size=128, collection_name=collection_a)

    async def run_test():
        for chunk_id, vector, payload in chunks_a:
            await store.upsert(chunk_id, vector, payload)

        store.set_collection(collection_b)
        for chunk_id, vector, payload in chunks_b:
            await store.upsert(chunk_id, vector, payload)

        stats_a = await store.get_stats(collection_name=collection_a)
        stats_b = await store.get_stats(collection_name=collection_b)
        return stats_a, stats_b

    stats_a, stats_b = asyncio.run(run_test())

    assert stats_a["total_vectors"] == len(chunks_a)
    assert stats_b["total_vectors"] == len(chunks_b)
    assert stats_a["collection_name"] == collection_a
    assert stats_b["collection_name"] == collection_b


@given(
    default_collection=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20),
    repo_collections=st.lists(
        st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20),
        min_size=1,
        max_size=5,
        unique=True,
    ),
    chunks_per_collection=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_reset_deletes_all_registered_collections(
    default_collection: str, repo_collections: list[str], chunks_per_collection: int
):
    """Reset should delete all registered collections."""
    repo_collections = [c for c in repo_collections if c != default_collection]
    if not repo_collections:
        return

    store = InMemoryVectorStore(vector_size=128, collection_name=default_collection)

    async def run_test():
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

        all_collections = [default_collection] + repo_collections
        for collection_name in all_collections:
            stats = await store.get_stats(collection_name=collection_name)
            assert stats["total_vectors"] == chunks_per_collection

        deleted_collections = []
        for collection_name in all_collections:
            deleted = await store.delete_collection(collection_name)
            if deleted:
                deleted_collections.append(collection_name)

        stats_after = {}
        for collection_name in all_collections:
            stats = await store.get_stats(collection_name=collection_name)
            stats_after[collection_name] = stats["total_vectors"]

        return deleted_collections, stats_after

    deleted_collections, stats_after = asyncio.run(run_test())

    expected_deleted = set([default_collection] + repo_collections)
    assert set(deleted_collections) == expected_deleted
    for collection_name in expected_deleted:
        assert stats_after[collection_name] == 0

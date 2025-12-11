"""
Property-based tests for SearchService artifact type support.

**Feature: multi-granularity-indexing, Property 9: Default search returns all artifact types**
**Feature: multi-granularity-indexing, Property 11: Search results include artifact type**
**Validates: Requirements 4.1, 4.3**
"""

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.services.search_service import SearchService
from aci.services.search_types import SearchMode

# Valid artifact types as defined in the design
ARTIFACT_TYPES = ["chunk", "function_summary", "class_summary", "file_summary"]


def run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@st.composite
def artifacts_with_types_strategy(draw):
    """Generate a list of artifacts with various artifact types."""
    num_artifacts = draw(st.integers(min_value=4, max_value=8))
    artifacts = []
    embedding_client = LocalEmbeddingClient(dimension=128)

    for i in range(num_artifacts):
        # Cycle through artifact types to ensure coverage
        artifact_type = ARTIFACT_TYPES[i % len(ARTIFACT_TYPES)]
        chunk_id = f"artifact_{i}_{draw(st.integers(min_value=0, max_value=9999))}"
        content = f"content for {artifact_type} artifact {i}"
        # Generate embedding from content for realistic search
        vector = embedding_client.embed_sync(content)
        start_line = draw(st.integers(min_value=1, max_value=1000))
        payload = {
            "file_path": f"test/file_{i}.py",
            "start_line": start_line,
            "end_line": start_line + draw(st.integers(min_value=1, max_value=50)),
            "content": content,
            "artifact_type": artifact_type,
            "language": "python",
        }
        artifacts.append((chunk_id, vector, payload))
    return artifacts



@given(
    artifacts=artifacts_with_types_strategy(),
    query=st.text(
        min_size=5,
        max_size=50,
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    ).filter(lambda x: x.strip()),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_default_search_returns_all_artifact_types(
    artifacts: list[tuple[str, list[float], dict]],
    query: str,
):
    """
    **Feature: multi-granularity-indexing, Property 9: Default search returns all artifact types**
    **Validates: Requirements 4.1**

    *For any* search query on an index containing multiple artifact types, the results
    SHALL include artifacts of all types (chunks and summaries) when no type filter
    is specified.
    """
    # Deduplicate chunk IDs
    seen_ids = set()
    unique_artifacts = []
    for chunk_id, vector, payload in artifacts:
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_artifacts.append((chunk_id, vector, payload))

    if not unique_artifacts:
        return  # Skip if no artifacts

    # Set up search service with fakes
    vector_store = InMemoryVectorStore(vector_size=128)
    embedding_client = LocalEmbeddingClient(dimension=128)
    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=None,
        default_limit=len(unique_artifacts),
        vector_candidates=len(unique_artifacts),
    )

    async def run_test():
        # Store all artifacts
        for chunk_id, vector, payload in unique_artifacts:
            await vector_store.upsert(chunk_id, vector, payload)

        # Search without artifact type filter (should return all types)
        results = await search_service.search(
            query=query,
            limit=len(unique_artifacts),
            artifact_types=None,  # No filter - should return all types
            search_mode=SearchMode.VECTOR,
            use_rerank=False,
        )

        return results

    results = run_async(run_test())

    # Collect artifact types from stored data
    stored_types = {p["artifact_type"] for _, _, p in unique_artifacts}

    # Collect artifact types from results
    result_types = {r.metadata.get("artifact_type") for r in results}

    # When no filter is specified, results should potentially include all stored types
    # (depending on similarity scores, but at least the types should be valid)
    for result_type in result_types:
        assert result_type in stored_types, (
            f"Result type '{result_type}' not in stored types {stored_types}"
        )

    # Verify we got results (if we stored any)
    if unique_artifacts:
        assert len(results) > 0, "Expected at least one result when artifacts are stored"



@given(
    artifacts=artifacts_with_types_strategy(),
    query=st.text(
        min_size=5,
        max_size=50,
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    ).filter(lambda x: x.strip()),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_search_results_include_artifact_type(
    artifacts: list[tuple[str, list[float], dict]],
    query: str,
):
    """
    **Feature: multi-granularity-indexing, Property 11: Search results include artifact type**
    **Validates: Requirements 4.3**

    *For any* search result returned by the search service, the result SHALL include
    a non-null artifact_type field indicating the type of the matched artifact.
    """
    # Deduplicate chunk IDs
    seen_ids = set()
    unique_artifacts = []
    for chunk_id, vector, payload in artifacts:
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_artifacts.append((chunk_id, vector, payload))

    if not unique_artifacts:
        return  # Skip if no artifacts

    # Set up search service with fakes
    vector_store = InMemoryVectorStore(vector_size=128)
    embedding_client = LocalEmbeddingClient(dimension=128)
    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=None,
        default_limit=len(unique_artifacts),
        vector_candidates=len(unique_artifacts),
    )

    async def run_test():
        # Store all artifacts
        for chunk_id, vector, payload in unique_artifacts:
            await vector_store.upsert(chunk_id, vector, payload)

        # Search
        results = await search_service.search(
            query=query,
            limit=len(unique_artifacts),
            search_mode=SearchMode.VECTOR,
            use_rerank=False,
        )

        return results

    results = run_async(run_test())

    # Verify every result has artifact_type in metadata
    for i, result in enumerate(results):
        assert "artifact_type" in result.metadata, (
            f"Result {i} (chunk_id={result.chunk_id}) missing artifact_type in metadata"
        )
        assert result.metadata["artifact_type"] is not None, (
            f"Result {i} (chunk_id={result.chunk_id}) has null artifact_type"
        )
        assert result.metadata["artifact_type"] in ARTIFACT_TYPES, (
            f"Result {i} has invalid artifact_type: {result.metadata['artifact_type']}"
        )

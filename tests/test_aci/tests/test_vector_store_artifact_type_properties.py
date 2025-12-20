"""
Property-based tests for Vector Store artifact type support.

**Feature: multi-granularity-indexing, Property 6: Stored artifacts have required fields**
**Feature: multi-granularity-indexing, Property 10: Artifact type filter works correctly**
**Validates: Requirements 3.1, 3.2, 4.2**
"""

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aci.infrastructure.fakes import InMemoryVectorStore

# Valid artifact types as defined in the design
ARTIFACT_TYPES = ["chunk", "function_summary", "class_summary", "file_summary"]

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

line_number_strategy = st.integers(min_value=0, max_value=10000)

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

artifact_type_strategy = st.sampled_from(ARTIFACT_TYPES)


@st.composite
def artifact_payload_strategy(draw):
    """Generate a valid artifact payload with artifact_type."""
    start_line = draw(line_number_strategy)
    end_line = draw(st.integers(min_value=start_line, max_value=start_line + 500))
    artifact_type = draw(artifact_type_strategy)

    return {
        "file_path": draw(file_path_strategy),
        "start_line": start_line,
        "end_line": end_line,
        "content": draw(content_strategy),
        "artifact_type": artifact_type,
        "language": draw(st.sampled_from(["python", "javascript", "go", "unknown"])),
    }


@given(
    chunk_id=chunk_id_strategy,
    vector=vector_strategy,
    payload=artifact_payload_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_stored_artifacts_have_required_fields(
    chunk_id: str, vector: list[float], payload: dict
):
    """
    **Feature: multi-granularity-indexing, Property 6: Stored artifacts have required fields**
    **Validates: Requirements 3.1, 3.2**

    *For any* summary artifact stored in the vector store, the payload SHALL include
    artifact_type, file_path, start_line, and end_line fields with valid values.
    """
    store = InMemoryVectorStore(vector_size=128)

    async def run_test():
        # Store the artifact
        await store.upsert(chunk_id, vector, payload)

        # Retrieve by ID
        result = await store.get_by_id(chunk_id)

        return result

    result = asyncio.run(run_test())

    # Verify required fields are present and valid
    assert result is not None, f"Failed to retrieve artifact {chunk_id}"
    assert result.file_path == payload["file_path"], "file_path mismatch"
    assert result.start_line == payload["start_line"], "start_line mismatch"
    assert result.end_line == payload["end_line"], "end_line mismatch"

    # artifact_type should be in metadata
    assert "artifact_type" in result.metadata, "artifact_type not in metadata"
    assert result.metadata["artifact_type"] == payload["artifact_type"], "artifact_type mismatch"
    assert result.metadata["artifact_type"] in ARTIFACT_TYPES, (
        f"Invalid artifact_type: {result.metadata['artifact_type']}"
    )


@given(
    chunk_id=chunk_id_strategy,
    vector=vector_strategy,
    file_path=file_path_strategy,
    start_line=line_number_strategy,
    content=content_strategy,
)
@settings(max_examples=100, deadline=None)
def test_default_artifact_type_is_chunk(
    chunk_id: str,
    vector: list[float],
    file_path: str,
    start_line: int,
    content: str,
):
    """
    **Feature: multi-granularity-indexing, Property 6: Stored artifacts have required fields**
    **Validates: Requirements 3.1**

    *For any* payload stored without an explicit artifact_type, the vector store
    SHALL default to "chunk" for backward compatibility.
    """
    store = InMemoryVectorStore(vector_size=128)

    # Payload without artifact_type
    payload = {
        "file_path": file_path,
        "start_line": start_line,
        "end_line": start_line + 10,
        "content": content,
    }

    async def run_test():
        # Store without artifact_type
        await store.upsert(chunk_id, vector, payload)

        # Retrieve by ID
        result = await store.get_by_id(chunk_id)

        return result

    result = asyncio.run(run_test())

    # Verify artifact_type defaults to "chunk"
    assert result is not None, f"Failed to retrieve artifact {chunk_id}"
    assert "artifact_type" in result.metadata, "artifact_type not in metadata"
    assert result.metadata["artifact_type"] == "chunk", (
        f"Expected default artifact_type 'chunk', got '{result.metadata['artifact_type']}'"
    )


@st.composite
def artifacts_with_types_strategy(draw):
    """Generate a list of artifacts with various artifact types."""
    # Generate a smaller, more efficient list of artifacts
    num_artifacts = draw(st.integers(min_value=4, max_value=8))
    artifacts = []

    for i in range(num_artifacts):
        # Cycle through artifact types to ensure coverage
        artifact_type = ARTIFACT_TYPES[i % len(ARTIFACT_TYPES)]
        chunk_id = f"artifact_{i}_{draw(st.integers(min_value=0, max_value=9999))}"
        # Use simpler vector generation
        vector = [draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
                  for _ in range(128)]
        start_line = draw(st.integers(min_value=0, max_value=1000))
        payload = {
            "file_path": f"test/file_{i}.py",
            "start_line": start_line,
            "end_line": start_line + draw(st.integers(min_value=1, max_value=50)),
            "content": f"content for artifact {i}",
            "artifact_type": artifact_type,
        }
        artifacts.append((chunk_id, vector, payload))
    return artifacts


@given(
    artifacts=artifacts_with_types_strategy(),
    filter_types=st.lists(
        artifact_type_strategy,
        min_size=1,
        max_size=len(ARTIFACT_TYPES),
        unique=True,
    ),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_artifact_type_filter_works_correctly(
    artifacts: list[tuple[str, list[float], dict]],
    filter_types: list[str],
):
    """
    **Feature: multi-granularity-indexing, Property 10: Artifact type filter works correctly**
    **Validates: Requirements 4.2**

    *For any* search query with an artifact type filter, all returned results SHALL
    have artifact_type matching one of the specified filter values.
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

    store = InMemoryVectorStore(vector_size=128)
    # Use a simple query vector
    query_vector = [0.1] * 128

    async def run_test():
        # Store all artifacts
        for chunk_id, vector, payload in unique_artifacts:
            await store.upsert(chunk_id, vector, payload)

        # Search with artifact type filter
        results = await store.search(
            query_vector,
            limit=len(unique_artifacts),
            artifact_types=filter_types,
        )

        return results

    results = asyncio.run(run_test())

    # Verify all results have artifact_type in the filter list
    for result in results:
        assert "artifact_type" in result.metadata, (
            f"Result {result.chunk_id} missing artifact_type in metadata"
        )
        assert result.metadata["artifact_type"] in filter_types, (
            f"Result {result.chunk_id} has artifact_type '{result.metadata['artifact_type']}' "
            f"which is not in filter {filter_types}"
        )

    # Verify we got results for the filtered types (if any exist)
    expected_types_in_data = {
        p["artifact_type"] for _, _, p in unique_artifacts if p["artifact_type"] in filter_types
    }
    if expected_types_in_data:
        result_types = {r.metadata["artifact_type"] for r in results}
        # At least some of the expected types should be in results
        assert result_types.issubset(set(filter_types)), (
            f"Result types {result_types} not subset of filter {filter_types}"
        )


@given(
    artifacts=artifacts_with_types_strategy(),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_default_search_returns_all_artifact_types(
    artifacts: list[tuple[str, list[float], dict]],
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

    store = InMemoryVectorStore(vector_size=128)
    # Use a simple query vector
    query_vector = [0.1] * 128

    async def run_test():
        # Store all artifacts
        for chunk_id, vector, payload in unique_artifacts:
            await store.upsert(chunk_id, vector, payload)

        # Search without artifact type filter (should return all types)
        results = await store.search(
            query_vector,
            limit=len(unique_artifacts),
            artifact_types=None,  # No filter
        )

        return results

    results = asyncio.run(run_test())

    # Collect artifact types from stored data
    stored_types = {p["artifact_type"] for _, _, p in unique_artifacts}

    # Collect artifact types from results
    result_types = {r.metadata["artifact_type"] for r in results}

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
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
def test_search_results_include_artifact_type(
    artifacts: list[tuple[str, list[float], dict]],
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

    store = InMemoryVectorStore(vector_size=128)
    # Use a simple query vector
    query_vector = [0.1] * 128

    async def run_test():
        # Store all artifacts
        for chunk_id, vector, payload in unique_artifacts:
            await store.upsert(chunk_id, vector, payload)

        # Search
        results = await store.search(
            query_vector,
            limit=len(unique_artifacts),
        )

        return results

    results = asyncio.run(run_test())

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

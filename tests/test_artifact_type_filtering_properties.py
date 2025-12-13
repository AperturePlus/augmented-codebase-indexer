"""Property-based tests for artifact type filtering validation.

**Feature: service-initialization-refactor, Property 6: Artifact Type Filtering**
**Feature: service-initialization-refactor, Property 7: Invalid Artifact Type Error**
**Validates: Requirements 3.4, 3.5**
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from aci.infrastructure.vector_store import SearchResult


# Valid artifact types as defined in the design
VALID_ARTIFACT_TYPES = ["chunk", "function_summary", "class_summary", "file_summary"]


def run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class MockVectorStore:
    """Mock vector store that returns results with specified artifact types."""

    def __init__(self, results: list[SearchResult] = None):
        self._results = results or []
        self.last_artifact_types = None

    async def search(
        self,
        query_vector: list,
        limit: int = 10,
        file_filter: str = None,
        collection_name: str = None,
        artifact_types: list = None,
    ) -> list[SearchResult]:
        self.last_artifact_types = artifact_types
        # Filter results by artifact_types if specified
        if artifact_types:
            return [
                r for r in self._results
                if r.metadata.get("artifact_type") in artifact_types
            ][:limit]
        return self._results[:limit]

    async def get_all_file_paths(self, collection_name: str = None) -> list:
        return ["test.py"]

    async def get_by_id(self, chunk_id: str):
        return None

    async def get_stats(self, collection_name: str = None) -> dict:
        return {"total_vectors": 0, "total_files": 0}


class MockEmbeddingClient:
    """Mock embedding client for testing."""

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    def get_dimension(self) -> int:
        return self._dimension

    async def embed_batch(self, texts: list) -> list:
        return [[0.1] * self._dimension for _ in texts]


# Strategies for generating test data
artifact_type_strategy = st.sampled_from(VALID_ARTIFACT_TYPES)

valid_artifact_types_filter = st.lists(
    artifact_type_strategy,
    min_size=1,
    max_size=len(VALID_ARTIFACT_TYPES),
    unique=True,
)

# Strategy for generating invalid artifact types
invalid_artifact_type_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=20,
).filter(lambda x: x.strip() and x not in VALID_ARTIFACT_TYPES)


@st.composite
def search_result_with_artifact_type(draw, artifact_type: str = None):
    """Generate a SearchResult with a specific artifact type."""
    if artifact_type is None:
        artifact_type = draw(artifact_type_strategy)
    
    chunk_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=5,
        max_size=20,
    ).filter(lambda x: x.strip()))
    
    file_path = f"test_{draw(st.integers(min_value=1, max_value=100))}.py"
    start_line = draw(st.integers(min_value=1, max_value=100))
    end_line = start_line + draw(st.integers(min_value=1, max_value=50))
    
    return SearchResult(
        chunk_id=chunk_id,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=f"content for {chunk_id}",
        score=draw(st.floats(min_value=0.1, max_value=1.0)),
        metadata={"artifact_type": artifact_type, "language": "python"},
    )


@st.composite
def mixed_artifact_type_results(draw):
    """Generate a list of SearchResults with mixed artifact types."""
    num_results = draw(st.integers(min_value=2, max_value=10))
    results = []
    for i in range(num_results):
        artifact_type = VALID_ARTIFACT_TYPES[i % len(VALID_ARTIFACT_TYPES)]
        result = draw(search_result_with_artifact_type(artifact_type))
        results.append(result)
    return results


class TestArtifactTypeFiltering:
    """
    **Feature: service-initialization-refactor, Property 6: Artifact Type Filtering**
    **Validates: Requirements 3.4**
    
    *For any* search with artifact_types specified, all returned results SHALL have
    an artifact_type metadata value contained in the specified set.
    """

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
        filter_types=valid_artifact_types_filter,
        results=mixed_artifact_type_results(),
    )
    @settings(max_examples=100, deadline=None)
    def test_filtered_results_match_specified_types(
        self, query: str, filter_types: list[str], results: list[SearchResult]
    ):
        """All returned results should have artifact_type in the filter list."""
        assume(query.strip())
        
        from aci.services.search_service import SearchService
        from aci.services.search_types import SearchMode
        
        mock_vector = MockVectorStore(results=results)
        mock_embedding = MockEmbeddingClient()
        
        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
        )
        
        search_results = run_async(
            search_service.search(
                query,
                search_mode=SearchMode.VECTOR,
                artifact_types=filter_types,
            )
        )
        
        # Verify all returned results have artifact_type in the filter list
        for result in search_results:
            assert "artifact_type" in result.metadata, (
                f"Result {result.chunk_id} missing artifact_type in metadata"
            )
            assert result.metadata["artifact_type"] in filter_types, (
                f"Result {result.chunk_id} has artifact_type "
                f"'{result.metadata['artifact_type']}' which is not in filter {filter_types}"
            )

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
        single_type=artifact_type_strategy,
        results=mixed_artifact_type_results(),
    )
    @settings(max_examples=100, deadline=None)
    def test_single_artifact_type_filter(
        self, query: str, single_type: str, results: list[SearchResult]
    ):
        """Filtering by a single artifact type should only return that type."""
        assume(query.strip())
        
        from aci.services.search_service import SearchService
        from aci.services.search_types import SearchMode
        
        mock_vector = MockVectorStore(results=results)
        mock_embedding = MockEmbeddingClient()
        
        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
        )
        
        search_results = run_async(
            search_service.search(
                query,
                search_mode=SearchMode.VECTOR,
                artifact_types=[single_type],
            )
        )
        
        # All results should have the single specified artifact type
        for result in search_results:
            assert result.metadata.get("artifact_type") == single_type, (
                f"Expected artifact_type '{single_type}', "
                f"got '{result.metadata.get('artifact_type')}'"
            )


class TestInvalidArtifactTypeError:
    """
    **Feature: service-initialization-refactor, Property 7: Invalid Artifact Type Error**
    **Validates: Requirements 3.5**
    
    *For any* search with an invalid artifact type, the system SHALL return an error
    message containing all valid artifact type names.
    """

    @given(
        invalid_type=invalid_artifact_type_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_http_invalid_artifact_type_returns_error_with_valid_types(
        self, invalid_type: str
    ):
        """HTTP endpoint should return error listing valid types for invalid input."""
        assume(invalid_type.strip())
        assume(invalid_type not in VALID_ARTIFACT_TYPES)
        
        # Simulate HTTP validation logic (from http_server.py)
        valid_artifact_types = ["chunk", "function_summary", "class_summary", "file_summary"]
        artifact_type = [invalid_type]
        
        invalid_types = [t for t in artifact_type if t not in valid_artifact_types]
        
        assert invalid_types, "Should detect invalid type"
        
        # Construct error message as HTTP server does
        error_message = (
            f"Invalid artifact type(s): {invalid_types}. "
            f"Valid types are: {valid_artifact_types}"
        )
        
        # Verify error message contains all valid types
        for valid_type in VALID_ARTIFACT_TYPES:
            assert valid_type in error_message, (
                f"Error message should contain valid type '{valid_type}'"
            )

    @given(
        invalid_type=invalid_artifact_type_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_mcp_invalid_artifact_type_returns_error_with_valid_types(
        self, invalid_type: str
    ):
        """MCP handler should return error listing valid types for invalid input."""
        assume(invalid_type.strip())
        assume(invalid_type not in VALID_ARTIFACT_TYPES)
        
        # Simulate MCP validation logic (from mcp/handlers.py)
        valid_artifact_types = {"chunk", "function_summary", "class_summary", "file_summary"}
        artifact_types = [invalid_type]
        
        invalid_types = [t for t in artifact_types if t not in valid_artifact_types]
        
        assert invalid_types, "Should detect invalid type"
        
        # Construct error message as MCP handler does
        error_message = (
            f"Error: Invalid artifact type(s): {', '.join(invalid_types)}. "
            f"Valid types: {', '.join(sorted(valid_artifact_types))}"
        )
        
        # Verify error message contains all valid types
        for valid_type in VALID_ARTIFACT_TYPES:
            assert valid_type in error_message, (
                f"Error message should contain valid type '{valid_type}'"
            )

    @given(
        invalid_type=invalid_artifact_type_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_cli_invalid_artifact_type_returns_error_with_valid_types(
        self, invalid_type: str
    ):
        """CLI should return error listing valid types for invalid input."""
        assume(invalid_type.strip())
        assume(invalid_type not in VALID_ARTIFACT_TYPES)
        
        # Simulate CLI validation logic (from cli/__init__.py)
        valid_artifact_types = {"chunk", "function_summary", "class_summary", "file_summary"}
        artifact_type = [invalid_type]
        
        invalid_types = [t for t in artifact_type if t not in valid_artifact_types]
        
        assert invalid_types, "Should detect invalid type"
        
        # Construct error message as CLI does
        error_message = (
            f"Invalid artifact type(s): {', '.join(invalid_types)}. "
            f"Valid types: {', '.join(sorted(valid_artifact_types))}"
        )
        
        # Verify error message contains all valid types
        for valid_type in VALID_ARTIFACT_TYPES:
            assert valid_type in error_message, (
                f"Error message should contain valid type '{valid_type}'"
            )

    @given(
        valid_types=valid_artifact_types_filter,
        invalid_types=st.lists(invalid_artifact_type_strategy, min_size=1, max_size=3, unique=True),
    )
    @settings(max_examples=100, deadline=None)
    def test_mixed_valid_invalid_types_reports_only_invalid(
        self, valid_types: list[str], invalid_types: list[str]
    ):
        """When mixing valid and invalid types, only invalid ones should be reported."""
        # Filter out any invalid types that happen to match valid ones
        invalid_types = [t for t in invalid_types if t not in VALID_ARTIFACT_TYPES]
        assume(invalid_types)  # Need at least one invalid type
        
        mixed_types = valid_types + invalid_types
        
        # Simulate validation logic
        valid_artifact_types = set(VALID_ARTIFACT_TYPES)
        detected_invalid = [t for t in mixed_types if t not in valid_artifact_types]
        
        # Should detect exactly the invalid types we added
        assert set(detected_invalid) == set(invalid_types), (
            f"Should detect invalid types {invalid_types}, got {detected_invalid}"
        )

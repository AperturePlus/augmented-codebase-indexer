"""Property-based tests for SearchService artifact_type metadata handling.

**Feature: hybrid-search-modes, Property 6: Summary artifact results have artifact_type populated**
**Feature: hybrid-search-modes, Property 3: Vector search respects artifact_type filter**
**Validates: Requirements 2.3, 6.1, 1.3, 6.3**
"""

import asyncio
from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.infrastructure.vector_store import SearchResult
from aci.services.search_service import SearchService
from aci.services.search_types import SearchMode

# Valid artifact types as defined in the design
ARTIFACT_TYPES = ["chunk", "function_summary", "class_summary", "file_summary"]
SUMMARY_TYPES = ["function_summary", "class_summary", "file_summary"]


@dataclass
class MockSearchResult:
    """Mock search result for testing."""
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float
    metadata: dict


class MockVectorStoreWithResults:
    """Mock vector store that returns configurable results with artifact_type."""

    def __init__(self, results: list[MockSearchResult] = None):
        self.search_called = False
        self.search_count = 0
        self.last_artifact_types = None
        self._results = results or []

    async def search(
        self,
        query_vector: list,
        limit: int = 10,
        file_filter: str = None,
        collection_name: str = None,
        artifact_types: list = None,
    ) -> list[SearchResult]:
        self.search_called = True
        self.search_count += 1
        self.last_artifact_types = artifact_types

        # Filter results by artifact_types if specified
        filtered_results = []
        for r in self._results:
            artifact_type = r.metadata.get("artifact_type", "chunk")
            if artifact_types is None or artifact_type in artifact_types:
                filtered_results.append(
                    SearchResult(
                        chunk_id=r.chunk_id,
                        file_path=r.file_path,
                        start_line=r.start_line,
                        end_line=r.end_line,
                        content=r.content,
                        score=r.score,
                        metadata=r.metadata,
                    )
                )
        return filtered_results[:limit]

    async def get_all_file_paths(self, collection_name: str = None) -> list:
        return list({r.file_path for r in self._results})

    async def get_by_id(self, chunk_id: str) -> SearchResult | None:
        for r in self._results:
            if r.chunk_id == chunk_id:
                return SearchResult(
                    chunk_id=r.chunk_id,
                    file_path=r.file_path,
                    start_line=r.start_line,
                    end_line=r.end_line,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                )
        return None

    async def get_stats(self, collection_name: str = None) -> dict:
        return {"total_vectors": len(self._results), "total_files": 1}


class MockEmbeddingClient:
    """Mock embedding client for testing."""

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    def get_dimension(self) -> int:
        return self._dimension

    async def embed_batch(self, texts: list) -> list:
        return [[0.1] * self._dimension for _ in texts]


def run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Strategies for generating test data - simplified to avoid filtering issues
artifact_type_strategy = st.sampled_from(ARTIFACT_TYPES)
summary_type_strategy = st.sampled_from(SUMMARY_TYPES)

# Simple alphanumeric query strategy
query_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=3,
    max_size=30,
)


@st.composite
def mock_search_result_strategy(draw, artifact_type: str = None):
    """Generate a mock search result with specified or random artifact_type."""
    if artifact_type is None:
        artifact_type = draw(artifact_type_strategy)

    idx = draw(st.integers(min_value=0, max_value=9999))
    start_line = draw(st.integers(min_value=1, max_value=100))
    return MockSearchResult(
        chunk_id=f"chunk_{idx}",
        file_path=f"src/test/file_{idx}.py",
        start_line=start_line,
        end_line=start_line + draw(st.integers(min_value=1, max_value=50)),
        content=f"content for chunk {idx}",
        score=draw(st.floats(min_value=0.1, max_value=1.0)),
        metadata={"artifact_type": artifact_type},
    )


@st.composite
def summary_results_strategy(draw):
    """Generate a list of summary artifact results."""
    num_results = draw(st.integers(min_value=1, max_value=5))
    results = []
    for i in range(num_results):
        summary_type = draw(summary_type_strategy)
        result = draw(mock_search_result_strategy(artifact_type=summary_type))
        # Ensure unique chunk_ids
        result.chunk_id = f"{result.chunk_id}_{i}"
        results.append(result)
    return results


@st.composite
def mixed_results_strategy(draw):
    """Generate a list of mixed artifact type results."""
    num_results = draw(st.integers(min_value=2, max_value=8))
    results = []
    for i in range(num_results):
        artifact_type = draw(artifact_type_strategy)
        result = draw(mock_search_result_strategy(artifact_type=artifact_type))
        result.chunk_id = f"{result.chunk_id}_{i}"
        results.append(result)
    return results


class TestSummaryArtifactResultsHaveArtifactTypePopulated:
    """
    **Feature: hybrid-search-modes, Property 6: Summary artifact results have artifact_type populated**
    **Validates: Requirements 2.3, 6.1**

    For any search result returned when querying for summary artifacts,
    the result SHALL have a non-null artifact_type metadata field.
    """

    @given(
        query=query_strategy,
        results=summary_results_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    def test_summary_mode_results_have_artifact_type(self, query, results):
        """SUMMARY mode results should have artifact_type populated."""
        mock_vector = MockVectorStoreWithResults(results)
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
        )

        search_results = run_async(
            search_service.search(query, search_mode=SearchMode.SUMMARY)
        )

        # All results should have artifact_type populated
        for result in search_results:
            assert "artifact_type" in result.metadata, (
                f"Result {result.chunk_id} missing artifact_type in metadata"
            )
            assert result.metadata["artifact_type"] is not None, (
                f"Result {result.chunk_id} has null artifact_type"
            )
            assert result.metadata["artifact_type"] in SUMMARY_TYPES, (
                f"Result {result.chunk_id} has non-summary artifact_type: "
                f"{result.metadata['artifact_type']}"
            )

    @given(
        query=query_strategy,
        summary_type=summary_type_strategy,
        results=summary_results_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    def test_specific_summary_type_results_have_artifact_type(
        self, query, summary_type, results
    ):
        """Searching for specific summary type should return results with that type."""
        mock_vector = MockVectorStoreWithResults(results)
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
        )

        search_results = run_async(
            search_service.search(
                query,
                search_mode=SearchMode.VECTOR,
                artifact_types=[summary_type],
            )
        )

        # All results should have artifact_type populated and match filter
        for result in search_results:
            assert "artifact_type" in result.metadata, (
                f"Result {result.chunk_id} missing artifact_type"
            )
            assert result.metadata["artifact_type"] is not None, (
                f"Result {result.chunk_id} has null artifact_type"
            )


class TestVectorSearchRespectsArtifactTypeFilter:
    """
    **Feature: hybrid-search-modes, Property 3: Vector search respects artifact_type filter**
    **Validates: Requirements 1.3, 6.3**

    For any vector search with artifact_types filter, all returned results
    SHALL have an artifact_type metadata value that is contained in the
    specified filter list.
    """

    @given(
        query=query_strategy,
        results=mixed_results_strategy(),
        filter_types=st.lists(
            artifact_type_strategy,
            min_size=1,
            max_size=len(ARTIFACT_TYPES),
            unique=True,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_vector_search_filters_by_artifact_type(self, query, results, filter_types):
        """Vector search should only return results matching artifact_type filter."""
        mock_vector = MockVectorStoreWithResults(results)
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

        # All results should have artifact_type in the filter list
        for result in search_results:
            assert "artifact_type" in result.metadata, (
                f"Result {result.chunk_id} missing artifact_type"
            )
            assert result.metadata["artifact_type"] in filter_types, (
                f"Result {result.chunk_id} has artifact_type "
                f"'{result.metadata['artifact_type']}' not in filter {filter_types}"
            )

    @given(
        query=query_strategy,
        results=mixed_results_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    def test_vector_search_passes_artifact_types_to_store(self, query, results):
        """Vector search should pass artifact_types filter to vector store."""
        mock_vector = MockVectorStoreWithResults(results)
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
        )

        filter_types = ["function_summary", "class_summary"]
        run_async(
            search_service.search(
                query,
                search_mode=SearchMode.VECTOR,
                artifact_types=filter_types,
            )
        )

        # Verify artifact_types was passed to vector store
        assert mock_vector.search_called
        assert mock_vector.last_artifact_types == filter_types

    @given(
        query=query_strategy,
        results=mixed_results_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    def test_hybrid_search_passes_artifact_types_to_vector_store(self, query, results):
        """Hybrid search should pass artifact_types filter to vector store."""
        mock_vector = MockVectorStoreWithResults(results)
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
        )

        filter_types = ["chunk", "function_summary"]
        run_async(
            search_service.search(
                query,
                search_mode=SearchMode.HYBRID,
                artifact_types=filter_types,
            )
        )

        # Verify artifact_types was passed to vector store
        assert mock_vector.search_called
        assert mock_vector.last_artifact_types == filter_types



class TestMissingArtifactTypeDefaultsToChunk:
    """
    **Feature: hybrid-search-modes, Property 11: Missing artifact_type defaults to chunk**
    **Validates: Requirements 6.4**

    For any stored vector data without artifact_type metadata, when filtering
    by artifact_types containing "chunk", the data SHALL be included in results.
    """

    @given(
        query=query_strategy,
        num_results=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_artifact_type_treated_as_chunk(self, query, num_results):
        """Results without artifact_type should be treated as 'chunk'."""
        # Create results without artifact_type in metadata
        results = []
        for i in range(num_results):
            results.append(MockSearchResult(
                chunk_id=f"chunk_{i}",
                file_path=f"src/test/file_{i}.py",
                start_line=1,
                end_line=10,
                content=f"content for chunk {i}",
                score=0.9 - (i * 0.1),
                metadata={},  # No artifact_type
            ))

        mock_vector = MockVectorStoreWithResults(results)
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
        )

        # Search with artifact_types containing "chunk"
        search_results = run_async(
            search_service.search(
                query,
                search_mode=SearchMode.VECTOR,
                artifact_types=["chunk"],
            )
        )

        # Results without artifact_type should be included (treated as "chunk")
        assert len(search_results) == num_results, (
            f"Expected {num_results} results, got {len(search_results)}"
        )

    @given(
        query=query_strategy,
        num_results=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_artifact_type_excluded_when_filtering_summaries(
        self, query, num_results
    ):
        """Results without artifact_type should be excluded when filtering for summaries."""
        # Create results without artifact_type in metadata
        results = []
        for i in range(num_results):
            results.append(MockSearchResult(
                chunk_id=f"chunk_{i}",
                file_path=f"src/test/file_{i}.py",
                start_line=1,
                end_line=10,
                content=f"content for chunk {i}",
                score=0.9 - (i * 0.1),
                metadata={},  # No artifact_type
            ))

        mock_vector = MockVectorStoreWithResults(results)
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
        )

        # Search with artifact_types containing only summary types
        search_results = run_async(
            search_service.search(
                query,
                search_mode=SearchMode.VECTOR,
                artifact_types=["function_summary", "class_summary"],
            )
        )

        # Results without artifact_type (treated as "chunk") should be excluded
        assert len(search_results) == 0, (
            f"Expected 0 results when filtering for summaries, got {len(search_results)}"
        )

    @given(
        query=query_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_mixed_results_with_missing_artifact_type(self, query):
        """Mixed results with and without artifact_type should filter correctly."""
        # Create mixed results
        results = [
            MockSearchResult(
                chunk_id="chunk_with_type",
                file_path="src/test/file1.py",
                start_line=1,
                end_line=10,
                content="content with type",
                score=0.9,
                metadata={"artifact_type": "chunk"},
            ),
            MockSearchResult(
                chunk_id="chunk_without_type",
                file_path="src/test/file2.py",
                start_line=1,
                end_line=10,
                content="content without type",
                score=0.8,
                metadata={},  # No artifact_type - should default to "chunk"
            ),
            MockSearchResult(
                chunk_id="summary_with_type",
                file_path="src/test/file3.py",
                start_line=1,
                end_line=10,
                content="summary content",
                score=0.7,
                metadata={"artifact_type": "function_summary"},
            ),
        ]

        mock_vector = MockVectorStoreWithResults(results)
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
        )

        # Search with artifact_types containing "chunk"
        search_results = run_async(
            search_service.search(
                query,
                search_mode=SearchMode.VECTOR,
                artifact_types=["chunk"],
            )
        )

        # Should include both chunks (with and without explicit artifact_type)
        assert len(search_results) == 2, (
            f"Expected 2 chunk results, got {len(search_results)}"
        )
        chunk_ids = {r.chunk_id for r in search_results}
        assert "chunk_with_type" in chunk_ids
        assert "chunk_without_type" in chunk_ids
        assert "summary_with_type" not in chunk_ids

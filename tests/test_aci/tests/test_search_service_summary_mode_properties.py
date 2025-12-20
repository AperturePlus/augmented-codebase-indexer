"""Property-based tests for SearchService SUMMARY mode behavior.

**Feature: hybrid-search-modes, Property 7: SUMMARY mode searches only summary types**
**Feature: hybrid-search-modes, Property 8: SUMMARY mode excludes grep**
**Feature: hybrid-search-modes, Property 9: HYBRID with summary types equals SUMMARY mode**
**Validates: Requirements 3.1, 3.2, 3.3, 3.4**
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.services.search_service import SearchService
from aci.services.search_types import SearchMode
from tests.search_service_test_utils import run_async


class MockGrepSearcher:
    """Mock grep searcher that tracks invocations."""

    def __init__(self):
        self.search_called = False
        self.search_count = 0

    async def search(
        self,
        query: str,
        file_paths: list,
        limit: int = 20,
        context_lines: int = 3,
        case_sensitive: bool = False,
        file_filter: str = None,
    ) -> list:
        self.search_called = True
        self.search_count += 1
        return []


class MockVectorStore:
    """Mock vector store that tracks invocations and artifact_types passed."""

    def __init__(self, results: list = None):
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
    ) -> list:
        self.search_called = True
        self.search_count += 1
        self.last_artifact_types = artifact_types
        return self._results

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


class TestSummaryModeSearchesOnlySummaryTypes:
    """
    **Feature: hybrid-search-modes, Property 7: SUMMARY mode searches only summary types**
    **Validates: Requirements 3.1, 3.4**

    For any search with SearchMode.SUMMARY, the effective artifact_types filter
    SHALL be exactly ["function_summary", "class_summary", "file_summary"].
    """

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_summary_mode_sets_summary_artifact_types(self, query):
        """SUMMARY mode should set artifact_types to summary types only."""
        assume(query.strip())

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(search_service.search(query, search_mode=SearchMode.SUMMARY))

        assert mock_vector.search_called
        expected_types = ["function_summary", "class_summary", "file_summary"]
        assert mock_vector.last_artifact_types == expected_types


class TestSummaryModeExcludesGrep:
    """
    **Feature: hybrid-search-modes, Property 8: SUMMARY mode excludes grep**
    **Validates: Requirements 3.2**

    For any search with SearchMode.SUMMARY, grep search SHALL NOT be invoked.
    """

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_summary_mode_does_not_invoke_grep(self, query):
        """SUMMARY mode should not invoke grep searcher."""
        assume(query.strip())

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(search_service.search(query, search_mode=SearchMode.SUMMARY))

        assert mock_vector.search_called
        assert not mock_grep.search_called

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
        file_filter=st.one_of(st.none(), st.just("*.py"), st.just("src/**/*.py")),
    )
    @settings(max_examples=100, deadline=None)
    def test_summary_mode_excludes_grep_with_file_filter(self, query, file_filter):
        """SUMMARY mode should not invoke grep even with file filter."""
        assume(query.strip())

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(
            search_service.search(
                query, search_mode=SearchMode.SUMMARY, file_filter=file_filter
            )
        )

        assert mock_vector.search_called
        assert not mock_grep.search_called


class TestHybridWithSummaryTypesEqualsSummaryMode:
    """
    **Feature: hybrid-search-modes, Property 9: HYBRID with summary types equals SUMMARY mode**
    **Validates: Requirements 3.3**

    For any query, searching with SearchMode.HYBRID and
    artifact_types=["function_summary", "class_summary", "file_summary"]
    SHALL produce the same results as searching with SearchMode.SUMMARY.
    """

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
        file_filter=st.one_of(st.none(), st.just("*.py"), st.just("src/**/*.py")),
    )
    @settings(max_examples=100, deadline=None)
    def test_hybrid_with_summary_types_matches_summary_mode_behavior(
        self, query, file_filter
    ):
        """HYBRID with summary artifact_types should behave like SUMMARY mode."""
        assume(query.strip())

        summary_types = ["function_summary", "class_summary", "file_summary"]

        # Test SUMMARY mode
        mock_grep_summary = MockGrepSearcher()
        mock_vector_summary = MockVectorStore()
        mock_embedding_summary = MockEmbeddingClient()

        search_service_summary = SearchService(
            embedding_client=mock_embedding_summary,
            vector_store=mock_vector_summary,
            grep_searcher=mock_grep_summary,
        )

        run_async(
            search_service_summary.search(
                query, search_mode=SearchMode.SUMMARY, file_filter=file_filter
            )
        )

        # Test HYBRID mode with summary artifact_types
        mock_grep_hybrid = MockGrepSearcher()
        mock_vector_hybrid = MockVectorStore()
        mock_embedding_hybrid = MockEmbeddingClient()

        search_service_hybrid = SearchService(
            embedding_client=mock_embedding_hybrid,
            vector_store=mock_vector_hybrid,
            grep_searcher=mock_grep_hybrid,
        )

        run_async(
            search_service_hybrid.search(
                query,
                search_mode=SearchMode.HYBRID,
                artifact_types=summary_types,
                file_filter=file_filter,
            )
        )

        # Both modes should:
        # 1. Call vector search
        assert mock_vector_summary.search_called
        assert mock_vector_hybrid.search_called

        # 2. NOT call grep search
        assert not mock_grep_summary.search_called
        assert not mock_grep_hybrid.search_called

        # 3. Use the same artifact_types for vector search
        assert mock_vector_summary.last_artifact_types == summary_types
        assert mock_vector_hybrid.last_artifact_types == summary_types

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_hybrid_with_summary_types_same_vector_invocation_count(self, query):
        """HYBRID with summary types should invoke vector search same number of times as SUMMARY."""
        assume(query.strip())

        summary_types = ["function_summary", "class_summary", "file_summary"]

        # Test SUMMARY mode
        mock_vector_summary = MockVectorStore()
        search_service_summary = SearchService(
            embedding_client=MockEmbeddingClient(),
            vector_store=mock_vector_summary,
            grep_searcher=MockGrepSearcher(),
        )
        run_async(search_service_summary.search(query, search_mode=SearchMode.SUMMARY))

        # Test HYBRID mode with summary artifact_types
        mock_vector_hybrid = MockVectorStore()
        search_service_hybrid = SearchService(
            embedding_client=MockEmbeddingClient(),
            vector_store=mock_vector_hybrid,
            grep_searcher=MockGrepSearcher(),
        )
        run_async(
            search_service_hybrid.search(
                query, search_mode=SearchMode.HYBRID, artifact_types=summary_types
            )
        )

        # Both should have same number of vector search invocations
        assert mock_vector_summary.search_count == mock_vector_hybrid.search_count

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_hybrid_with_summary_types_zero_grep_invocations(self, query):
        """HYBRID with summary types should have zero grep invocations like SUMMARY."""
        assume(query.strip())

        summary_types = ["function_summary", "class_summary", "file_summary"]

        # Test SUMMARY mode
        mock_grep_summary = MockGrepSearcher()
        search_service_summary = SearchService(
            embedding_client=MockEmbeddingClient(),
            vector_store=MockVectorStore(),
            grep_searcher=mock_grep_summary,
        )
        run_async(search_service_summary.search(query, search_mode=SearchMode.SUMMARY))

        # Test HYBRID mode with summary artifact_types
        mock_grep_hybrid = MockGrepSearcher()
        search_service_hybrid = SearchService(
            embedding_client=MockEmbeddingClient(),
            vector_store=MockVectorStore(),
            grep_searcher=mock_grep_hybrid,
        )
        run_async(
            search_service_hybrid.search(
                query, search_mode=SearchMode.HYBRID, artifact_types=summary_types
            )
        )

        # Both should have zero grep invocations
        assert mock_grep_summary.search_count == 0
        assert mock_grep_hybrid.search_count == 0
        assert mock_grep_summary.search_count == mock_grep_hybrid.search_count

"""Property-based tests for SearchService artifact_types filtering in hybrid search.

**Feature: hybrid-search-modes, Property 1: Hybrid search with artifact_types excludes grep**
**Feature: hybrid-search-modes, Property 4: Summary-only artifact_types triggers vector-only**
**Validates: Requirements 1.1, 2.1**
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.services.search_service import SearchService
from aci.services.search_types import SearchMode
from tests.support.search_service_test_utils import run_async


class MockGrepSearcher:
    """Mock grep searcher that tracks invocations."""

    def __init__(self):
        self.search_called = False
        self.search_count = 0

    async def search(
        self, query: str, file_paths: list, limit: int = 20,
        context_lines: int = 3, case_sensitive: bool = False,
        file_filter: str = None,
    ) -> list:
        self.search_called = True
        self.search_count += 1
        return []


class MockVectorStore:
    """Mock vector store that tracks invocations."""

    def __init__(self, results: list = None):
        self.search_called = False
        self.search_count = 0
        self.last_artifact_types = None
        self._results = results or []

    async def search(
        self, query_vector: list, limit: int = 10, file_filter: str = None,
        collection_name: str = None, artifact_types: list = None,
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


# Strategy for generating non-chunk artifact types (summary types only)
summary_artifact_types = st.lists(
    st.sampled_from(["function_summary", "class_summary", "file_summary"]),
    min_size=1, max_size=3, unique=True,
)


class TestHybridSearchWithArtifactTypesExcludesGrep:
    """
    **Feature: hybrid-search-modes, Property 1: Hybrid search with artifact_types excludes grep**
    **Validates: Requirements 1.1**
    """

    @given(
        query=st.text(min_size=3, max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N"))),
        artifact_types=summary_artifact_types,
    )
    @settings(max_examples=100, deadline=None)
    def test_hybrid_with_non_chunk_artifact_types_excludes_grep(self, query, artifact_types):
        """Hybrid search with artifact_types not containing 'chunk' should skip grep."""
        assume(query.strip())
        assume("chunk" not in artifact_types)

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(search_service.search(
            query, search_mode=SearchMode.HYBRID, artifact_types=artifact_types,
        ))

        assert mock_vector.search_called
        assert not mock_grep.search_called
        assert mock_vector.last_artifact_types == artifact_types



class TestSummaryOnlyArtifactTypesTriggersVectorOnly:
    """
    **Feature: hybrid-search-modes, Property 4: Summary-only artifact_types triggers vector-only**
    **Validates: Requirements 2.1**
    """

    @given(
        query=st.text(min_size=3, max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N"))),
        artifact_types=summary_artifact_types,
        file_filter=st.one_of(st.none(), st.just("*.py"), st.just("src/**/*.py")),
    )
    @settings(max_examples=100, deadline=None)
    def test_summary_only_artifact_types_skips_grep(self, query, artifact_types, file_filter):
        """Summary-only artifact_types should trigger vector-only search."""
        assume(query.strip())

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(search_service.search(
            query, search_mode=SearchMode.HYBRID,
            artifact_types=artifact_types, file_filter=file_filter,
        ))

        assert mock_vector.search_called
        assert not mock_grep.search_called

    @given(
        query=st.text(min_size=3, max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @settings(max_examples=100, deadline=None)
    def test_each_summary_type_individually_skips_grep(self, query):
        """Each individual summary type should skip grep when used alone."""
        assume(query.strip())

        for summary_type in ["function_summary", "class_summary", "file_summary"]:
            mock_grep = MockGrepSearcher()
            mock_vector = MockVectorStore()
            mock_embedding = MockEmbeddingClient()

            search_service = SearchService(
                embedding_client=mock_embedding,
                vector_store=mock_vector,
                grep_searcher=mock_grep,
            )

            run_async(search_service.search(
                query, search_mode=SearchMode.HYBRID, artifact_types=[summary_type],
            ))

            assert mock_vector.search_called, f"Vector not called for {summary_type}"
            assert not mock_grep.search_called, f"Grep called for {summary_type}"



# Strategy for generating artifact types that include "chunk"
artifact_types_with_chunk = st.lists(
    st.sampled_from(["chunk", "function_summary", "class_summary", "file_summary"]),
    min_size=1, max_size=4, unique=True,
).filter(lambda x: "chunk" in x)


class TestHybridSearchWithoutArtifactTypesIncludesBoth:
    """
    **Feature: hybrid-search-modes, Property 2: Hybrid search without artifact_types includes both**
    **Validates: Requirements 1.2**
    """

    @given(
        query=st.text(min_size=3, max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N"))),
        file_filter=st.one_of(st.none(), st.just("*.py")),
    )
    @settings(max_examples=100, deadline=None)
    def test_hybrid_without_artifact_types_includes_grep(self, query, file_filter):
        """Hybrid search without artifact_types should include grep search."""
        assume(query.strip())

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(search_service.search(
            query, search_mode=SearchMode.HYBRID,
            artifact_types=None, file_filter=file_filter,
        ))

        # Both vector and grep should be called when artifact_types is None
        assert mock_vector.search_called
        assert mock_grep.search_called


class TestChunkInArtifactTypesEnablesGrep:
    """
    **Feature: hybrid-search-modes, Property 5: Chunk in artifact_types enables grep in hybrid**
    **Validates: Requirements 2.2**
    """

    @given(
        query=st.text(min_size=3, max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N"))),
        artifact_types=artifact_types_with_chunk,
    )
    @settings(max_examples=100, deadline=None)
    def test_chunk_in_artifact_types_enables_grep(self, query, artifact_types):
        """Hybrid search with 'chunk' in artifact_types should include grep."""
        assume(query.strip())
        assume("chunk" in artifact_types)

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(search_service.search(
            query, search_mode=SearchMode.HYBRID, artifact_types=artifact_types,
        ))

        # Both vector and grep should be called when artifact_types contains "chunk"
        assert mock_vector.search_called
        assert mock_grep.search_called

    @given(
        query=st.text(min_size=3, max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @settings(max_examples=100, deadline=None)
    def test_chunk_only_artifact_type_enables_grep(self, query):
        """Hybrid search with only 'chunk' artifact_type should include grep."""
        assume(query.strip())

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(search_service.search(
            query, search_mode=SearchMode.HYBRID, artifact_types=["chunk"],
        ))

        assert mock_vector.search_called
        assert mock_grep.search_called

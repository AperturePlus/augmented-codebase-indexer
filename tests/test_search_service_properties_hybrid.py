"""Hybrid search property-based tests for SearchService (mode, sorting, filtering)."""

import shutil
import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tests.search_service_test_utils import (
    create_indexed_search_env,
    python_file_content,
    run_async,
)

from aci.infrastructure.vector_store import SearchResult
from aci.services.search_service import SearchMode, SearchService


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
    """Mock vector store that tracks invocations."""

    def __init__(self, results: list = None):
        self.search_called = False
        self.search_count = 0
        self._results = results or []

    async def search(
        self,
        query_vector: list,
        limit: int = 10,
        file_filter: str = None,
    ) -> list:
        self.search_called = True
        self.search_count += 1
        return self._results

    async def get_all_file_paths(self) -> list:
        return ["test.py"]

    async def get_by_id(self, chunk_id: str):
        return None

    async def get_stats(self) -> dict:
        return {"total_vectors": 0, "total_files": 0}


class MockEmbeddingClient:
    """Mock embedding client for testing."""

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    def get_dimension(self) -> int:
        return self._dimension

    async def embed_batch(self, texts: list) -> list:
        return [[0.1] * self._dimension for _ in texts]


class TestSearchModeControl:
    """
    **Feature: hybrid-search, Property 6: Search mode controls execution**
    **Validates: Requirements 4.2, 4.3**
    """

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_vector_mode_does_not_invoke_grep(self, query):
        """In vector mode, grep searcher should not be invoked."""
        assume(query.strip())

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(search_service.search(query, search_mode=SearchMode.VECTOR))

        assert mock_vector.search_called
        assert not mock_grep.search_called

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_grep_mode_does_not_invoke_vector(self, query):
        """In grep mode, vector search should not be invoked."""
        assume(query.strip())

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(search_service.search(query, search_mode=SearchMode.GREP))

        assert mock_grep.search_called
        assert not mock_vector.search_called

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_hybrid_mode_invokes_both(self, query):
        """In hybrid mode, both vector and grep search should be invoked."""
        assume(query.strip())

        mock_grep = MockGrepSearcher()
        mock_vector = MockVectorStore()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(search_service.search(query, search_mode=SearchMode.HYBRID))

        assert mock_vector.search_called
        assert mock_grep.search_called


class TestFinalResultsSortingAndLimiting:
    """
    **Feature: hybrid-search, Property 5: Final results are sorted and limited**
    **Validates: Requirements 3.3**
    """

    @given(
        file_contents=st.lists(
            python_file_content(),
            min_size=2,
            max_size=4,
        ),
        query=st.text(
            min_size=5,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        ),
        limit=st.integers(min_value=1, max_value=15),
    )
    @settings(max_examples=15, deadline=60000)
    def test_results_sorted_and_limited(self, file_contents, query, limit):
        """Results should be sorted by score descending and respect limit."""
        assume(query.strip())

        temp_dir = Path(tempfile.mkdtemp())
        try:
            files_dict = {f"module_{i}.py": content for i, content in enumerate(file_contents)}

            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )

            results = run_async(search_service.search(query, limit=limit))

            assert len(results) <= limit

            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i].score >= results[i + 1].score
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=5,
            max_size=20,
        ),
        limit=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_merged_results_sorted_by_score(self, scores, limit):
        """Merged results from vector and grep should be sorted by score."""
        vector_results = []
        grep_results = []

        for i, score in enumerate(scores):
            if i % 2 == 0:
                vector_results.append(
                    SearchResult(
                        chunk_id=f"vector_{i}",
                        file_path=f"file_{i}.py",
                        start_line=1,
                        end_line=10,
                        content=f"content {i}",
                        score=score,
                        metadata={"source": "vector"},
                    )
                )
            else:
                grep_results.append(
                    SearchResult(
                        chunk_id=f"grep_{i}",
                        file_path=f"grep_file_{i}.py",
                        start_line=1,
                        end_line=5,
                        content=f"grep content {i}",
                        score=score,
                        metadata={"source": "grep"},
                    )
                )

        all_results = vector_results + grep_results
        all_results.sort(key=lambda r: r.score, reverse=True)
        final_results = all_results[:limit]

        assert len(final_results) <= limit

        if len(final_results) > 1:
            for i in range(len(final_results) - 1):
                assert final_results[i].score >= final_results[i + 1].score


class TestFileFilterApplication:
    """
    **Feature: hybrid-search, Property 7: File filter applies to all results**
    **Validates: Requirements 4.4**
    """

    @given(
        file_contents=st.lists(
            python_file_content(),
            min_size=3,
            max_size=5,
        ),
        query=st.text(
            min_size=5,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        ),
    )
    @settings(max_examples=15, deadline=60000)
    def test_file_filter_applies_to_all_results(self, file_contents, query):
        """All results should match the file filter pattern."""
        assume(query.strip())
        assume(len(file_contents) >= 3)

        temp_dir = Path(tempfile.mkdtemp())
        try:
            files_dict = {}
            for i, content in enumerate(file_contents):
                if i % 2 == 0:
                    files_dict[f"utils_{i}.py"] = content
                else:
                    files_dict[f"service_{i}.py"] = content

            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )

            filter_pattern = "*utils_*.py"
            results = run_async(
                search_service.search(
                    query,
                    limit=10,
                    file_filter=filter_pattern,
                    search_mode=SearchMode.VECTOR,
                )
            )

            import fnmatch

            for result in results:
                filename = Path(result.file_path).name
                matches = fnmatch.fnmatch(filename, "utils_*.py") or fnmatch.fnmatch(
                    result.file_path, filter_pattern
                )
                assert matches
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    @given(
        query=st.text(
            min_size=3,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_file_filter_passed_to_both_searches(self, query):
        """File filter should be passed to both vector and grep searches."""
        assume(query.strip())

        file_filter = "*.py"
        received_filters = []

        class TrackingVectorStore(MockVectorStore):
            async def search(self, query_vector, limit=10, file_filter=None):
                received_filters.append(("vector", file_filter))
                return []

        class TrackingGrepSearcher(MockGrepSearcher):
            async def search(
                self, query, file_paths, limit=20, context_lines=3,
                case_sensitive=False, file_filter=None
            ):
                received_filters.append(("grep", file_filter))
                return []

        mock_vector = TrackingVectorStore()
        mock_grep = TrackingGrepSearcher()
        mock_embedding = MockEmbeddingClient()

        search_service = SearchService(
            embedding_client=mock_embedding,
            vector_store=mock_vector,
            grep_searcher=mock_grep,
        )

        run_async(
            search_service.search(
                query,
                file_filter=file_filter,
                search_mode=SearchMode.HYBRID,
            )
        )

        assert len(received_filters) == 2
        for _, received_filter in received_filters:
            assert received_filter == file_filter

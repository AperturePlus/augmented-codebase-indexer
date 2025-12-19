"""Property-based tests for SearchService vector behaviors."""

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


class TestSearchResultsOrdering:
    """
    **Feature: codebase-semantic-search, Property 11: Search Results Ordering**
    **Validates: Requirements 4.2**

    *For any* search query, returned results should be sorted by
    similarity score in descending order (highest score first).
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
    )
    @settings(
        max_examples=15,
        deadline=60000,
    )
    def test_results_sorted_by_score_descending(self, file_contents, query):
        """Search results should be sorted by score in descending order."""
        assume(query.strip())  # Non-empty query

        temp_dir = Path(tempfile.mkdtemp())
        try:
            files_dict = {f"module_{i}.py": content for i, content in enumerate(file_contents)}

            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )

            results = run_async(search_service.search(query, limit=10))

            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i].score >= results[i + 1].score, (
                        f"Results not sorted: {results[i].score} < {results[i + 1].score}"
                    )
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSearchResultCompleteness:
    """
    **Feature: codebase-semantic-search, Property 12: Search Result Completeness**
    **Validates: Requirements 4.3**

    *For any* SearchResult returned by the search engine, file_path,
    start_line, end_line, and score fields should not be null or empty.
    """

    @given(
        file_contents=st.lists(
            python_file_content(),
            min_size=1,
            max_size=3,
        ),
        query=st.text(
            min_size=5,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        ),
    )
    @settings(
        max_examples=15,
        deadline=60000,
    )
    def test_results_have_complete_fields(self, file_contents, query):
        """All search results should have complete required fields."""
        assume(query.strip())

        temp_dir = Path(tempfile.mkdtemp())
        try:
            files_dict = {f"module_{i}.py": content for i, content in enumerate(file_contents)}

            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )

            results = run_async(search_service.search(query, limit=10))

            for result in results:
                assert result.file_path, "file_path should not be empty"
                assert len(result.file_path) > 0, "file_path should have content"
                assert result.start_line > 0, f"start_line should be positive, got {result.start_line}"
                assert result.end_line > 0, f"end_line should be positive, got {result.end_line}"
                assert result.end_line >= result.start_line, (
                    f"end_line ({result.end_line}) should be >= start_line ({result.start_line})"
                )
                assert result.score is not None, "score should not be None"
                assert isinstance(result.score, (int, float)), "score should be numeric"
                assert result.content, "content should not be empty"
                assert result.chunk_id, "chunk_id should not be empty"
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSearchResultLimit:
    """
    **Feature: codebase-semantic-search, Property 13: Search Result Limit**
    **Validates: Requirements 4.4**

    *For any* search query with limit K, the number of returned
    results should be <= K.
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
        limit=st.integers(min_value=1, max_value=20),
    )
    @settings(
        max_examples=15,
        deadline=60000,
    )
    def test_results_respect_limit(self, file_contents, query, limit):
        """Search results should not exceed the specified limit."""
        assume(query.strip())

        temp_dir = Path(tempfile.mkdtemp())
        try:
            files_dict = {f"module_{i}.py": content for i, content in enumerate(file_contents)}

            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )

            results = run_async(search_service.search(query, limit=limit))

            assert len(results) <= limit, f"Got {len(results)} results, expected <= {limit}"
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSearchFileFilter:
    """
    **Feature: codebase-semantic-search, Property 14: Search File Filter**
    **Validates: Requirements 4.5**

    *For any* search query with a file path filter pattern, all returned
    results should have file_path matching that filter pattern.
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
    @settings(
        max_examples=15,
        deadline=60000,
    )
    def test_results_match_file_filter(self, file_contents, query):
        """Search results should only include files matching the filter."""
        assume(query.strip())
        assume(len(file_contents) >= 2)

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
            results = run_async(search_service.search(query, limit=10, file_filter=filter_pattern))

            import fnmatch

            for result in results:
                filename = Path(result.file_path).name
                assert fnmatch.fnmatch(filename, "utils_*.py") or fnmatch.fnmatch(
                    result.file_path, filter_pattern
                ), f"Result file_path '{result.file_path}' does not match filter '{filter_pattern}'"
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

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
    )
    @settings(
        max_examples=10,
        deadline=60000,
    )
    def test_exact_file_filter(self, file_contents, query):
        """Search with exact file path should only return results from that file."""
        assume(query.strip())

        temp_dir = Path(tempfile.mkdtemp())
        try:
            files_dict = {f"module_{i}.py": content for i, content in enumerate(file_contents)}

            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )

            target_file = str(temp_dir / "module_0.py")

            results = run_async(search_service.search(query, limit=10, file_filter=target_file))

            for result in results:
                assert result.file_path == target_file, (
                    f"Result from '{result.file_path}' but expected '{target_file}'"
                )
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

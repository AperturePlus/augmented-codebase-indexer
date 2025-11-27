"""
Property-based tests for SearchService.

Tests the correctness properties for search operations including
result ordering, completeness, limits, and file filtering.

Uses InMemoryVectorStore + LocalEmbeddingClient for testing without
external dependencies.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Set

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from aci.core.file_scanner import FileScanner
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.infrastructure.vector_store import SearchResult
from aci.services.indexing_service import IndexingService
from aci.services.search_service import SearchService


# Strategies for generating test data
@st.composite
def python_function_code(draw):
    """Generate valid Python function code."""
    func_name = draw(st.from_regex(r"[a-z][a-z0-9_]{2,10}", fullmatch=True))
    body_lines = draw(st.integers(min_value=1, max_value=5))
    body = "\n".join([f"    x = {i}" for i in range(body_lines)])
    return f"def {func_name}():\n{body}\n    return x\n"


@st.composite
def python_file_content(draw):
    """Generate valid Python file content with multiple functions."""
    num_functions = draw(st.integers(min_value=1, max_value=3))
    functions = [draw(python_function_code()) for _ in range(num_functions)]
    return "\n\n".join(functions)


def create_test_file(directory: Path, filename: str, content: str) -> Path:
    """Create a test file with given content."""
    file_path = directory / filename
    file_path.write_text(content, encoding="utf-8")
    return file_path


def run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def create_indexed_search_env(temp_dir: Path, file_contents: dict):
    """
    Create an indexed environment for search testing.
    
    Args:
        temp_dir: Temporary directory for files
        file_contents: Dict mapping filename to content
        
    Returns:
        Tuple of (search_service, vector_store, metadata_store)
    """
    vector_store = InMemoryVectorStore()
    embedding_client = LocalEmbeddingClient()
    metadata_store = IndexMetadataStore(temp_dir / "metadata.db")
    file_scanner = FileScanner(extensions={".py"})
    
    # Create indexing service
    indexing_service = IndexingService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        max_workers=1,
    )
    
    # Create test files
    for filename, content in file_contents.items():
        create_test_file(temp_dir, filename, content)
    
    # Index the directory
    run_async(indexing_service.index_directory(temp_dir))
    
    # Create search service
    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=None,
        default_limit=10,
    )
    
    return search_service, vector_store, metadata_store


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
        query=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'))),
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
            # Create file contents dict
            files_dict = {f"module_{i}.py": content for i, content in enumerate(file_contents)}
            
            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )
            
            # Perform search
            results = run_async(search_service.search(query, limit=10))
            
            # Verify ordering
            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i].score >= results[i + 1].score, \
                        f"Results not sorted: {results[i].score} < {results[i + 1].score}"
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
        query=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'))),
    )
    @settings(
        max_examples=15,
        deadline=60000,
    )
    def test_results_have_complete_fields(self, file_contents, query):
        """All search results should have complete required fields."""
        assume(query.strip())  # Non-empty query
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create file contents dict
            files_dict = {f"module_{i}.py": content for i, content in enumerate(file_contents)}
            
            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )
            
            # Perform search
            results = run_async(search_service.search(query, limit=10))
            
            # Verify completeness of each result
            for result in results:
                # file_path should not be empty
                assert result.file_path, "file_path should not be empty"
                assert len(result.file_path) > 0, "file_path should have content"
                
                # start_line and end_line should be positive
                assert result.start_line > 0, f"start_line should be positive, got {result.start_line}"
                assert result.end_line > 0, f"end_line should be positive, got {result.end_line}"
                assert result.end_line >= result.start_line, \
                    f"end_line ({result.end_line}) should be >= start_line ({result.start_line})"
                
                # score should be a valid number
                assert result.score is not None, "score should not be None"
                assert isinstance(result.score, (int, float)), "score should be numeric"
                
                # content should not be empty
                assert result.content, "content should not be empty"
                
                # chunk_id should not be empty
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
        query=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'))),
        limit=st.integers(min_value=1, max_value=20),
    )
    @settings(
        max_examples=15,
        deadline=60000,
    )
    def test_results_respect_limit(self, file_contents, query, limit):
        """Search results should not exceed the specified limit."""
        assume(query.strip())  # Non-empty query
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create file contents dict
            files_dict = {f"module_{i}.py": content for i, content in enumerate(file_contents)}
            
            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )
            
            # Perform search with limit
            results = run_async(search_service.search(query, limit=limit))
            
            # Verify limit is respected
            assert len(results) <= limit, \
                f"Got {len(results)} results, expected <= {limit}"
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
        query=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'))),
    )
    @settings(
        max_examples=15,
        deadline=60000,
    )
    def test_results_match_file_filter(self, file_contents, query):
        """Search results should only include files matching the filter."""
        assume(query.strip())  # Non-empty query
        assume(len(file_contents) >= 2)  # Need at least 2 files
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create file contents dict with different naming patterns
            files_dict = {}
            for i, content in enumerate(file_contents):
                if i % 2 == 0:
                    files_dict[f"utils_{i}.py"] = content
                else:
                    files_dict[f"service_{i}.py"] = content
            
            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )
            
            # Search with filter for utils_* files
            filter_pattern = "*utils_*.py"
            results = run_async(search_service.search(
                query, 
                limit=10, 
                file_filter=filter_pattern
            ))
            
            # Verify all results match the filter
            import fnmatch
            for result in results:
                # Extract just the filename from the full path
                filename = Path(result.file_path).name
                assert fnmatch.fnmatch(filename, "utils_*.py") or fnmatch.fnmatch(result.file_path, filter_pattern), \
                    f"Result file_path '{result.file_path}' does not match filter '{filter_pattern}'"
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    @given(
        file_contents=st.lists(
            python_file_content(),
            min_size=2,
            max_size=4,
        ),
        query=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z'))),
    )
    @settings(
        max_examples=10,
        deadline=60000,
    )
    def test_exact_file_filter(self, file_contents, query):
        """Search with exact file path should only return results from that file."""
        assume(query.strip())  # Non-empty query
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create file contents dict
            files_dict = {f"module_{i}.py": content for i, content in enumerate(file_contents)}
            
            search_service, vector_store, metadata_store = create_indexed_search_env(
                temp_dir, files_dict
            )
            
            # Get the exact path of the first file
            target_file = str(temp_dir / "module_0.py")
            
            # Search with exact file filter
            results = run_async(search_service.search(
                query, 
                limit=10, 
                file_filter=target_file
            ))
            
            # Verify all results are from the target file
            for result in results:
                assert result.file_path == target_file, \
                    f"Result from '{result.file_path}' but expected '{target_file}'"
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

"""
Property-based tests for GrepSearcher.

Tests the grep-style text search functionality for hybrid search.
"""

import asyncio
import os
import tempfile

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aci.infrastructure.grep_searcher import GrepSearcher

# Strategies for generating test data

# Non-empty text that can be searched for (ASCII alphanumeric to avoid Unicode case issues)
ASCII_ALPHANUMERIC = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
query_strategy = st.text(
    alphabet=st.sampled_from(ASCII_ALPHANUMERIC),
    min_size=1,
    max_size=20,
).filter(lambda x: x.strip() != "")

# File content lines - printable ASCII to avoid encoding issues
line_content_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        whitelist_characters=" \t",
    ),
    min_size=0,
    max_size=100,
)

# Strategy for generating file content with embedded query
@st.composite
def file_with_query_strategy(draw):
    """Generate file content that contains a specific query string exactly once."""
    query = draw(query_strategy)
    num_lines_before = draw(st.integers(min_value=0, max_value=10))
    num_lines_after = draw(st.integers(min_value=0, max_value=10))

    # Generate lines that don't contain the query (non-empty to avoid edge cases)
    lines_before = []
    for _ in range(num_lines_before):
        line = draw(st.text(
            alphabet=st.sampled_from(ASCII_ALPHANUMERIC + " "),
            min_size=1,
            max_size=50,
        ))
        # Ensure line doesn't contain query (case-insensitive)
        while query.lower() in line.lower():
            line = draw(st.text(
                alphabet=st.sampled_from(ASCII_ALPHANUMERIC + " "),
                min_size=1,
                max_size=50,
            ))
        lines_before.append(line)

    # Line containing the query
    prefix = draw(st.text(
        alphabet=st.sampled_from(ASCII_ALPHANUMERIC + " "),
        min_size=0,
        max_size=20,
    ))
    suffix = draw(st.text(
        alphabet=st.sampled_from(ASCII_ALPHANUMERIC + " "),
        min_size=0,
        max_size=20,
    ))
    # Remove any accidental query occurrences in prefix/suffix
    while query.lower() in prefix.lower():
        prefix = draw(st.text(
            alphabet=st.sampled_from(ASCII_ALPHANUMERIC + " "),
            min_size=0,
            max_size=20,
        ))
    while query.lower() in suffix.lower():
        suffix = draw(st.text(
            alphabet=st.sampled_from(ASCII_ALPHANUMERIC + " "),
            min_size=0,
            max_size=20,
        ))
    match_line = f"{prefix}{query}{suffix}"

    lines_after = []
    for _ in range(num_lines_after):
        line = draw(st.text(
            alphabet=st.sampled_from(ASCII_ALPHANUMERIC + " "),
            min_size=1,
            max_size=50,
        ))
        # Ensure line doesn't contain query (case-insensitive)
        while query.lower() in line.lower():
            line = draw(st.text(
                alphabet=st.sampled_from(ASCII_ALPHANUMERIC + " "),
                min_size=1,
                max_size=50,
            ))
        lines_after.append(line)

    all_lines = lines_before + [match_line] + lines_after
    # Join with newlines - no trailing newline
    content = "\n".join(all_lines)
    match_line_num = num_lines_before + 1  # 1-indexed
    total_lines = len(all_lines)

    return {
        "query": query,
        "content": content,
        "match_line_num": match_line_num,
        "total_lines": total_lines,
    }


@st.composite
def file_with_multiple_matches_strategy(draw):
    """Generate file content with multiple matches for a query."""
    query = draw(query_strategy)
    num_matches = draw(st.integers(min_value=1, max_value=5))

    lines = []
    match_line_nums = []

    for _i in range(num_matches):
        # Add some non-matching lines
        num_filler = draw(st.integers(min_value=0, max_value=3))
        for _ in range(num_filler):
            lines.append(draw(line_content_strategy))

        # Add matching line
        prefix = draw(line_content_strategy)
        suffix = draw(line_content_strategy)
        lines.append(f"{prefix}{query}{suffix}")
        match_line_nums.append(len(lines))  # 1-indexed

    # Add trailing lines
    num_trailing = draw(st.integers(min_value=0, max_value=3))
    for _ in range(num_trailing):
        lines.append(draw(line_content_strategy))

    content = "\n".join(lines)

    return {
        "query": query,
        "content": content,
        "match_line_nums": match_line_nums,
        "total_lines": len(lines),
    }


def create_temp_file(content: str, suffix: str = ".txt") -> str:
    """Create a temporary file with given content and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        os.close(fd)
        raise
    return path


@given(data=file_with_query_strategy())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_grep_results_contain_query_text(data: dict):
    """
    **Feature: hybrid-search, Property 1: Grep results contain query text**
    **Validates: Requirements 1.1**

    *For any* search query and any set of files, all results returned by the
    Grep_Searcher SHALL contain the exact query text (respecting case sensitivity settings).
    """
    query = data["query"]
    content = data["content"]

    # Create temp file
    temp_path = create_temp_file(content)
    temp_dir = os.path.dirname(temp_path)
    file_name = os.path.basename(temp_path)

    try:
        searcher = GrepSearcher(base_path=temp_dir)

        async def run_test():
            results = await searcher.search(
                query=query,
                file_paths=[file_name],
                limit=20,
                context_lines=3,
                case_sensitive=True,
            )
            return results

        results = asyncio.run(run_test())

        # All results must contain the query text
        for result in results:
            assert query in result.content, (
                f"Query '{query}' not found in result content: {result.content[:100]}..."
            )
    finally:
        os.unlink(temp_path)


@given(data=file_with_query_strategy(), context_lines=st.integers(min_value=0, max_value=10))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_grep_context_extraction_bounded(data: dict, context_lines: int):
    """
    **Feature: hybrid-search, Property 2: Grep context extraction is bounded correctly**
    **Validates: Requirements 1.2**

    *For any* file with a match at line N, the returned SearchResult SHALL have:
    - start_line = max(1, N - context_lines)
    - end_line = min(file_length, N + context_lines)
    - content containing all lines from start_line to end_line
    """
    query = data["query"]
    content = data["content"]
    match_line_num = data["match_line_num"]
    total_lines = data["total_lines"]

    temp_path = create_temp_file(content)
    temp_dir = os.path.dirname(temp_path)
    file_name = os.path.basename(temp_path)

    try:
        searcher = GrepSearcher(base_path=temp_dir)

        async def run_test():
            results = await searcher.search(
                query=query,
                file_paths=[file_name],
                limit=20,
                context_lines=context_lines,
                case_sensitive=True,
            )
            return results

        results = asyncio.run(run_test())

        # Should have at least one result
        assert len(results) >= 1, "Expected at least one result"

        result = results[0]

        # Verify context bounds
        expected_start = max(1, match_line_num - context_lines)
        expected_end = min(total_lines, match_line_num + context_lines)

        assert result.start_line == expected_start, (
            f"Expected start_line={expected_start}, got {result.start_line}"
        )
        assert result.end_line == expected_end, (
            f"Expected end_line={expected_end}, got {result.end_line}"
        )

        # Verify content contains the expected number of lines
        content_lines = result.content.rstrip("\n").split("\n")
        expected_line_count = expected_end - expected_start + 1
        assert len(content_lines) == expected_line_count, (
            f"Expected {expected_line_count} lines, got {len(content_lines)}"
        )
    finally:
        os.unlink(temp_path)


@given(data=file_with_query_strategy())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_grep_results_valid_format(data: dict):
    """
    **Feature: hybrid-search, Property 3: Grep results have valid SearchResult format**
    **Validates: Requirements 1.3**

    *For any* grep search result, the SearchResult object SHALL have:
    - Non-empty chunk_id starting with "grep:"
    - Non-empty file_path that exists
    - start_line >= 1
    - end_line >= start_line
    - Non-empty content
    - metadata containing "source": "grep"
    """
    query = data["query"]
    content = data["content"]

    temp_path = create_temp_file(content)
    temp_dir = os.path.dirname(temp_path)
    file_name = os.path.basename(temp_path)

    try:
        searcher = GrepSearcher(base_path=temp_dir)

        async def run_test():
            results = await searcher.search(
                query=query,
                file_paths=[file_name],
                limit=20,
                context_lines=3,
                case_sensitive=True,
            )
            return results

        results = asyncio.run(run_test())

        for i, result in enumerate(results):
            # chunk_id starts with "grep:"
            assert result.chunk_id.startswith("grep:"), (
                f"Result {i}: chunk_id should start with 'grep:', got '{result.chunk_id}'"
            )
            assert len(result.chunk_id) > 5, (
                f"Result {i}: chunk_id too short: '{result.chunk_id}'"
            )

            # file_path is non-empty
            assert result.file_path, f"Result {i}: file_path is empty"

            # start_line >= 1
            assert result.start_line >= 1, (
                f"Result {i}: start_line should be >= 1, got {result.start_line}"
            )

            # end_line >= start_line
            assert result.end_line >= result.start_line, (
                f"Result {i}: end_line ({result.end_line}) < start_line ({result.start_line})"
            )

            # content is non-empty
            assert result.content, f"Result {i}: content is empty"

            # metadata contains source: grep
            assert result.metadata.get("source") == "grep", (
                f"Result {i}: metadata should contain 'source': 'grep', got {result.metadata}"
            )
    finally:
        os.unlink(temp_path)


@given(
    data=file_with_multiple_matches_strategy(),
    limit=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_grep_respects_result_limit(data: dict, limit: int):
    """
    **Feature: hybrid-search, Property 8: Grep respects result limit**
    **Validates: Requirements 5.3**

    *For any* grep search with limit N, the number of returned results SHALL be <= N.
    """
    query = data["query"]
    content = data["content"]

    temp_path = create_temp_file(content)
    temp_dir = os.path.dirname(temp_path)
    file_name = os.path.basename(temp_path)

    try:
        searcher = GrepSearcher(base_path=temp_dir)

        async def run_test():
            results = await searcher.search(
                query=query,
                file_paths=[file_name],
                limit=limit,
                context_lines=3,
                case_sensitive=True,
            )
            return results

        results = asyncio.run(run_test())

        assert len(results) <= limit, (
            f"Got {len(results)} results, expected <= {limit}"
        )
    finally:
        os.unlink(temp_path)


@given(data=file_with_query_strategy())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_case_insensitive_grep_matches(data: dict):
    """
    **Feature: hybrid-search, Property 9: Case-insensitive grep matches regardless of case**
    **Validates: Requirements 5.4**

    *For any* query Q and file content C where C contains Q with different casing,
    a case-insensitive grep search SHALL return a match.
    """
    query = data["query"]
    content = data["content"]

    # Skip if query has no letters (case doesn't matter)
    if not any(c.isalpha() for c in query):
        return

    # Modify content to have different case than query
    # Swap case of the query in the content
    swapped_query = query.swapcase()
    modified_content = content.replace(query, swapped_query)

    temp_path = create_temp_file(modified_content)
    temp_dir = os.path.dirname(temp_path)
    file_name = os.path.basename(temp_path)

    try:
        searcher = GrepSearcher(base_path=temp_dir)

        async def run_test():
            # Case-insensitive search should find the swapped-case query
            results = await searcher.search(
                query=query,
                file_paths=[file_name],
                limit=20,
                context_lines=3,
                case_sensitive=False,
            )
            return results

        results = asyncio.run(run_test())

        # Should find at least one match
        assert len(results) >= 1, (
            f"Case-insensitive search for '{query}' should find '{swapped_query}' in content"
        )

        # Verify the match contains the swapped case version
        found_match = False
        for result in results:
            if swapped_query in result.content:
                found_match = True
                break

        assert found_match, (
            f"Expected to find '{swapped_query}' in results"
        )
    finally:
        os.unlink(temp_path)

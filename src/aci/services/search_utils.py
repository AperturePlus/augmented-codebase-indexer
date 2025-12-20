"""
Search utility functions for Project ACI.

Pure functions for deduplication, score normalization, and query parsing.
"""

import fnmatch

from aci.infrastructure.vector_store import SearchResult


def is_near_duplicate(grep_result: SearchResult, vector_results: list[SearchResult]) -> bool:
    """
    Check if grep result overlaps with any vector chunk.

    A grep result is a near-duplicate if:
    - Same file_path as a vector result
    - grep start_line >= vector start_line
    - grep end_line <= vector end_line

    Args:
        grep_result: A single grep search result
        vector_results: List of vector search results

    Returns:
        True if grep result is contained within any vector chunk
    """
    for vr in vector_results:
        if (
            grep_result.file_path == vr.file_path
            and grep_result.start_line >= vr.start_line
            and grep_result.end_line <= vr.end_line
        ):
            return True
    return False


def deduplicate_grep_results(
    grep_results: list[SearchResult], vector_results: list[SearchResult]
) -> list[SearchResult]:
    """
    Filter grep results that overlap with vector chunks.

    Retains only grep results that are NOT contained within any vector chunk.

    Args:
        grep_results: Results from grep search
        vector_results: Results from vector search

    Returns:
        Filtered list of grep results with duplicates removed
    """
    return [gr for gr in grep_results if not is_near_duplicate(gr, vector_results)]



def deduplicate_by_location(results: list[SearchResult]) -> list[SearchResult]:
    """
    Remove duplicate results based on file location.

    When multiple results have the same (file_path, start_line, end_line),
    keeps only the one with the highest score.

    Args:
        results: List of search results (may contain duplicates)

    Returns:
        Deduplicated list with highest-scoring result per location
    """
    seen: dict[tuple[str, int, int], SearchResult] = {}

    for result in results:
        key = (result.file_path, result.start_line, result.end_line)
        if key not in seen or result.score > seen[key].score:
            seen[key] = result

    # Return in original score order
    deduped = list(seen.values())
    deduped.sort(key=lambda r: r.score, reverse=True)
    return deduped


def normalize_scores(
    grep_results: list[SearchResult],
    vector_results: list[SearchResult],
) -> tuple[list[SearchResult], list[SearchResult]]:
    """
    Normalize grep and vector scores to a comparable range.

    Strategy: Scale grep scores relative to the maximum vector score.
    Grep results represent exact matches and should have high but not
    overwhelming scores compared to semantic matches.

    Args:
        grep_results: Results from grep search (typically score=1.0)
        vector_results: Results from vector search (typically 0.0-1.0)

    Returns:
        Tuple of (normalized_grep_results, vector_results)
        Vector results are returned unchanged.
    """
    if not grep_results or not vector_results:
        return grep_results, vector_results

    max_vector_score = max(r.score for r in vector_results)
    if max_vector_score <= 0:
        return grep_results, vector_results

    max_grep_score = max(r.score for r in grep_results)
    if max_grep_score <= 0:
        return grep_results, vector_results

    scale_factor = max_vector_score / max_grep_score

    normalized_grep = [
        SearchResult(
            chunk_id=result.chunk_id,
            file_path=result.file_path,
            start_line=result.start_line,
            end_line=result.end_line,
            content=result.content,
            score=result.score * scale_factor,
            metadata=result.metadata,
        )
        for result in grep_results
    ]

    return normalized_grep, vector_results


def parse_query_modifiers(query: str) -> tuple[str, str | None, list[str]]:
    """
    Parse query string for modifiers like file filters and exclusions.

    Supported syntax:
    - `path:*.py` or `file:src/**` - include only matching paths
    - `-path:tests` or `exclude:tests` - exclude matching paths
    - Multiple exclusions allowed: `-path:tests -path:fixtures`

    Args:
        query: Raw query string with potential modifiers

    Returns:
        Tuple of (clean_query, file_filter, exclude_patterns)

    Examples:
        >>> parse_query_modifiers("parse go syntax -path:tests")
        ("parse go syntax", None, ["tests"])
        >>> parse_query_modifiers("search query path:src/*.py")
        ("search query", "src/*.py", [])
    """
    file_filter = None
    exclude_patterns: list[str] = []
    clean_parts = []

    tokens = query.split()

    for token in tokens:
        if token.startswith("path:") or token.startswith("file:"):
            file_filter = token.split(":", 1)[1]
        elif token.startswith("-path:") or token.startswith("exclude:"):
            pattern = token.split(":", 1)[1]
            if pattern:
                exclude_patterns.append(pattern)
        else:
            clean_parts.append(token)

    clean_query = " ".join(clean_parts).strip()
    return clean_query, file_filter, exclude_patterns


def apply_exclusions(
    results: list[SearchResult], exclude_patterns: list[str]
) -> list[SearchResult]:
    """
    Filter out results matching any exclusion pattern.

    Args:
        results: Search results to filter
        exclude_patterns: Glob patterns to exclude (matched against file_path)

    Returns:
        Filtered results with exclusions removed
    """
    if not exclude_patterns:
        return results

    filtered = []
    for result in results:
        excluded = False
        for pattern in exclude_patterns:
            if pattern in result.file_path or fnmatch.fnmatch(result.file_path, f"*{pattern}*"):
                excluded = True
                break
        if not excluded:
            filtered.append(result)

    return filtered

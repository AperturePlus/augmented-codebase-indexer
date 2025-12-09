"""
Search Service for Project ACI.

Provides semantic search functionality over indexed codebases.
"""

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from aci.infrastructure.embedding_client import EmbeddingClientInterface
from aci.infrastructure.grep_searcher import GrepSearcherInterface
from aci.infrastructure.vector_store import SearchResult, VectorStoreInterface

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Search mode for controlling which search methods are used."""

    HYBRID = "hybrid"
    VECTOR = "vector"
    GREP = "grep"


def _is_near_duplicate(grep_result: SearchResult, vector_results: List[SearchResult]) -> bool:
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


def _deduplicate_results(
    grep_results: List[SearchResult], vector_results: List[SearchResult]
) -> List[SearchResult]:
    """
    Filter grep results that overlap with vector chunks.

    Retains only grep results that are NOT contained within any vector chunk.

    Args:
        grep_results: Results from grep search
        vector_results: Results from vector search

    Returns:
        Filtered list of grep results with duplicates removed
    """
    return [gr for gr in grep_results if not _is_near_duplicate(gr, vector_results)]


def _deduplicate_by_location(results: List[SearchResult]) -> List[SearchResult]:
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


def parse_query_modifiers(query: str) -> tuple[str, Optional[str], List[str]]:
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
    import re

    file_filter = None
    exclude_patterns: List[str] = []
    clean_parts = []

    # Split by whitespace but preserve quoted strings
    tokens = query.split()

    for token in tokens:
        # Check for file filter: path:pattern or file:pattern
        if token.startswith("path:") or token.startswith("file:"):
            file_filter = token.split(":", 1)[1]
        # Check for exclusion: -path:pattern or exclude:pattern
        elif token.startswith("-path:") or token.startswith("exclude:"):
            pattern = token.split(":", 1)[1]
            if pattern:
                exclude_patterns.append(pattern)
        else:
            clean_parts.append(token)

    clean_query = " ".join(clean_parts).strip()
    return clean_query, file_filter, exclude_patterns


def _apply_exclusions(
    results: List[SearchResult], exclude_patterns: List[str]
) -> List[SearchResult]:
    """
    Filter out results matching any exclusion pattern.

    Args:
        results: Search results to filter
        exclude_patterns: Glob patterns to exclude (matched against file_path)

    Returns:
        Filtered results with exclusions removed
    """
    import fnmatch

    if not exclude_patterns:
        return results

    filtered = []
    for result in results:
        excluded = False
        for pattern in exclude_patterns:
            # Match pattern anywhere in path (not just at start)
            if pattern in result.file_path or fnmatch.fnmatch(result.file_path, f"*{pattern}*"):
                excluded = True
                break
        if not excluded:
            filtered.append(result)

    return filtered


class RerankerInterface(ABC):
    """Interface for re-ranking search results."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Re-rank candidates based on query relevance.

        Args:
            query: Original search query
            candidates: Candidate results from vector search
            top_k: Number of results to return

        Returns:
            Re-ranked list of results
        """
        pass


class SearchService:
    """
    Service for semantic code search.

    Converts queries to embeddings, searches the vector store,
    and optionally re-ranks results. Supports hybrid search combining
    vector and grep search methods.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClientInterface,
        vector_store: VectorStoreInterface,
        reranker: Optional[RerankerInterface] = None,
        grep_searcher: Optional[GrepSearcherInterface] = None,
        default_limit: int = 10,
        recall_multiplier: int = 5,
        vector_candidates: int = 20,
        grep_candidates: int = 20,
    ):
        """
        Initialize the search service.

        Args:
            embedding_client: Client for generating query embeddings
            vector_store: Store for vector search
            reranker: Optional re-ranker for result refinement
            grep_searcher: Optional grep searcher for keyword search
            default_limit: Default number of results to return
            recall_multiplier: Multiplier for initial recall when re-ranking
            vector_candidates: Number of candidates to retrieve from vector search
            grep_candidates: Number of candidates to retrieve from grep search
        """
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._reranker = reranker
        self._grep_searcher = grep_searcher
        self._default_limit = default_limit
        self._recall_multiplier = recall_multiplier
        self._vector_candidates = vector_candidates
        self._grep_candidates = grep_candidates

    async def search(
        self,
        query: str,
        limit: Optional[int] = None,
        file_filter: Optional[str] = None,
        use_rerank: bool = True,
        search_mode: SearchMode = SearchMode.HYBRID,
    ) -> List[SearchResult]:
        """
        Perform semantic search.

        Supports query modifiers:
        - `path:*.py` or `file:src/**` - include only matching paths
        - `-path:tests` or `exclude:tests` - exclude matching paths

        Args:
            query: Natural language search query (may include modifiers)
            limit: Maximum results to return (default: default_limit)
            file_filter: Optional glob pattern for file paths
            use_rerank: Whether to use re-ranker if available
            search_mode: Search mode (HYBRID, VECTOR, or GREP)

        Returns:
            List of SearchResult sorted by relevance
        """
        limit = limit or self._default_limit

        # Parse query for modifiers (path filters, exclusions)
        clean_query, query_file_filter, exclude_patterns = parse_query_modifiers(query)

        # Query file_filter overrides parameter if specified
        effective_filter = query_file_filter or file_filter

        # Use clean query for search
        search_query = clean_query if clean_query else query

        # Execute searches based on mode
        vector_results: List[SearchResult] = []
        grep_results: List[SearchResult] = []

        if search_mode == SearchMode.HYBRID:
            # Run both searches in parallel
            vector_results, grep_results = await self._execute_hybrid_search(
                search_query, effective_filter
            )
        elif search_mode == SearchMode.VECTOR:
            vector_results = await self._execute_vector_search(search_query, effective_filter)
        elif search_mode == SearchMode.GREP:
            grep_results = await self._execute_grep_search(search_query, effective_filter)

        # Merge and deduplicate results
        if vector_results and grep_results:
            deduplicated_grep = _deduplicate_results(grep_results, vector_results)
            candidates = vector_results + deduplicated_grep
        elif vector_results:
            candidates = vector_results
        else:
            candidates = grep_results

        # Remove exact location duplicates (same file + line range)
        candidates = _deduplicate_by_location(candidates)

        # Apply exclusion patterns before reranking
        if exclude_patterns:
            candidates = _apply_exclusions(candidates, exclude_patterns)

        # Re-rank if enabled and reranker available
        if use_rerank and self._reranker and candidates:
            reranked = self._reranker.rerank(search_query, candidates, limit)
            if inspect.iscoroutine(reranked):
                results = await reranked
            else:
                results = reranked
        else:
            # Sort by score descending and limit
            candidates.sort(key=lambda r: r.score, reverse=True)
            results = candidates[:limit]

        return results

    async def _execute_vector_search(
        self, query: str, file_filter: Optional[str]
    ) -> List[SearchResult]:
        """Execute vector search and return results."""
        try:
            # Generate query embedding
            embeddings = await self._embedding_client.embed_batch([query])
            query_vector = embeddings[0]

            # Search vector store
            return await self._vector_store.search(
                query_vector=query_vector,
                limit=self._vector_candidates,
                file_filter=file_filter,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _execute_grep_search(
        self, query: str, file_filter: Optional[str]
    ) -> List[SearchResult]:
        """Execute grep search and return results."""
        if not self._grep_searcher:
            return []

        try:
            # Get all indexed file paths from vector store
            file_paths = await self._vector_store.get_all_file_paths()

            return await self._grep_searcher.search(
                query=query,
                file_paths=file_paths,
                limit=self._grep_candidates,
                file_filter=file_filter,
            )
        except Exception as e:
            logger.error(f"Grep search failed: {e}")
            return []

    async def _execute_hybrid_search(
        self, query: str, file_filter: Optional[str]
    ) -> tuple[List[SearchResult], List[SearchResult]]:
        """Execute both vector and grep search in parallel."""
        try:
            vector_task = self._execute_vector_search(query, file_filter)
            grep_task = self._execute_grep_search(query, file_filter)

            vector_results, grep_results = await asyncio.gather(
                vector_task, grep_task, return_exceptions=True
            )

            # Handle exceptions from gather
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed in hybrid mode: {vector_results}")
                vector_results = []
            if isinstance(grep_results, Exception):
                logger.error(f"Grep search failed in hybrid mode: {grep_results}")
                grep_results = []

            return vector_results, grep_results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return [], []

    async def search_by_file(
        self,
        query: str,
        file_path: str,
        limit: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search within a specific file.

        Args:
            query: Natural language search query
            file_path: Exact file path to search in
            limit: Maximum results to return

        Returns:
            List of SearchResult from the specified file
        """
        # Use exact file path as filter
        return await self.search(
            query=query,
            limit=limit,
            file_filter=file_path,
            use_rerank=False,  # Don't rerank for single-file search
        )

    async def get_similar_chunks(
        self,
        chunk_id: str,
        limit: int = 5,
    ) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk.

        Args:
            chunk_id: ID of the reference chunk
            limit: Maximum results to return

        Returns:
            List of similar chunks (excluding the reference)
        """
        # Get the reference chunk
        reference = await self._vector_store.get_by_id(chunk_id)
        if not reference:
            return []

        # Search using the reference content
        results = await self.search(
            query=reference.content,
            limit=limit + 1,  # +1 to account for self-match
            use_rerank=False,
        )

        # Filter out the reference chunk itself
        return [r for r in results if r.chunk_id != chunk_id][:limit]

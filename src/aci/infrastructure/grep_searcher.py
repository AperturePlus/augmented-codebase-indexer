"""
Grep Searcher for Project ACI.

Provides Python-native text-based search for exact keyword matching across files.
"""

import fnmatch
import logging
import os
from abc import ABC, abstractmethod

from aci.infrastructure.vector_store import SearchResult

logger = logging.getLogger(__name__)


class GrepSearcherError(Exception):
    """Base exception for grep searcher errors."""

    pass


class GrepSearcherInterface(ABC):
    """Abstract interface for grep-style text search."""

    @abstractmethod
    async def search(
        self,
        query: str,
        file_paths: list[str],
        limit: int = 20,
        context_lines: int = 3,
        case_sensitive: bool = False,
        file_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for exact text matches in files.

        Args:
            query: Text pattern to search for
            file_paths: List of file paths to search
            limit: Maximum number of results
            context_lines: Lines of context before/after match
            case_sensitive: Whether to match case
            file_filter: Optional glob pattern for file paths

        Returns:
            List of SearchResult with matches and context
        """
        pass


class GrepSearcher(GrepSearcherInterface):
    """
    Python-native grep implementation.

    Uses built-in file I/O for portability.
    Processes files line-by-line to minimize memory usage.
    """

    def __init__(self, base_path: str):
        """
        Initialize grep searcher.

        Args:
            base_path: Root directory for relative file paths
        """
        self._base_path = base_path

    def _is_binary_file(self, file_path: str) -> bool:
        """
        Check if a file is binary by reading first chunk.

        Args:
            file_path: Path to the file

        Returns:
            True if file appears to be binary
        """
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(8192)
                # Check for null bytes which indicate binary content
                return b"\x00" in chunk
        except OSError:
            return True  # Treat unreadable files as binary

    def _resolve_path(self, file_path: str) -> str:
        """
        Resolve a file path relative to base_path.

        Args:
            file_path: Relative or absolute file path

        Returns:
            Absolute file path
        """
        if os.path.isabs(file_path):
            return file_path
        return os.path.join(self._base_path, file_path)

    async def search(
        self,
        query: str,
        file_paths: list[str],
        limit: int = 20,
        context_lines: int = 3,
        case_sensitive: bool = False,
        file_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for exact text matches in files.

        Args:
            query: Text pattern to search for
            file_paths: List of file paths to search
            limit: Maximum number of results
            context_lines: Lines of context before/after match
            case_sensitive: Whether to match case
            file_filter: Optional glob pattern for file paths

        Returns:
            List of SearchResult with matches and context
        """
        # Return empty list for empty query
        if not query or not query.strip():
            return []

        results: list[SearchResult] = []
        search_query = query if case_sensitive else query.lower()

        for file_path in file_paths:
            if len(results) >= limit:
                break

            # Apply file filter if specified
            if file_filter and not fnmatch.fnmatch(file_path, file_filter):
                continue

            resolved_path = self._resolve_path(file_path)

            # Skip if file doesn't exist
            if not os.path.isfile(resolved_path):
                logger.debug(f"Skipping non-existent file: {file_path}")
                continue

            # Skip binary files
            if self._is_binary_file(resolved_path):
                logger.debug(f"Skipping binary file: {file_path}")
                continue

            # Search the file
            file_results = self._search_file(
                file_path=file_path,
                resolved_path=resolved_path,
                search_query=search_query,
                original_query=query,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                remaining_limit=limit - len(results),
            )

            results.extend(file_results)

        return results[:limit]

    def _search_file(
        self,
        file_path: str,
        resolved_path: str,
        search_query: str,
        original_query: str,
        case_sensitive: bool,
        context_lines: int,
        remaining_limit: int,
    ) -> list[SearchResult]:
        """
        Search a single file for matches.

        Args:
            file_path: Original file path (for result)
            resolved_path: Absolute path to file
            search_query: Query to search (already lowercased if case-insensitive)
            original_query: Original query string
            case_sensitive: Whether search is case-sensitive
            context_lines: Lines of context to include
            remaining_limit: Maximum results to return

        Returns:
            List of SearchResult for this file
        """
        results: list[SearchResult] = []

        try:
            with open(resolved_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return []
        except UnicodeDecodeError as e:
            logger.warning(f"Encoding error in file {file_path}: {e}")
            return []

        total_lines = len(lines)

        for line_num, line in enumerate(lines, start=1):
            if len(results) >= remaining_limit:
                break

            # Check for match
            compare_line = line if case_sensitive else line.lower()
            if search_query not in compare_line:
                continue

            # Calculate context bounds
            start_line = max(1, line_num - context_lines)
            end_line = min(total_lines, line_num + context_lines)

            # Extract content with context
            content_lines = lines[start_line - 1 : end_line]
            content = "".join(content_lines)

            # Create SearchResult
            result = SearchResult(
                chunk_id=f"grep:{file_path}:{start_line}",
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                content=content,
                score=1.0,  # Initial score, will be replaced by reranker
                metadata={
                    "source": "grep",
                    "match_line": line_num,
                    "query": original_query,
                },
            )

            results.append(result)

        return results

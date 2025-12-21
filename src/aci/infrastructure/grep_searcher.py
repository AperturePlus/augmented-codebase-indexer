"""
Grep Searcher for Project ACI.

Provides Python-native text search across files.
"""

import fnmatch
import logging
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from heapq import heappush, heappushpop
from itertools import count

from aci.infrastructure.vector_store import SearchResult

logger = logging.getLogger(__name__)


class GrepSearcherError(Exception):
    """Base exception for grep searcher errors."""

    pass


class TextSearchMode(str, Enum):
    """Text search mode."""

    SUBSTRING = "substring"
    REGEX = "regex"
    FUZZY = "fuzzy"


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
        mode: TextSearchMode = TextSearchMode.SUBSTRING,
        all_terms: bool = False,
        fuzzy_min_score: float = 0.6,
    ) -> list[SearchResult]:
        """
        Search for text matches in files.

        Args:
            query: Text/regex query to search for
            file_paths: List of file paths to search
            limit: Maximum number of results
            context_lines: Lines of context before/after match
            case_sensitive: Whether to match case
            file_filter: Optional glob pattern for file paths
            mode: Search mode (substring, regex, fuzzy)
            all_terms: If True, split query by whitespace and require all terms match
            fuzzy_min_score: Minimum per-term fuzzy score (0.0-1.0) to accept a match

        Returns:
            List of SearchResult with matches and context
        """
        raise NotImplementedError


_FUZZY_SEPARATORS = frozenset(
    {
        " ",
        "\t",
        "\r",
        "\n",
        "_",
        "-",
        ".",
        "/",
        "\\",
        ":",
        ";",
        ",",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "'",
        '"',
        "`",
        "|",
        "+",
        "=",
        "*",
        "&",
        "^",
        "%",
        "$",
        "#",
        "@",
        "!",
        "?",
    }
)


def _split_terms(query: str) -> list[str]:
    return [t for t in query.split() if t]


def _is_word_boundary(text: str, index: int) -> bool:
    if index <= 0:
        return True
    prev = text[index - 1]
    curr = text[index]
    if prev in _FUZZY_SEPARATORS:
        return True
    if prev.islower() and curr.isupper():
        return True
    if prev.isdigit() and curr.isalpha():
        return True
    return False


def _fuzzy_subsequence_positions(pattern: str, text: str) -> list[int] | None:
    if not pattern:
        return []

    positions: list[int] = []
    start = 0
    for ch in pattern:
        idx = text.find(ch, start)
        if idx < 0:
            return None
        positions.append(idx)
        start = idx + 1
    return positions


def _fuzzy_score(pattern: str, text: str, original_text: str) -> float:
    if not pattern:
        return 0.0

    if len(pattern) < 2:
        return 1.0 if pattern in text else 0.0

    positions = _fuzzy_subsequence_positions(pattern, text)
    if positions is None:
        return 0.0

    score = float(len(pattern))

    first = positions[0]
    if first == 0:
        score += 2.0
    if _is_word_boundary(original_text, first):
        score += 2.0

    gaps = 0
    consecutive = 0
    for prev, curr in zip(positions, positions[1:], strict=False):
        gap = curr - prev - 1
        if gap == 0:
            consecutive += 1
        else:
            gaps += gap

        if _is_word_boundary(original_text, curr):
            score += 0.5

    score += consecutive * 1.0
    score -= gaps * 0.05
    score -= first * 0.01

    max_score = (
        len(pattern)
        + 4.0
        + max(0, len(pattern) - 1) * 1.0
        + max(0, len(pattern) - 1) * 0.5
    )
    if max_score <= 0:
        return 0.0

    normalized = score / max_score
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


class GrepSearcher(GrepSearcherInterface):
    """
    Python-native grep implementation.

    Uses built-in file I/O for portability.
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
                return b"\x00" in chunk
        except OSError:
            return True

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
        mode: TextSearchMode | str = TextSearchMode.SUBSTRING,
        all_terms: bool = False,
        fuzzy_min_score: float = 0.6,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 20

        try:
            context_lines = max(0, int(context_lines))
        except (TypeError, ValueError):
            context_lines = 3

        try:
            fuzzy_min_score = float(fuzzy_min_score)
        except (TypeError, ValueError):
            fuzzy_min_score = 0.6
        fuzzy_min_score = max(0.0, min(1.0, fuzzy_min_score))

        if not isinstance(mode, TextSearchMode):
            try:
                mode = TextSearchMode(str(mode))
            except ValueError as e:
                raise GrepSearcherError(f"Invalid search mode: {mode}") from e

        if mode == TextSearchMode.FUZZY:
            return self._search_fuzzy(
                query=query,
                file_paths=file_paths,
                limit=limit,
                context_lines=context_lines,
                case_sensitive=case_sensitive,
                file_filter=file_filter,
                fuzzy_min_score=fuzzy_min_score,
            )

        if mode == TextSearchMode.REGEX:
            return self._search_regex(
                query=query,
                file_paths=file_paths,
                limit=limit,
                context_lines=context_lines,
                case_sensitive=case_sensitive,
                file_filter=file_filter,
                all_terms=all_terms,
            )

        return self._search_substring(
            query=query,
            file_paths=file_paths,
            limit=limit,
            context_lines=context_lines,
            case_sensitive=case_sensitive,
            file_filter=file_filter,
            all_terms=all_terms,
        )

    def _search_substring(
        self,
        query: str,
        file_paths: list[str],
        limit: int,
        context_lines: int,
        case_sensitive: bool,
        file_filter: str | None,
        all_terms: bool,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []

        if all_terms:
            terms = _split_terms(query)
            if not terms:
                return []
            search_terms = terms if case_sensitive else [t.lower() for t in terms]
        else:
            search_terms = [query if case_sensitive else query.lower()]

        for file_path in file_paths:
            if len(results) >= limit:
                break

            if file_filter and not fnmatch.fnmatch(file_path, file_filter):
                continue

            resolved_path = self._resolve_path(file_path)
            if not os.path.isfile(resolved_path):
                logger.debug(f"Skipping non-existent file: {file_path}")
                continue

            if self._is_binary_file(resolved_path):
                logger.debug(f"Skipping binary file: {file_path}")
                continue

            file_results = self._search_file_substring(
                file_path=file_path,
                resolved_path=resolved_path,
                search_terms=search_terms,
                original_query=query,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                remaining_limit=limit - len(results),
                all_terms=all_terms,
            )
            results.extend(file_results)

        return results[:limit]

    def _search_regex(
        self,
        query: str,
        file_paths: list[str],
        limit: int,
        context_lines: int,
        case_sensitive: bool,
        file_filter: str | None,
        all_terms: bool,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []

        pattern_parts = _split_terms(query) if all_terms else [query]
        if not pattern_parts:
            return []

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            patterns = [re.compile(p, flags) for p in pattern_parts]
        except re.error as e:
            raise GrepSearcherError(f"Invalid regex pattern: {e}") from e

        for file_path in file_paths:
            if len(results) >= limit:
                break

            if file_filter and not fnmatch.fnmatch(file_path, file_filter):
                continue

            resolved_path = self._resolve_path(file_path)
            if not os.path.isfile(resolved_path):
                logger.debug(f"Skipping non-existent file: {file_path}")
                continue

            if self._is_binary_file(resolved_path):
                logger.debug(f"Skipping binary file: {file_path}")
                continue

            file_results = self._search_file_regex(
                file_path=file_path,
                resolved_path=resolved_path,
                patterns=patterns,
                original_query=query,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                remaining_limit=limit - len(results),
            )
            results.extend(file_results)

        return results[:limit]

    def _search_fuzzy(
        self,
        query: str,
        file_paths: list[str],
        limit: int,
        context_lines: int,
        case_sensitive: bool,
        file_filter: str | None,
        fuzzy_min_score: float,
    ) -> list[SearchResult]:
        terms = _split_terms(query)
        if not terms:
            return []

        if not case_sensitive:
            terms = [t.lower() for t in terms]

        heap: list[tuple[float, int, SearchResult]] = []
        counter = count()
        max_line_length = 2000

        for file_path in file_paths:
            if file_filter and not fnmatch.fnmatch(file_path, file_filter):
                continue

            resolved_path = self._resolve_path(file_path)
            if not os.path.isfile(resolved_path):
                continue
            if self._is_binary_file(resolved_path):
                continue

            try:
                with open(resolved_path, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except OSError:
                continue
            except UnicodeDecodeError:
                continue

            total_lines = len(lines)

            for line_num, line in enumerate(lines, start=1):
                candidate = line.rstrip("\n")
                if not candidate:
                    continue
                if len(candidate) > max_line_length:
                    continue

                candidate_match = candidate if case_sensitive else candidate.lower()

                per_term_scores: list[float] = []
                ok = True
                for term in terms:
                    s = _fuzzy_score(term, candidate_match, candidate)
                    if s < fuzzy_min_score:
                        ok = False
                        break
                    per_term_scores.append(s)

                if not ok:
                    continue

                score = 0.0
                total_weight = 0
                for term, s in zip(terms, per_term_scores, strict=False):
                    w = max(1, len(term))
                    score += s * w
                    total_weight += w
                score = score / max(1, total_weight)

                if heap and len(heap) >= limit and score <= heap[0][0]:
                    continue

                start_line = max(1, line_num - context_lines)
                end_line = min(total_lines, line_num + context_lines)
                content = "".join(lines[start_line - 1 : end_line])

                result = SearchResult(
                    chunk_id=f"fuzzy:{file_path}:{start_line}:{line_num}",
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    score=score,
                    metadata={
                        "source": "fuzzy",
                        "match_line": line_num,
                        "query": query,
                        "terms": terms,
                        "min_score": fuzzy_min_score,
                    },
                )

                item = (score, next(counter), result)
                if len(heap) < limit:
                    heappush(heap, item)
                else:
                    heappushpop(heap, item)

        heap.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [r for _, _, r in heap]

    def _search_file_substring(
        self,
        file_path: str,
        resolved_path: str,
        search_terms: list[str],
        original_query: str,
        case_sensitive: bool,
        context_lines: int,
        remaining_limit: int,
        all_terms: bool,
    ) -> list[SearchResult]:
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

            compare_line = line if case_sensitive else line.lower()

            if all_terms:
                if not all(term in compare_line for term in search_terms):
                    continue
            else:
                if search_terms[0] not in compare_line:
                    continue

            start_line = max(1, line_num - context_lines)
            end_line = min(total_lines, line_num + context_lines)
            content = "".join(lines[start_line - 1 : end_line])

            results.append(
                SearchResult(
                    chunk_id=f"grep:{file_path}:{start_line}",
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    score=1.0,
                    metadata={
                        "source": "grep",
                        "text_mode": "substring",
                        "match_line": line_num,
                        "query": original_query,
                        "all_terms": all_terms,
                    },
                )
            )

        return results

    def _search_file_regex(
        self,
        file_path: str,
        resolved_path: str,
        patterns: list[re.Pattern[str]],
        original_query: str,
        case_sensitive: bool,
        context_lines: int,
        remaining_limit: int,
    ) -> list[SearchResult]:
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

            if not all(p.search(line) for p in patterns):
                continue

            start_line = max(1, line_num - context_lines)
            end_line = min(total_lines, line_num + context_lines)
            content = "".join(lines[start_line - 1 : end_line])

            results.append(
                SearchResult(
                    chunk_id=f"grep_regex:{file_path}:{start_line}:{line_num}",
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    score=1.0,
                    metadata={
                        "source": "grep",
                        "text_mode": "regex",
                        "match_line": line_num,
                        "query": original_query,
                        "case_sensitive": case_sensitive,
                    },
                )
            )

        return results

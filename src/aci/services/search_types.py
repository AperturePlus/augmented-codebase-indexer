"""
Search types and interfaces for Project ACI.

Contains enums and abstract interfaces used by search components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from aci.infrastructure.vector_store import SearchResult


class SearchMode(str, Enum):
    """Search mode for controlling which search methods are used."""

    HYBRID = "hybrid"
    VECTOR = "vector"
    GREP = "grep"
    FUZZY = "fuzzy"
    SUMMARY = "summary"


@dataclass(frozen=True)
class TextSearchOptions:
    """
    Options for file-based text search (grep/regex/fuzzy).

    These options apply when SearchMode is GREP, FUZZY, or HYBRID (grep side).
    """

    context_lines: int = 3
    case_sensitive: bool = False
    regex: bool = False
    all_terms: bool = False
    fuzzy_min_score: float = 0.6


class RerankerInterface(ABC):
    """Interface for re-ranking search results."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
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

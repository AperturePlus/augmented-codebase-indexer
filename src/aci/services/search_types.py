"""
Search types and interfaces for Project ACI.

Contains enums and abstract interfaces used by search components.
"""

from abc import ABC, abstractmethod
from enum import Enum

from aci.infrastructure.vector_store import SearchResult


class SearchMode(str, Enum):
    """Search mode for controlling which search methods are used."""

    HYBRID = "hybrid"
    VECTOR = "vector"
    GREP = "grep"
    SUMMARY = "summary"


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

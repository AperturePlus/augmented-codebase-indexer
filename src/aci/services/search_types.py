"""
Search types and interfaces for Project ACI.

Contains enums and abstract interfaces used by search components.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List

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

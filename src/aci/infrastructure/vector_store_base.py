"""
Vector Store base types and interfaces.

Contains abstract interface, data classes, and exceptions for vector stores.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


def is_glob_pattern(path: str) -> bool:
    """
    Check if a path contains glob wildcard characters.
    
    Args:
        path: File path to check
        
    Returns:
        True if path contains *, ?, or [ characters (glob wildcards)
    """
    return any(c in path for c in "*?[")


class VectorStoreError(Exception):
    """Base exception for vector store errors."""

    pass


@dataclass
class SearchResult:
    """Search result from vector store."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float
    metadata: Dict


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    async def upsert(self, chunk_id: str, vector: List[float], payload: dict) -> None:
        """Insert or update a vector with its payload."""
        pass

    @abstractmethod
    async def delete_by_file(self, file_path: str) -> int:
        """Delete all vectors for a file, return count deleted."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        file_filter: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            file_filter: Optional glob pattern for file paths
            collection_name: Optional collection to search. If provided, searches
                that collection without modifying instance state. If None, uses
                the instance's default collection.

        Returns:
            List of SearchResult sorted by score descending
        """
        pass

    @abstractmethod
    async def get_stats(self, collection_name: Optional[str] = None) -> dict:
        """
        Get storage statistics.

        Args:
            collection_name: Optional collection to get stats for. If provided,
                returns stats for that collection without modifying instance state.
                If None, uses the instance's default collection.

        Returns:
            Dictionary with storage statistics
        """
        pass

    @abstractmethod
    async def get_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """Get a specific chunk by ID."""
        pass

    @abstractmethod
    async def get_all_file_paths(self, collection_name: Optional[str] = None) -> List[str]:
        """
        Get all unique file paths in the store.

        Args:
            collection_name: Optional collection to query. If provided, returns
                file paths from that collection without modifying instance state.
                If None, uses the instance's default collection.

        Returns:
            List of unique file paths
        """
        pass

    async def reset(self) -> None:
        """Reset/clear the vector store (optional)."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection entirely.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if the collection was deleted, False if it did not exist
        """
        pass

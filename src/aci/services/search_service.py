"""
Search Service for Project ACI.

Provides semantic search functionality over indexed codebases.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from aci.infrastructure.embedding_client import EmbeddingClientInterface
from aci.infrastructure.vector_store import SearchResult, VectorStoreInterface

logger = logging.getLogger(__name__)


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
    and optionally re-ranks results.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClientInterface,
        vector_store: VectorStoreInterface,
        reranker: Optional[RerankerInterface] = None,
        default_limit: int = 10,
        recall_multiplier: int = 5,
    ):
        """
        Initialize the search service.

        Args:
            embedding_client: Client for generating query embeddings
            vector_store: Store for vector search
            reranker: Optional re-ranker for result refinement
            default_limit: Default number of results to return
            recall_multiplier: Multiplier for initial recall when re-ranking
        """
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._reranker = reranker
        self._default_limit = default_limit
        self._recall_multiplier = recall_multiplier

    async def search(
        self,
        query: str,
        limit: Optional[int] = None,
        file_filter: Optional[str] = None,
        use_rerank: bool = True,
    ) -> List[SearchResult]:
        """
        Perform semantic search.

        Args:
            query: Natural language search query
            limit: Maximum results to return (default: default_limit)
            file_filter: Optional glob pattern for file paths
            use_rerank: Whether to use re-ranker if available

        Returns:
            List of SearchResult sorted by relevance
        """
        limit = limit or self._default_limit

        # Generate query embedding
        embeddings = await self._embedding_client.embed_batch([query])
        query_vector = embeddings[0]

        # Determine recall limit
        if use_rerank and self._reranker:
            recall_limit = limit * self._recall_multiplier
        else:
            recall_limit = limit

        # Search vector store
        candidates = await self._vector_store.search(
            query_vector=query_vector,
            limit=recall_limit,
            file_filter=file_filter,
        )

        # Re-rank if enabled and reranker available
        if use_rerank and self._reranker and candidates:
            reranked = self._reranker.rerank(query, candidates, limit)
            if inspect.iscoroutine(reranked):
                results = await reranked
            else:
                results = reranked
        else:
            results = candidates[:limit]

        return results

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

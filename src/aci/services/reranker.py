"""
Reranker for Project ACI.

Provides cross-encoder based re-ranking for search results.
"""

import logging
from typing import List, Optional

from aci.infrastructure.vector_store import SearchResult
from aci.services.search_service import RerankerInterface

logger = logging.getLogger(__name__)


class CrossEncoderReranker(RerankerInterface):
    """
    Cross-encoder based re-ranker using sentence-transformers.
    
    Uses a cross-encoder model to compute query-document relevance scores
    for more accurate ranking than bi-encoder similarity.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self._model_name = model_name
        self._device = device
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name, device=self._device)
                logger.info(f"Loaded cross-encoder model: {self._model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install sentence-transformers"
                )

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Re-rank candidates using cross-encoder scores.
        
        Args:
            query: Original search query
            candidates: Candidate results from vector search
            top_k: Number of results to return
            
        Returns:
            Re-ranked list of top_k results
        """
        if not candidates:
            return []
        
        self._load_model()
        
        # Create query-document pairs
        pairs = [(query, c.content) for c in candidates]
        
        # Get cross-encoder scores
        scores = self._model.predict(pairs)
        
        # Combine with candidates and sort
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k with updated scores
        results = []
        for candidate, score in scored_results[:top_k]:
            # Create new SearchResult with cross-encoder score
            results.append(SearchResult(
                chunk_id=candidate.chunk_id,
                file_path=candidate.file_path,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                content=candidate.content,
                score=float(score),
                metadata=candidate.metadata,
            ))
        
        return results


class SimpleReranker(RerankerInterface):
    """
    Simple reranker for testing - just returns top_k candidates.
    
    Useful for testing without loading heavy ML models.
    """

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Return top_k candidates without re-ranking."""
        return candidates[:top_k]

"""
Reciprocal Rank Fusion (RRF) fuser.

Merges ranked result lists from multiple analysis backends into a
single unified ranking.  RRF combines ranks without requiring score
normalisation across heterogeneous backends.

Reference: Cormack, Clarke & Butt, "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods", SIGIR 2009.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RRFFuser:
    """Merge ranked lists using Reciprocal Rank Fusion."""

    def fuse(
        self,
        ranked_lists: list[list[str]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Merge *ranked_lists* using the RRF formula.

        Each input list is an ordered sequence of identifiers (symbol IDs
        or chunk IDs) ranked by relevance (best first).

        The RRF score for an item is::

            score(item) = Σ  1 / (k + rank_i)

        where *rank_i* is the 1-based rank of the item in list *i* and
        the sum runs over all lists that contain the item.

        When only a single list is provided the ranking is passed through
        unchanged (Req 5.9).

        Args:
            ranked_lists: Ordered lists of identifiers from each backend.
            k: Smoothing constant (default 60).

        Returns:
            ``(id, rrf_score)`` pairs sorted by descending score.
        """
        if not ranked_lists:
            return []

        # Single-list passthrough
        non_empty = [rl for rl in ranked_lists if rl]
        if len(non_empty) == 0:
            return []
        if len(non_empty) == 1:
            return [(item, 1.0 / (k + rank)) for rank, item in enumerate(non_empty[0], start=1)]

        # Multi-list fusion
        scores: dict[str, float] = {}
        for ranked_list in non_empty:
            for rank, item in enumerate(ranked_list, start=1):
                scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)

        # Sort by descending score, then by id for determinism
        fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return fused

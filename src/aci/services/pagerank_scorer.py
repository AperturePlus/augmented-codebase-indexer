"""
PageRank scorer for code graphs.

Runs power iteration over adjacency data read from a
:class:`GraphStoreInterface` and writes the resulting scores back
to the store.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from aci.core.graph_store import GraphStoreInterface

logger = logging.getLogger(__name__)


class PageRankScorer:
    """Compute PageRank scores over a code graph via power iteration."""

    def __init__(
        self,
        graph_store: GraphStoreInterface,
        damping: float = 0.85,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> None:
        self._store = graph_store
        self._damping = damping
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    def compute(self, graph_type: str = "call") -> dict[str, float]:
        """Compute PageRank for all nodes in the given graph type.

        Reads all edges of *graph_type* from the graph store, builds an
        in-memory adjacency structure, runs power iteration until
        convergence or *max_iterations*, and stores the scores back.

        Returns the computed scores dict for immediate use.
        """
        edges = self._store.get_all_edges(graph_type=graph_type)

        if not edges:
            logger.debug("No edges of type %r found; skipping PageRank.", graph_type)
            return {}

        # Collect all node IDs referenced by edges
        all_nodes: set[str] = set()
        # out_links[source] = list of targets
        out_links: dict[str, list[str]] = defaultdict(list)
        # in_links[target] = list of sources
        in_links: dict[str, list[str]] = defaultdict(list)

        for e in edges:
            all_nodes.add(e.source_id)
            all_nodes.add(e.target_id)
            out_links[e.source_id].append(e.target_id)
            in_links[e.target_id].append(e.source_id)

        n = len(all_nodes)
        if n == 0:
            return {}

        # Initialize uniform scores
        initial = 1.0 / n
        scores: dict[str, float] = dict.fromkeys(all_nodes, initial)
        damping = self._damping
        base = (1.0 - damping) / n

        for iteration in range(self._max_iterations):
            new_scores: dict[str, float] = {}
            # Accumulate dangling node mass (nodes with no outgoing edges)
            dangling_sum = sum(
                scores[node] for node in all_nodes if not out_links.get(node)
            )

            for node in all_nodes:
                rank = base + damping * (dangling_sum / n)
                for src in in_links.get(node, []):
                    out_degree = len(out_links[src])
                    rank += damping * scores[src] / out_degree
                new_scores[node] = rank

            # Check convergence
            diff = sum(abs(new_scores[node] - scores[node]) for node in all_nodes)
            scores = new_scores

            if diff < self._tolerance:
                logger.debug(
                    "PageRank converged after %d iterations (diff=%.2e).",
                    iteration + 1,
                    diff,
                )
                break
        else:
            logger.debug(
                "PageRank reached max iterations (%d) with diff=%.2e.",
                self._max_iterations,
                diff,  # type: ignore[possibly-undefined]
            )

        # Store scores back to the graph store
        self._store.store_pagerank_scores(scores, graph_type)

        return scores

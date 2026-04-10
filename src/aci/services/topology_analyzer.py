"""
Topology analyzer for code graphs.

Performs graph-level computations over the GraphStoreInterface:
transitive caller/callee traversal, circular dependency detection,
and topological sorting of the dependency graph.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from aci.core.graph_store import GraphStoreInterface

logger = logging.getLogger(__name__)


class TopologyAnalyzer:
    """Stateless topology analyzer over a :class:`GraphStoreInterface`."""

    def __init__(self, graph_store: GraphStoreInterface) -> None:
        self._store = graph_store

    # ------------------------------------------------------------------
    # Transitive traversal
    # ------------------------------------------------------------------

    def transitive_callers(
        self, symbol_id: str, max_depth: int = 3
    ) -> list[str]:
        """Return FQNs of all transitive callers up to *max_depth*.

        Delegates to the GraphStore CTE-based neighbor query in the
        ``"callers"`` direction.
        """
        nodes = self._store.get_neighbors(
            symbol_id, direction="callers", depth=max_depth
        )
        return [n.symbol_id for n in nodes]

    def transitive_callees(
        self, symbol_id: str, max_depth: int = 3
    ) -> list[str]:
        """Return FQNs of all transitive callees up to *max_depth*.

        Delegates to the GraphStore CTE-based neighbor query in the
        ``"callees"`` direction.
        """
        nodes = self._store.get_neighbors(
            symbol_id, direction="callees", depth=max_depth
        )
        return [n.symbol_id for n in nodes]

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def detect_cycles(self) -> list[list[str]]:
        """Detect circular dependency cycles in the dependency graph.

        Returns each cycle as an ordered list of module paths forming
        the cycle.  Uses Johnson's algorithm variant via iterative DFS
        over the ``"import"`` edge subgraph.
        """
        edges = self._store.get_all_edges(graph_type="import")

        # Build adjacency list
        adj: dict[str, list[str]] = defaultdict(list)
        all_nodes: set[str] = set()
        for e in edges:
            adj[e.source_id].append(e.target_id)
            all_nodes.add(e.source_id)
            all_nodes.add(e.target_id)

        if not all_nodes:
            return []

        # Find all elementary cycles using DFS-based approach
        cycles: list[list[str]] = []
        visited_global: set[str] = set()

        for start in sorted(all_nodes):
            # Stack-based DFS from each unvisited start node
            # (path, current_node, neighbor_index)
            stack: list[tuple[list[str], str, int]] = [
                ([start], start, 0),
            ]
            path_set: set[str] = {start}

            while stack:
                path, node, idx = stack[-1]
                neighbors = adj.get(node, [])

                if idx < len(neighbors):
                    # Advance neighbor index
                    stack[-1] = (path, node, idx + 1)
                    neighbor = neighbors[idx]

                    if neighbor == start and len(path) > 1:
                        # Found a cycle back to start
                        cycle = list(path)
                        # Normalize: rotate so smallest element is first
                        min_idx = cycle.index(min(cycle))
                        normalized = cycle[min_idx:] + cycle[:min_idx]
                        if normalized not in cycles:
                            cycles.append(normalized)
                    elif neighbor not in path_set and neighbor not in visited_global:
                        path_set.add(neighbor)
                        stack.append((path + [neighbor], neighbor, 0))
                else:
                    # Backtrack
                    stack.pop()
                    path_set.discard(node)

            visited_global.add(start)

        return cycles

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """Topological sort of the dependency graph (acyclic subgraph).

        Uses Kahn's algorithm.  Nodes involved in cycles are excluded
        from the result (only the acyclic portion is sorted).

        Returns module paths in dependency order (a module appears after
        all modules it depends on).
        """
        edges = self._store.get_all_edges(graph_type="import")

        # Build adjacency and in-degree maps
        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = defaultdict(int)
        all_nodes: set[str] = set()

        for e in edges:
            adj[e.source_id].append(e.target_id)
            all_nodes.add(e.source_id)
            all_nodes.add(e.target_id)

        # Initialize in-degree for all nodes
        for node in all_nodes:
            if node not in in_degree:
                in_degree[node] = 0
        for e in edges:
            in_degree[e.target_id] += 1

        # Kahn's algorithm
        queue: list[str] = sorted(
            [n for n in all_nodes if in_degree[n] == 0]
        )
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in sorted(adj.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            # Keep queue sorted for deterministic output
            queue.sort()

        return result

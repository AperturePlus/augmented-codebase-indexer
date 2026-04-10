"""
Abstract interface for the code graph store.

Defines the contract for graph persistence backends (nodes, edges,
PageRank scores, and the cross-file symbol index).  The interface uses
synchronous methods because the reference implementation is SQLite,
which is inherently synchronous and runs in-process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aci.core.graph_models import GraphEdge, GraphNode, SymbolIndexEntry


class GraphStoreInterface(ABC):
    """Abstract interface for the code graph store."""

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_node(self, node: GraphNode) -> None: ...

    @abstractmethod
    def upsert_nodes_batch(self, nodes: list[GraphNode]) -> None: ...

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_edge(self, edge: GraphEdge) -> None: ...

    @abstractmethod
    def upsert_edges_batch(self, edges: list[GraphEdge]) -> None: ...

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    @abstractmethod
    def delete_by_file(self, file_path: str) -> None: ...

    # ------------------------------------------------------------------
    # Traversal / query
    # ------------------------------------------------------------------

    @abstractmethod
    def get_neighbors(
        self,
        symbol_id: str,
        direction: str,
        depth: int = 1,
        include_inferred: bool = True,
    ) -> list[GraphNode]: ...

    @abstractmethod
    def get_edges(
        self,
        symbol_id: str,
        direction: str,
        depth: int = 1,
        include_inferred: bool = True,
    ) -> list[GraphEdge]: ...

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    @abstractmethod
    def get_pagerank(self, symbol_id: str, graph_type: str = "call") -> float: ...

    @abstractmethod
    def store_pagerank_scores(
        self, scores: dict[str, float], graph_type: str
    ) -> None: ...

    # ------------------------------------------------------------------
    # Symbol / module queries
    # ------------------------------------------------------------------

    @abstractmethod
    def query_symbol(self, symbol_id: str) -> GraphNode | None: ...

    @abstractmethod
    def query_module(self, file_path: str) -> dict: ...

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @abstractmethod
    def export_json(self, path: str) -> None: ...

    @abstractmethod
    def import_json(self, path: str, mode: str) -> None: ...

    # ------------------------------------------------------------------
    # Bulk accessors
    # ------------------------------------------------------------------

    @abstractmethod
    def get_all_edges(self, graph_type: str | None = None) -> list[GraphEdge]: ...

    @abstractmethod
    def get_all_nodes(self) -> list[GraphNode]: ...

    # ------------------------------------------------------------------
    # Symbol index operations
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_symbol(self, entry: SymbolIndexEntry) -> None: ...

    @abstractmethod
    def upsert_symbols_batch(self, entries: list[SymbolIndexEntry]) -> None: ...

    @abstractmethod
    def lookup_symbol(self, fqn: str) -> SymbolIndexEntry | None: ...

    @abstractmethod
    def lookup_symbols_by_name(self, short_name: str) -> list[SymbolIndexEntry]: ...

    @abstractmethod
    def get_symbols_in_file(self, file_path: str) -> list[SymbolIndexEntry]: ...

    @abstractmethod
    def delete_symbols_by_file(self, file_path: str) -> None: ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def close(self) -> None: ...

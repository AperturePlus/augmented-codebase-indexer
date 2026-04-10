"""
Unit tests for SQLiteGraphStore.

Tests schema creation, node/edge CRUD, delete_by_file, neighbor traversal,
include_inferred filtering, PageRank, query_symbol, and query_module.
"""

from __future__ import annotations

import pytest

from aci.core.graph_models import GraphEdge, GraphNode, SymbolIndexEntry, SymbolLocation
from aci.infrastructure.graph_store.sqlite import SQLiteGraphStore


@pytest.fixture()
def store() -> SQLiteGraphStore:
    """In-memory SQLiteGraphStore for fast tests."""
    s = SQLiteGraphStore(":memory:")
    s.initialize()
    return s


# ------------------------------------------------------------------
# Schema idempotency
# ------------------------------------------------------------------


def test_schema_creation_is_idempotent(store: SQLiteGraphStore) -> None:
    # Calling initialize a second time should not raise
    store.initialize()
    store.initialize()
    assert store.get_all_nodes() == []


# ------------------------------------------------------------------
# Node CRUD
# ------------------------------------------------------------------


def test_upsert_and_query_node(store: SQLiteGraphStore) -> None:
    node = GraphNode("a.b.foo", "foo", "function", "a/b.py", 1, 10, "python")
    store.upsert_node(node)
    result = store.query_symbol("a.b.foo")
    assert result is not None
    assert result.symbol_id == "a.b.foo"
    assert result.symbol_name == "foo"


def test_upsert_nodes_batch(store: SQLiteGraphStore) -> None:
    nodes = [
        GraphNode(f"mod.fn{i}", f"fn{i}", "function", "mod.py", i, i + 5, "python")
        for i in range(5)
    ]
    store.upsert_nodes_batch(nodes)
    assert len(store.get_all_nodes()) == 5


# ------------------------------------------------------------------
# Edge CRUD
# ------------------------------------------------------------------


def test_upsert_and_get_edge(store: SQLiteGraphStore) -> None:
    _seed_chain(store, 2)
    edges = store.get_all_edges()
    assert len(edges) == 1
    assert edges[0].source_id == "n0"
    assert edges[0].target_id == "n1"


def test_upsert_edges_batch(store: SQLiteGraphStore) -> None:
    nodes = [GraphNode(f"n{i}", f"n{i}", "function", "f.py", i, i + 1, "py") for i in range(3)]
    store.upsert_nodes_batch(nodes)
    edges = [
        GraphEdge("n0", "n1", "call"),
        GraphEdge("n1", "n2", "call"),
    ]
    store.upsert_edges_batch(edges)
    assert len(store.get_all_edges()) == 2


# ------------------------------------------------------------------
# delete_by_file
# ------------------------------------------------------------------


def test_delete_by_file_removes_all_related_data(store: SQLiteGraphStore) -> None:
    store.upsert_node(GraphNode("a.foo", "foo", "function", "a.py", 1, 5, "python"))
    store.upsert_node(GraphNode("b.bar", "bar", "function", "b.py", 1, 5, "python"))
    store.upsert_edge(GraphEdge("a.foo", "b.bar", "call", file_path="a.py"))
    store.upsert_symbol(
        SymbolIndexEntry(
            fqn="a.foo",
            definition=SymbolLocation("a.py", 1, 5),
            references=[SymbolLocation("b.py", 10, 10)],
            graph_node_id="a.foo",
        )
    )

    store.delete_by_file("a.py")

    assert store.query_symbol("a.foo") is None  # node gone
    assert store.get_all_edges() == []  # edge gone (file_path=a.py)
    assert store.lookup_symbol("a.foo") is None  # symbol index gone
    # b.bar should still exist
    assert store.query_symbol("b.bar") is not None


# ------------------------------------------------------------------
# get_neighbors — depth 1, 2, 3 and direction
# ------------------------------------------------------------------


def _seed_chain(store: SQLiteGraphStore, length: int) -> None:
    """Create a linear call chain: n0 -> n1 -> n2 -> ... -> n{length-1}."""
    nodes = [
        GraphNode(f"n{i}", f"n{i}", "function", "f.py", i, i + 1, "py")
        for i in range(length)
    ]
    store.upsert_nodes_batch(nodes)
    edges = [GraphEdge(f"n{i}", f"n{i+1}", "call") for i in range(length - 1)]
    store.upsert_edges_batch(edges)


def test_get_neighbors_callees_depth_1(store: SQLiteGraphStore) -> None:
    _seed_chain(store, 4)  # n0 -> n1 -> n2 -> n3
    neighbors = store.get_neighbors("n0", "callees", depth=1)
    ids = {n.symbol_id for n in neighbors}
    assert ids == {"n1"}


def test_get_neighbors_callees_depth_2(store: SQLiteGraphStore) -> None:
    _seed_chain(store, 4)
    neighbors = store.get_neighbors("n0", "callees", depth=2)
    ids = {n.symbol_id for n in neighbors}
    assert ids == {"n1", "n2"}


def test_get_neighbors_callees_depth_3(store: SQLiteGraphStore) -> None:
    _seed_chain(store, 4)
    neighbors = store.get_neighbors("n0", "callees", depth=3)
    ids = {n.symbol_id for n in neighbors}
    assert ids == {"n1", "n2", "n3"}


def test_get_neighbors_callers_depth_1(store: SQLiteGraphStore) -> None:
    _seed_chain(store, 4)
    neighbors = store.get_neighbors("n3", "callers", depth=1)
    ids = {n.symbol_id for n in neighbors}
    assert ids == {"n2"}


def test_get_neighbors_callers_depth_2(store: SQLiteGraphStore) -> None:
    _seed_chain(store, 4)
    neighbors = store.get_neighbors("n3", "callers", depth=2)
    ids = {n.symbol_id for n in neighbors}
    assert ids == {"n1", "n2"}


def test_get_neighbors_callers_depth_3(store: SQLiteGraphStore) -> None:
    _seed_chain(store, 4)
    neighbors = store.get_neighbors("n3", "callers", depth=3)
    ids = {n.symbol_id for n in neighbors}
    assert ids == {"n0", "n1", "n2"}


# ------------------------------------------------------------------
# include_inferred filtering
# ------------------------------------------------------------------


def test_include_inferred_true_returns_inferred_edges(store: SQLiteGraphStore) -> None:
    store.upsert_nodes_batch([
        GraphNode("a", "a", "function", "f.py", 1, 2, "py"),
        GraphNode("b", "b", "function", "f.py", 3, 4, "py"),
    ])
    store.upsert_edge(GraphEdge("a", "b", "call", inferred=True, confidence=0.7))

    neighbors = store.get_neighbors("a", "callees", depth=1, include_inferred=True)
    assert len(neighbors) == 1
    assert neighbors[0].symbol_id == "b"


def test_include_inferred_false_excludes_inferred_edges(store: SQLiteGraphStore) -> None:
    store.upsert_nodes_batch([
        GraphNode("a", "a", "function", "f.py", 1, 2, "py"),
        GraphNode("b", "b", "function", "f.py", 3, 4, "py"),
    ])
    store.upsert_edge(GraphEdge("a", "b", "call", inferred=True, confidence=0.7))

    neighbors = store.get_neighbors("a", "callees", depth=1, include_inferred=False)
    assert len(neighbors) == 0


def test_include_inferred_false_keeps_non_inferred(store: SQLiteGraphStore) -> None:
    store.upsert_nodes_batch([
        GraphNode("a", "a", "function", "f.py", 1, 2, "py"),
        GraphNode("b", "b", "function", "f.py", 3, 4, "py"),
        GraphNode("c", "c", "function", "f.py", 5, 6, "py"),
    ])
    store.upsert_edge(GraphEdge("a", "b", "call", inferred=False))
    store.upsert_edge(GraphEdge("a", "c", "call", inferred=True, confidence=0.6))

    neighbors = store.get_neighbors("a", "callees", depth=1, include_inferred=False)
    ids = {n.symbol_id for n in neighbors}
    assert ids == {"b"}


# ------------------------------------------------------------------
# PageRank
# ------------------------------------------------------------------


def test_get_pagerank_returns_zero_for_unknown(store: SQLiteGraphStore) -> None:
    assert store.get_pagerank("nonexistent") == 0.0


def test_store_and_get_pagerank(store: SQLiteGraphStore) -> None:
    store.store_pagerank_scores({"a": 0.5, "b": 0.3}, "call")
    assert store.get_pagerank("a", "call") == pytest.approx(0.5)
    assert store.get_pagerank("b", "call") == pytest.approx(0.3)
    assert store.get_pagerank("a", "dependency") == 0.0  # different graph_type


# ------------------------------------------------------------------
# query_symbol / query_module
# ------------------------------------------------------------------


def test_query_symbol_returns_none_for_missing(store: SQLiteGraphStore) -> None:
    assert store.query_symbol("does.not.exist") is None


def test_query_module_returns_empty_for_missing(store: SQLiteGraphStore) -> None:
    result = store.query_module("nonexistent.py")
    assert result["nodes"] == []
    assert result["edges"] == []


def test_query_module_returns_nodes_and_edges(store: SQLiteGraphStore) -> None:
    store.upsert_node(GraphNode("m.foo", "foo", "function", "m.py", 1, 5, "py"))
    store.upsert_node(GraphNode("m.bar", "bar", "function", "m.py", 6, 10, "py"))
    store.upsert_edge(GraphEdge("m.foo", "m.bar", "call", file_path="m.py"))

    result = store.query_module("m.py")
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1


# ------------------------------------------------------------------
# Symbol index operations
# ------------------------------------------------------------------


def test_symbol_round_trip(store: SQLiteGraphStore) -> None:
    entry = SymbolIndexEntry(
        fqn="pkg.mod.MyClass",
        definition=SymbolLocation("pkg/mod.py", 10, 50),
        references=[SymbolLocation("pkg/other.py", 5, 5)],
        graph_node_id="pkg.mod.MyClass",
        summary="A class",
    )
    store.upsert_symbol(entry)
    result = store.lookup_symbol("pkg.mod.MyClass")
    assert result is not None
    assert result.fqn == "pkg.mod.MyClass"
    assert result.definition.file_path == "pkg/mod.py"
    assert len(result.references) == 1


def test_lookup_symbols_by_name(store: SQLiteGraphStore) -> None:
    store.upsert_symbol(
        SymbolIndexEntry("a.b.Foo", SymbolLocation("a/b.py", 1, 10), graph_node_id="a.b.Foo")
    )
    store.upsert_symbol(
        SymbolIndexEntry("c.d.Foo", SymbolLocation("c/d.py", 1, 10), graph_node_id="c.d.Foo")
    )
    results = store.lookup_symbols_by_name("Foo")
    assert len(results) == 2


def test_get_symbols_in_file(store: SQLiteGraphStore) -> None:
    store.upsert_symbol(
        SymbolIndexEntry("a.foo", SymbolLocation("a.py", 1, 5), graph_node_id="a.foo")
    )
    store.upsert_symbol(
        SymbolIndexEntry("a.bar", SymbolLocation("a.py", 6, 10), graph_node_id="a.bar")
    )
    store.upsert_symbol(
        SymbolIndexEntry("b.baz", SymbolLocation("b.py", 1, 5), graph_node_id="b.baz")
    )
    results = store.get_symbols_in_file("a.py")
    assert len(results) == 2
    fqns = {r.fqn for r in results}
    assert fqns == {"a.foo", "a.bar"}

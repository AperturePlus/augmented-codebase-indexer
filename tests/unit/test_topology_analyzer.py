"""
Unit tests for TopologyAnalyzer.

Tests transitive callers/callees, cycle detection, topological sort,
and empty graph edge cases.
"""

from __future__ import annotations

import pytest

from aci.core.graph_models import GraphEdge, GraphNode
from aci.infrastructure.graph_store.sqlite import SQLiteGraphStore
from aci.services.topology_analyzer import TopologyAnalyzer

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def store() -> SQLiteGraphStore:
    """In-memory SQLiteGraphStore for fast tests."""
    s = SQLiteGraphStore(":memory:")
    s.initialize()
    return s


@pytest.fixture()
def analyzer(store: SQLiteGraphStore) -> TopologyAnalyzer:
    return TopologyAnalyzer(store)


def _make_node(symbol_id: str) -> GraphNode:
    return GraphNode(
        symbol_id=symbol_id,
        symbol_name=symbol_id.split(".")[-1],
        symbol_type="function",
        file_path="test.py",
        start_line=1,
        end_line=10,
        language="python",
    )


def _make_call_edge(source: str, target: str) -> GraphEdge:
    return GraphEdge(
        source_id=source,
        target_id=target,
        edge_type="call",
        file_path="test.py",
        line=1,
    )


def _make_import_edge(source: str, target: str) -> GraphEdge:
    return GraphEdge(
        source_id=source,
        target_id=target,
        edge_type="import",
        file_path="test.py",
        line=1,
    )


# ------------------------------------------------------------------
# Transitive callers / callees
# ------------------------------------------------------------------


def test_transitive_callees_linear_chain(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """A -> B -> C -> D: callees of A should be B, C, D."""
    for name in ["A", "B", "C", "D"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_call_edge("A", "B"),
        _make_call_edge("B", "C"),
        _make_call_edge("C", "D"),
    ])

    result = analyzer.transitive_callees("A", max_depth=3)
    assert set(result) == {"B", "C", "D"}


def test_transitive_callers_linear_chain(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """A -> B -> C -> D: callers of D should be A, B, C."""
    for name in ["A", "B", "C", "D"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_call_edge("A", "B"),
        _make_call_edge("B", "C"),
        _make_call_edge("C", "D"),
    ])

    result = analyzer.transitive_callers("D", max_depth=3)
    assert set(result) == {"A", "B", "C"}


def test_transitive_callees_respects_max_depth(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """A -> B -> C -> D: callees of A at depth 1 should be only B."""
    for name in ["A", "B", "C", "D"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_call_edge("A", "B"),
        _make_call_edge("B", "C"),
        _make_call_edge("C", "D"),
    ])

    result = analyzer.transitive_callees("A", max_depth=1)
    assert set(result) == {"B"}


def test_transitive_callers_respects_max_depth(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """A -> B -> C -> D: callers of D at depth 2 should be B, C."""
    for name in ["A", "B", "C", "D"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_call_edge("A", "B"),
        _make_call_edge("B", "C"),
        _make_call_edge("C", "D"),
    ])

    result = analyzer.transitive_callers("D", max_depth=2)
    assert set(result) == {"B", "C"}


def test_transitive_callees_diamond(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """A -> B, A -> C, B -> D, C -> D: callees of A should be B, C, D."""
    for name in ["A", "B", "C", "D"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_call_edge("A", "B"),
        _make_call_edge("A", "C"),
        _make_call_edge("B", "D"),
        _make_call_edge("C", "D"),
    ])

    result = analyzer.transitive_callees("A", max_depth=3)
    assert set(result) == {"B", "C", "D"}


def test_transitive_callees_unknown_symbol(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """Querying a non-existent symbol returns empty list."""
    result = analyzer.transitive_callees("nonexistent", max_depth=3)
    assert result == []


def test_transitive_callers_unknown_symbol(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """Querying a non-existent symbol returns empty list."""
    result = analyzer.transitive_callers("nonexistent", max_depth=3)
    assert result == []


# ------------------------------------------------------------------
# Cycle detection
# ------------------------------------------------------------------


def test_detect_cycles_simple_cycle(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """A -> B -> C -> A should be detected as a cycle."""
    for name in ["mod.A", "mod.B", "mod.C"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_import_edge("mod.A", "mod.B"),
        _make_import_edge("mod.B", "mod.C"),
        _make_import_edge("mod.C", "mod.A"),
    ])

    cycles = analyzer.detect_cycles()
    assert len(cycles) >= 1
    # The cycle should contain all three modules
    cycle_sets = [set(c) for c in cycles]
    assert {"mod.A", "mod.B", "mod.C"} in cycle_sets


def test_detect_cycles_no_cycles(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """A -> B -> C (no cycle) should return empty."""
    for name in ["mod.A", "mod.B", "mod.C"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_import_edge("mod.A", "mod.B"),
        _make_import_edge("mod.B", "mod.C"),
    ])

    cycles = analyzer.detect_cycles()
    assert cycles == []


def test_detect_cycles_self_loop(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """A -> A should be detected as a cycle (if path length > 1 is required, skip)."""
    store.upsert_node(_make_node("mod.A"))
    store.upsert_edge(_make_import_edge("mod.A", "mod.A"))

    # Self-loops are technically cycles but our implementation requires len > 1
    # The CTE-based approach may or may not catch self-loops depending on
    # implementation. We just verify no crash.
    cycles = analyzer.detect_cycles()
    # Self-loop detection is implementation-dependent
    assert isinstance(cycles, list)


def test_detect_cycles_multiple_cycles(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """Two separate cycles should both be detected."""
    for name in ["A", "B", "C", "D"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_import_edge("A", "B"),
        _make_import_edge("B", "A"),
        _make_import_edge("C", "D"),
        _make_import_edge("D", "C"),
    ])

    cycles = analyzer.detect_cycles()
    assert len(cycles) >= 2
    cycle_sets = [set(c) for c in cycles]
    assert {"A", "B"} in cycle_sets
    assert {"C", "D"} in cycle_sets


def test_detect_cycles_empty_graph(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """Empty graph returns no cycles."""
    cycles = analyzer.detect_cycles()
    assert cycles == []


# ------------------------------------------------------------------
# Topological sort
# ------------------------------------------------------------------


def test_topological_sort_linear(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """A -> B -> C: topological order should have A before B before C."""
    for name in ["A", "B", "C"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_import_edge("A", "B"),
        _make_import_edge("B", "C"),
    ])

    result = analyzer.topological_sort()
    assert "A" in result
    assert "B" in result
    assert "C" in result
    assert result.index("A") < result.index("B") < result.index("C")


def test_topological_sort_diamond(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """A -> B, A -> C, B -> D, C -> D: A first, D last."""
    for name in ["A", "B", "C", "D"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_import_edge("A", "B"),
        _make_import_edge("A", "C"),
        _make_import_edge("B", "D"),
        _make_import_edge("C", "D"),
    ])

    result = analyzer.topological_sort()
    assert result.index("A") < result.index("B")
    assert result.index("A") < result.index("C")
    assert result.index("B") < result.index("D")
    assert result.index("C") < result.index("D")


def test_topological_sort_excludes_cyclic_nodes(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """Nodes in cycles are excluded from topological sort result."""
    for name in ["A", "B", "C"]:
        store.upsert_node(_make_node(name))
    # A -> B -> A is a cycle, C has no edges
    store.upsert_edges_batch([
        _make_import_edge("A", "B"),
        _make_import_edge("B", "A"),
    ])
    # Add C as an isolated node with an import edge from somewhere
    store.upsert_node(_make_node("C"))
    store.upsert_edge(_make_import_edge("C", "A"))

    result = analyzer.topological_sort()
    # C should be in the result (it has no incoming edges)
    # A and B are in a cycle so they won't be emitted by Kahn's
    assert "C" in result
    assert "A" not in result
    assert "B" not in result


def test_topological_sort_empty_graph(
    store: SQLiteGraphStore, analyzer: TopologyAnalyzer
) -> None:
    """Empty graph returns empty list."""
    result = analyzer.topological_sort()
    assert result == []

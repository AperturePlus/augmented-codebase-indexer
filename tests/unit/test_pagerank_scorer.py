"""
Unit tests for PageRankScorer.

Tests convergence on known graphs, score storage, unknown symbol
handling, and performance on moderate-sized graphs.
"""

from __future__ import annotations

import time

import pytest

from aci.core.graph_models import GraphEdge, GraphNode
from aci.infrastructure.graph_store.sqlite import SQLiteGraphStore
from aci.services.pagerank_scorer import PageRankScorer

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
def scorer(store: SQLiteGraphStore) -> PageRankScorer:
    return PageRankScorer(store)


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


# ------------------------------------------------------------------
# Convergence on known graphs
# ------------------------------------------------------------------


def test_pagerank_simple_chain(
    store: SQLiteGraphStore, scorer: PageRankScorer
) -> None:
    """A -> B -> C: C should have the highest score (most pointed-to)."""
    for name in ["A", "B", "C"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_call_edge("A", "B"),
        _make_call_edge("B", "C"),
    ])

    scores = scorer.compute(graph_type="call")
    assert len(scores) == 3
    # All scores should be positive and sum to ~1.0
    total = sum(scores.values())
    assert abs(total - 1.0) < 0.01
    # C is the terminal node pointed to by B, should have high score
    assert scores["C"] > scores["A"]


def test_pagerank_star_topology(
    store: SQLiteGraphStore, scorer: PageRankScorer
) -> None:
    """A, B, C all call D: D should have the highest score."""
    for name in ["A", "B", "C", "D"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_call_edge("A", "D"),
        _make_call_edge("B", "D"),
        _make_call_edge("C", "D"),
    ])

    scores = scorer.compute(graph_type="call")
    assert scores["D"] > scores["A"]
    assert scores["D"] > scores["B"]
    assert scores["D"] > scores["C"]


def test_pagerank_cycle(
    store: SQLiteGraphStore, scorer: PageRankScorer
) -> None:
    """A -> B -> C -> A: all nodes should have roughly equal scores."""
    for name in ["A", "B", "C"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_call_edge("A", "B"),
        _make_call_edge("B", "C"),
        _make_call_edge("C", "A"),
    ])

    scores = scorer.compute(graph_type="call")
    # In a symmetric cycle, all scores should be approximately equal
    values = list(scores.values())
    assert max(values) - min(values) < 0.01


def test_pagerank_scores_sum_to_one(
    store: SQLiteGraphStore, scorer: PageRankScorer
) -> None:
    """PageRank scores should sum to approximately 1.0."""
    for name in ["A", "B", "C", "D", "E"]:
        store.upsert_node(_make_node(name))
    store.upsert_edges_batch([
        _make_call_edge("A", "B"),
        _make_call_edge("A", "C"),
        _make_call_edge("B", "D"),
        _make_call_edge("C", "D"),
        _make_call_edge("D", "E"),
    ])

    scores = scorer.compute(graph_type="call")
    total = sum(scores.values())
    assert abs(total - 1.0) < 0.01


# ------------------------------------------------------------------
# Score storage
# ------------------------------------------------------------------


def test_scores_stored_in_graph_store(
    store: SQLiteGraphStore, scorer: PageRankScorer
) -> None:
    """After compute(), scores should be retrievable via get_pagerank()."""
    for name in ["A", "B"]:
        store.upsert_node(_make_node(name))
    store.upsert_edge(_make_call_edge("A", "B"))

    scores = scorer.compute(graph_type="call")

    for symbol_id, expected_score in scores.items():
        stored = store.get_pagerank(symbol_id, graph_type="call")
        assert abs(stored - expected_score) < 1e-9


def test_get_pagerank_returns_zero_for_unknown(
    store: SQLiteGraphStore,
) -> None:
    """get_pagerank() returns 0.0 for symbols not in the graph."""
    assert store.get_pagerank("nonexistent", graph_type="call") == 0.0


# ------------------------------------------------------------------
# Empty graph
# ------------------------------------------------------------------


def test_pagerank_empty_graph(
    store: SQLiteGraphStore, scorer: PageRankScorer
) -> None:
    """Empty graph returns empty scores dict."""
    scores = scorer.compute(graph_type="call")
    assert scores == {}


# ------------------------------------------------------------------
# Configurable parameters
# ------------------------------------------------------------------


def test_custom_damping(store: SQLiteGraphStore) -> None:
    """Custom damping factor should affect scores."""
    for name in ["A", "B"]:
        store.upsert_node(_make_node(name))
    store.upsert_edge(_make_call_edge("A", "B"))

    scorer_low = PageRankScorer(store, damping=0.5)
    scores_low = scorer_low.compute(graph_type="call")

    # Reset scores
    store.store_pagerank_scores({}, "call")

    scorer_high = PageRankScorer(store, damping=0.99)
    scores_high = scorer_high.compute(graph_type="call")

    # With higher damping, the difference between A and B should be larger
    diff_low = abs(scores_low["B"] - scores_low["A"])
    diff_high = abs(scores_high["B"] - scores_high["A"])
    assert diff_high > diff_low


# ------------------------------------------------------------------
# Performance
# ------------------------------------------------------------------


def test_pagerank_moderate_graph_within_budget(
    store: SQLiteGraphStore,
) -> None:
    """PageRank on a 1000-node graph should complete within 5 seconds."""
    nodes = [f"sym_{i}" for i in range(1000)]
    store.upsert_nodes_batch([_make_node(n) for n in nodes])

    # Create a chain + some cross-links for a realistic graph
    edges = []
    for i in range(999):
        edges.append(_make_call_edge(nodes[i], nodes[i + 1]))
    # Add some cross-links
    for i in range(0, 1000, 10):
        edges.append(_make_call_edge(nodes[i], nodes[(i + 50) % 1000]))
    store.upsert_edges_batch(edges)

    scorer = PageRankScorer(store)
    start = time.monotonic()
    scores = scorer.compute(graph_type="call")
    elapsed = time.monotonic() - start

    assert len(scores) == 1000
    assert elapsed < 5.0, f"PageRank took {elapsed:.2f}s, exceeds 5s budget"

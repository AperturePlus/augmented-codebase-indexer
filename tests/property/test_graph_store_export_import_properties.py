"""
Property-based tests for graph export/import round-trip.

**Feature: semantic-code-intelligence, Property: Graph Export/Import Round-Trip**
**Validates: Requirements 13.1, 13.2, 13.3, 13.4**

For any valid graph state (nodes + edges + pagerank scores), exporting
then importing in "replace" mode produces an equivalent graph.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.core.graph_models import GraphEdge, GraphNode
from aci.infrastructure.graph_store.sqlite import SQLiteGraphStore

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_identifier = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="._"),
    min_size=1,
    max_size=60,
).filter(lambda s: s.strip() != "" and not s.startswith(".") and not s.endswith("."))

_symbol_type = st.sampled_from(["function", "class", "method", "module", "variable"])
_language = st.sampled_from(["python", "javascript", "go", "java", "cpp", ""])
_line = st.integers(min_value=1, max_value=50000)


@st.composite
def graph_node_strategy(draw: st.DrawFn) -> GraphNode:
    sid = draw(_identifier)
    short = sid.rsplit(".", 1)[-1] if "." in sid else sid
    start = draw(_line)
    end = draw(st.integers(min_value=start, max_value=start + 500))
    return GraphNode(
        symbol_id=sid,
        symbol_name=short,
        symbol_type=draw(_symbol_type),
        file_path=draw(st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/._-"),
            min_size=1, max_size=80,
        ).filter(lambda s: s.strip() != "")),
        start_line=start,
        end_line=end,
        language=draw(_language),
    )


@st.composite
def graph_state_strategy(draw: st.DrawFn) -> tuple[list[GraphNode], list[GraphEdge], dict[str, float]]:
    """Generate a consistent graph state: nodes, edges between those nodes, and pagerank scores."""
    nodes = draw(st.lists(graph_node_strategy(), min_size=0, max_size=8, unique_by=lambda n: n.symbol_id))
    edges: list[GraphEdge] = []
    if len(nodes) >= 2:
        node_ids = [n.symbol_id for n in nodes]
        edge_type = st.sampled_from(["call", "import"])
        raw_edges = draw(
            st.lists(
                st.tuples(
                    st.sampled_from(node_ids),
                    st.sampled_from(node_ids),
                    edge_type,
                ),
                min_size=0,
                max_size=10,
            )
        )
        seen: set[tuple[str, str, str]] = set()
        for src, tgt, et in raw_edges:
            if src != tgt and (src, tgt, et) not in seen:
                seen.add((src, tgt, et))
                edges.append(GraphEdge(source_id=src, target_id=tgt, edge_type=et))

    # PageRank scores for a subset of nodes
    scores: dict[str, float] = {}
    if nodes:
        scored_nodes = draw(
            st.lists(
                st.sampled_from([n.symbol_id for n in nodes]),
                min_size=0,
                max_size=len(nodes),
                unique=True,
            )
        )
        for sid in scored_nodes:
            scores[sid] = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))

    return nodes, edges, scores


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@given(state=graph_state_strategy())
@settings(max_examples=100, deadline=None)
def test_export_import_replace_round_trip(
    state: tuple[list[GraphNode], list[GraphEdge], dict[str, float]],
) -> None:
    """
    **Feature: semantic-code-intelligence, Property: Graph Export/Import Round-Trip**
    **Validates: Requirements 13.1, 13.2, 13.3, 13.4**

    For any valid graph state, export → import (replace) produces an
    equivalent graph: same nodes, same edges, same pagerank scores.
    The exported JSON must contain a schema_version field.
    """
    nodes, edges, scores = state

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = str(Path(tmpdir) / "graph.json")

        # --- Populate source store ---
        src = SQLiteGraphStore(":memory:")
        src.initialize()
        src.upsert_nodes_batch(nodes)
        src.upsert_edges_batch(edges)
        if scores:
            src.store_pagerank_scores(scores, "call")

        # --- Export ---
        src.export_json(export_path)

        # Verify schema_version is present
        with open(export_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "schema_version" in data, "Exported JSON must contain schema_version"
        assert data["schema_version"] == "1.0"
        assert "exported_at" in data

        # --- Import into a fresh store ---
        dst = SQLiteGraphStore(":memory:")
        dst.initialize()
        dst.import_json(export_path, mode="replace")

        # --- Verify equivalence ---
        dst_nodes = dst.get_all_nodes()
        dst_edges = dst.get_all_edges()

        # Nodes: same set by symbol_id
        src_node_ids = {n.symbol_id for n in nodes}
        dst_node_ids = {n.symbol_id for n in dst_nodes}
        assert src_node_ids == dst_node_ids, (
            f"Node sets differ: {src_node_ids - dst_node_ids} missing, "
            f"{dst_node_ids - src_node_ids} extra"
        )

        # Verify node attributes
        dst_node_map = {n.symbol_id: n for n in dst_nodes}
        for n in nodes:
            dn = dst_node_map[n.symbol_id]
            assert dn.symbol_name == n.symbol_name
            assert dn.symbol_type == n.symbol_type
            assert dn.file_path == n.file_path
            assert dn.start_line == n.start_line
            assert dn.end_line == n.end_line
            assert dn.language == n.language

        # Edges: same set by (source_id, target_id, edge_type)
        src_edge_keys = {(e.source_id, e.target_id, e.edge_type) for e in edges}
        dst_edge_keys = {(e.source_id, e.target_id, e.edge_type) for e in dst_edges}
        assert src_edge_keys == dst_edge_keys

        # PageRank scores
        for sid, expected_score in scores.items():
            actual = dst.get_pagerank(sid, "call")
            assert abs(actual - expected_score) < 1e-9, (
                f"PageRank mismatch for {sid}: expected {expected_score}, got {actual}"
            )

        src.close()
        dst.close()

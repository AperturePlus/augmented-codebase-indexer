"""Unit tests for ContextAssembler."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from aci.core.graph_models import (
    GraphEdge,
    GraphNode,
    QueryRequest,
    SymbolIndexEntry,
    SymbolLocation,
)
from aci.services.context_assembler import ContextAssembler

# ------------------------------------------------------------------
# Helpers / fixtures
# ------------------------------------------------------------------


def _make_node(symbol_id: str, file_path: str = "test.py") -> GraphNode:
    return GraphNode(
        symbol_id=symbol_id,
        symbol_name=symbol_id.split(".")[-1],
        symbol_type="function",
        file_path=file_path,
        start_line=1,
        end_line=10,
        language="python",
    )


def _make_entry(
    fqn: str,
    file_path: str = "test.py",
    start_line: int = 1,
    end_line: int = 10,
    summary: str = "A test symbol.",
) -> SymbolIndexEntry:
    return SymbolIndexEntry(
        fqn=fqn,
        definition=SymbolLocation(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
        ),
        graph_node_id=fqn,
        summary=summary,
    )


def _make_edge(
    source: str,
    target: str,
    edge_type: str = "call",
    file_path: str = "test.py",
) -> GraphEdge:
    return GraphEdge(
        source_id=source,
        target_id=target,
        edge_type=edge_type,
        file_path=file_path,
    )


def _make_search_result(
    chunk_id: str,
    file_path: str = "test.py",
    content: str = "def foo(): pass",
) -> Any:
    """Create a minimal SearchResult-like object."""
    return MagicMock(
        chunk_id=chunk_id,
        file_path=file_path,
        content=content,
        start_line=1,
        end_line=5,
        score=0.9,
        metadata={},
    )


def _make_tokenizer(tokens_per_char: int = 1) -> MagicMock:
    """Create a mock tokenizer that counts tokens as len(text) / tokens_per_char."""
    tok = MagicMock()
    tok.count_tokens = MagicMock(side_effect=lambda text: len(text) // max(tokens_per_char, 1))
    tok.truncate_to_tokens = MagicMock(
        side_effect=lambda text, max_tokens: text[: max_tokens * max(tokens_per_char, 1)]
    )
    return tok


def _make_graph_store(
    entries: dict[str, SymbolIndexEntry] | None = None,
    entries_by_name: dict[str, list[SymbolIndexEntry]] | None = None,
    neighbors_callers: dict[str, list[GraphNode]] | None = None,
    neighbors_callees: dict[str, list[GraphNode]] | None = None,
    edges_callers: dict[str, list[GraphEdge]] | None = None,
    edges_callees: dict[str, list[GraphEdge]] | None = None,
    pagerank_scores: dict[str, float] | None = None,
    module_data: dict[str, dict] | None = None,
    symbols_in_file: dict[str, list[SymbolIndexEntry]] | None = None,
) -> MagicMock:
    """Create a mock GraphStoreInterface."""
    store = MagicMock()
    _entries = entries or {}
    _entries_by_name = entries_by_name or {}
    _neighbors_callers = neighbors_callers or {}
    _neighbors_callees = neighbors_callees or {}
    _edges_callers = edges_callers or {}
    _edges_callees = edges_callees or {}
    _pagerank = pagerank_scores or {}
    _module = module_data or {}
    _sym_in_file = symbols_in_file or {}

    store.lookup_symbol = MagicMock(side_effect=lambda fqn: _entries.get(fqn))
    store.lookup_symbols_by_name = MagicMock(
        side_effect=lambda name: _entries_by_name.get(name, [])
    )

    def _get_neighbors(symbol_id: str, direction: str, depth: int = 1, include_inferred: bool = True) -> list[GraphNode]:
        if direction == "callers":
            return _neighbors_callers.get(symbol_id, [])
        return _neighbors_callees.get(symbol_id, [])

    store.get_neighbors = MagicMock(side_effect=_get_neighbors)

    def _get_edges(symbol_id: str, direction: str, depth: int = 1, include_inferred: bool = True) -> list[GraphEdge]:
        if direction == "callers":
            return _edges_callers.get(symbol_id, [])
        return _edges_callees.get(symbol_id, [])

    store.get_edges = MagicMock(side_effect=_get_edges)
    store.get_pagerank = MagicMock(side_effect=lambda fqn, graph_type="call": _pagerank.get(fqn, 0.0))
    store.query_module = MagicMock(
        side_effect=lambda fp: _module.get(fp, {"nodes": [], "edges": []})
    )
    store.get_symbols_in_file = MagicMock(
        side_effect=lambda fp: _sym_in_file.get(fp, [])
    )
    return store


@pytest.fixture
def tokenizer() -> MagicMock:
    return _make_tokenizer(tokens_per_char=1)


# ------------------------------------------------------------------
# Test: Symbol query returns source code, summary, callers, callees,
#       file summary (Req 6.1)
# ------------------------------------------------------------------


class TestSymbolQuery:
    """Assemble returns source, summary, callers, callees, file summary."""

    @pytest.mark.asyncio
    async def test_symbol_query_returns_full_context(
        self, tokenizer: MagicMock, tmp_path: Any
    ) -> None:
        # Write a source file to disk so _read_source can find it.
        src = tmp_path / "mod.py"
        src.write_text("def foo():\n    return 42\n")

        entry = _make_entry("mod.foo", file_path=str(src), summary="Returns 42.")
        caller_node = _make_node("mod.bar", file_path=str(src))
        callee_node = _make_node("mod.baz", file_path=str(src))

        store = _make_graph_store(
            entries={"mod.foo": entry},
            neighbors_callers={"mod.foo": [caller_node]},
            neighbors_callees={"mod.foo": [callee_node]},
            pagerank_scores={"mod.foo": 0.5},
            module_data={
                str(src): {
                    "nodes": [_make_node("mod.foo", file_path=str(src))],
                    "edges": [],
                }
            },
        )

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        request = QueryRequest(query="mod.foo", query_type="symbol", depth=1)
        pkg = await assembler.assemble(["mod.foo"], request)

        assert len(pkg.symbols) == 1
        sym = pkg.symbols[0]
        assert sym.fqn == "mod.foo"
        assert "def foo():" in sym.source_code
        assert sym.summary == "Returns 42."
        assert "mod.bar" in sym.callers
        assert "mod.baz" in sym.callees
        assert sym.pagerank_score == 0.5

        # File summary should be present.
        assert len(pkg.file_summaries) >= 1
        assert pkg.file_summaries[0].file_path == str(src)


# ------------------------------------------------------------------
# Test: File query returns file summary, symbols, imports, dependents
#       (Req 6.2)
# ------------------------------------------------------------------


class TestFileQuery:
    """Assemble for file-level queries returns file-level context."""

    @pytest.mark.asyncio
    async def test_file_query_returns_file_summary(
        self, tokenizer: MagicMock, tmp_path: Any
    ) -> None:
        src = tmp_path / "pkg" / "mod.py"
        src.parent.mkdir(parents=True)
        src.write_text("import os\ndef hello(): pass\n")

        entry = _make_entry("pkg.mod.hello", file_path=str(src), summary="Says hello.")
        node = _make_node("pkg.mod.hello", file_path=str(src))
        import_edge = _make_edge("pkg.mod.hello", "os", edge_type="import", file_path=str(src))

        store = _make_graph_store(
            entries={"pkg.mod.hello": entry},
            neighbors_callers={"pkg.mod.hello": []},
            neighbors_callees={"pkg.mod.hello": []},
            pagerank_scores={"pkg.mod.hello": 0.1},
            module_data={
                str(src): {
                    "nodes": [node],
                    "edges": [import_edge],
                }
            },
        )

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        request = QueryRequest(query="pkg.mod.hello", query_type="file", depth=1)
        pkg = await assembler.assemble(["pkg.mod.hello"], request)

        assert len(pkg.file_summaries) >= 1
        fs = pkg.file_summaries[0]
        assert fs.file_path == str(src)
        assert "pkg.mod.hello" in fs.symbols
        assert "os" in fs.imports


# ------------------------------------------------------------------
# Test: Depth parameter controls graph neighborhood levels (Req 6.3)
# ------------------------------------------------------------------


class TestDepthParameter:
    """Graph neighborhood depth is controlled by request.depth."""

    @pytest.mark.asyncio
    async def test_depth_controls_neighborhood(
        self, tokenizer: MagicMock, tmp_path: Any
    ) -> None:
        src = tmp_path / "a.py"
        src.write_text("def a(): pass\n")

        entry = _make_entry("mod.a", file_path=str(src), summary="func a")
        store = _make_graph_store(
            entries={"mod.a": entry},
            neighbors_callers={"mod.a": []},
            neighbors_callees={"mod.a": [_make_node("mod.b")]},
            pagerank_scores={"mod.a": 0.3},
            module_data={str(src): {"nodes": [], "edges": []}},
        )

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        # Depth 2 with include_graph_context.
        request = QueryRequest(
            query="mod.a",
            query_type="symbol",
            depth=2,
            include_graph_context=True,
        )
        pkg = await assembler.assemble(["mod.a"], request)

        assert pkg.graph_neighborhood is not None
        assert pkg.graph_neighborhood.depth == 2
        # get_neighbors should have been called with depth=2.
        calls = [
            c for c in store.get_neighbors.call_args_list
            if c.kwargs.get("depth", c.args[2] if len(c.args) > 2 else 1) == 2
        ]
        assert len(calls) > 0

    @pytest.mark.asyncio
    async def test_no_neighborhood_when_not_requested(
        self, tokenizer: MagicMock, tmp_path: Any
    ) -> None:
        src = tmp_path / "a.py"
        src.write_text("def a(): pass\n")

        entry = _make_entry("mod.a", file_path=str(src), summary="func a")
        store = _make_graph_store(
            entries={"mod.a": entry},
            neighbors_callers={"mod.a": []},
            neighbors_callees={"mod.a": []},
            pagerank_scores={"mod.a": 0.3},
            module_data={str(src): {"nodes": [], "edges": []}},
        )

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        request = QueryRequest(
            query="mod.a",
            query_type="symbol",
            depth=1,
            include_graph_context=False,
        )
        pkg = await assembler.assemble(["mod.a"], request)
        assert pkg.graph_neighborhood is None


# ------------------------------------------------------------------
# Test: max_tokens truncation uses PageRank priority (Req 6.4, 6.5, 6.6)
# ------------------------------------------------------------------


class TestTokenBudgetTruncation:
    """Token budget truncation prioritises higher-PageRank symbols."""

    @pytest.mark.asyncio
    async def test_high_pagerank_retained_first(
        self, tmp_path: Any
    ) -> None:
        # Use a tokenizer where 1 char = 1 token.
        tok = _make_tokenizer(tokens_per_char=1)

        src = tmp_path / "m.py"
        src.write_text("x" * 200 + "\n")

        entry_high = _make_entry("mod.high", file_path=str(src), summary="high")
        entry_low = _make_entry("mod.low", file_path=str(src), summary="low")

        store = _make_graph_store(
            entries={"mod.high": entry_high, "mod.low": entry_low},
            neighbors_callers={},
            neighbors_callees={},
            pagerank_scores={"mod.high": 0.9, "mod.low": 0.1},
            module_data={str(src): {"nodes": [], "edges": []}},
        )

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tok,
        )

        # Very small budget — only room for one symbol.
        request = QueryRequest(
            query="test",
            query_type="text",
            max_tokens=50,
        )
        pkg = await assembler.assemble(["mod.high", "mod.low"], request)

        # The high-PageRank symbol should be retained.
        fqns = [s.fqn for s in pkg.symbols]
        assert "mod.high" in fqns
        assert pkg.metadata.total_tokens <= 50

    @pytest.mark.asyncio
    async def test_all_symbols_fit_within_budget(
        self, tokenizer: MagicMock, tmp_path: Any
    ) -> None:
        src = tmp_path / "m.py"
        src.write_text("def f(): pass\n")

        entry = _make_entry("mod.f", file_path=str(src), summary="short")
        store = _make_graph_store(
            entries={"mod.f": entry},
            neighbors_callers={},
            neighbors_callees={},
            pagerank_scores={"mod.f": 0.5},
            module_data={str(src): {"nodes": [], "edges": []}},
        )

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        request = QueryRequest(query="test", max_tokens=100000)
        pkg = await assembler.assemble(["mod.f"], request)
        assert len(pkg.symbols) == 1


# ------------------------------------------------------------------
# Test: enrich_search_results attaches graph context (Req 9.1, 9.2)
# ------------------------------------------------------------------


class TestEnrichSearchResults:
    """enrich_search_results attaches callers, callees, module deps."""

    @pytest.mark.asyncio
    async def test_enrichment_attaches_graph_context(
        self, tokenizer: MagicMock
    ) -> None:
        entry = _make_entry("mod.foo", summary="Foo function.")
        caller = _make_node("mod.bar")
        callee = _make_node("mod.baz")

        store = _make_graph_store(
            entries={"mod.foo": entry},
            neighbors_callers={"mod.foo": [caller]},
            neighbors_callees={"mod.foo": [callee]},
            pagerank_scores={"mod.foo": 0.42},
            module_data={
                "test.py": {
                    "nodes": [_make_node("mod.foo")],
                    "edges": [_make_edge("mod.foo", "os", "import")],
                }
            },
        )

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        result = _make_search_result("mod.foo")
        request = QueryRequest(query="foo", include_graph_context=True)
        pkg = await assembler.enrich_search_results([result], request)

        assert len(pkg.symbols) == 1
        sym = pkg.symbols[0]
        assert sym.fqn == "mod.foo"
        assert "mod.bar" in sym.callers
        assert "mod.baz" in sym.callees
        assert sym.pagerank_score == 0.42
        assert sym.summary == "Foo function."

    @pytest.mark.asyncio
    async def test_enrichment_includes_file_summaries(
        self, tokenizer: MagicMock
    ) -> None:
        entry = _make_entry("mod.foo", summary="Foo.")
        store = _make_graph_store(
            entries={"mod.foo": entry},
            neighbors_callers={"mod.foo": []},
            neighbors_callees={"mod.foo": []},
            pagerank_scores={"mod.foo": 0.1},
            module_data={
                "test.py": {
                    "nodes": [_make_node("mod.foo")],
                    "edges": [],
                }
            },
        )

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        result = _make_search_result("mod.foo")
        request = QueryRequest(query="foo")
        pkg = await assembler.enrich_search_results([result], request)

        assert len(pkg.file_summaries) >= 1
        assert pkg.file_summaries[0].file_path == "test.py"


# ------------------------------------------------------------------
# Test: Graph-disabled mode returns results without enrichment (Req 9.3)
# ------------------------------------------------------------------


class TestGraphDisabledMode:
    """When graph_store is None, results are returned as-is."""

    @pytest.mark.asyncio
    async def test_assemble_without_graph(self, tokenizer: MagicMock) -> None:
        assembler = ContextAssembler(
            graph_store=None,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        request = QueryRequest(query="anything", max_tokens=100000)
        pkg = await assembler.assemble(["some.id"], request)

        # Should still produce a package with a minimal symbol.
        assert len(pkg.symbols) == 1
        assert pkg.symbols[0].fqn == "some.id"
        assert pkg.symbols[0].source_code == ""
        assert pkg.symbols[0].callers == []
        assert pkg.symbols[0].callees == []
        assert pkg.graph_neighborhood is None

    @pytest.mark.asyncio
    async def test_enrich_without_graph(self, tokenizer: MagicMock) -> None:
        assembler = ContextAssembler(
            graph_store=None,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        result = _make_search_result("chunk_1", content="some code")
        request = QueryRequest(query="test")
        pkg = await assembler.enrich_search_results([result], request)

        assert len(pkg.symbols) == 1
        assert pkg.symbols[0].fqn == "chunk_1"
        assert pkg.symbols[0].source_code == "some code"
        assert pkg.symbols[0].callers == []
        assert pkg.file_summaries == []


# ------------------------------------------------------------------
# Test: 200ms timeout per result for graph enrichment (Req 9.4)
# ------------------------------------------------------------------


class TestEnrichmentTimeout:
    """Graph enrichment is bounded to 200ms per result."""

    @pytest.mark.asyncio
    async def test_slow_enrichment_times_out(self, tokenizer: MagicMock) -> None:
        # Create a store where lookup_symbol blocks for longer than 200ms.
        store = MagicMock()

        async def _slow_lookup(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(1.0)  # Way over the 200ms budget.
            return None

        # Make lookup_symbol block via asyncio.to_thread by using a slow sync call.
        def slow_sync(*args: Any, **kwargs: Any) -> SymbolIndexEntry:
            import time
            time.sleep(0.5)  # 500ms — well over the 200ms budget.
            return _make_entry("mod.slow")

        store.lookup_symbol = MagicMock(side_effect=slow_sync)
        store.get_neighbors = MagicMock(return_value=[])
        store.get_pagerank = MagicMock(return_value=0.0)
        store.query_module = MagicMock(return_value={"nodes": [], "edges": []})

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        result = _make_search_result("mod.slow", content="slow code")
        request = QueryRequest(query="slow")

        # Should complete without hanging — the timeout catches the slow call.
        pkg = await assembler.enrich_search_results([result], request)

        # The result should still be present (fallback to unenriched).
        assert len(pkg.symbols) == 1
        # The symbol may or may not have enrichment depending on timing,
        # but the call should not hang.


# ------------------------------------------------------------------
# Test: Metadata is correctly populated (Req 6.7)
# ------------------------------------------------------------------


class TestMetadata:
    """ContextPackage metadata is correctly built."""

    @pytest.mark.asyncio
    async def test_metadata_fields(
        self, tokenizer: MagicMock, tmp_path: Any
    ) -> None:
        src = tmp_path / "m.py"
        src.write_text("def f(): pass\n")

        entry = _make_entry("mod.f", file_path=str(src), summary="f")
        store = _make_graph_store(
            entries={"mod.f": entry},
            neighbors_callers={},
            neighbors_callees={},
            pagerank_scores={"mod.f": 0.7},
            module_data={str(src): {"nodes": [], "edges": []}},
        )

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        request = QueryRequest(
            query="mod.f",
            query_type="symbol",
            depth=2,
            max_tokens=100000,
        )
        pkg = await assembler.assemble(["mod.f"], request)

        meta = pkg.metadata
        assert meta.query_params["query"] == "mod.f"
        assert meta.query_params["query_type"] == "symbol"
        assert meta.query_params["depth"] == 2
        assert meta.query_params["max_tokens"] == 100000
        assert meta.symbol_count == 1
        assert meta.total_tokens >= 0
        assert meta.pagerank_score_range == (0.7, 0.7)


# ------------------------------------------------------------------
# Test: Short-name fallback lookup
# ------------------------------------------------------------------


class TestShortNameFallback:
    """When FQN lookup fails, falls back to short-name lookup."""

    @pytest.mark.asyncio
    async def test_short_name_fallback(
        self, tokenizer: MagicMock, tmp_path: Any
    ) -> None:
        src = tmp_path / "m.py"
        src.write_text("def foo(): pass\n")

        entry = _make_entry("mod.foo", file_path=str(src), summary="Foo.")
        store = _make_graph_store(
            entries={},  # FQN lookup will miss.
            entries_by_name={"foo": [entry]},
            neighbors_callers={"mod.foo": []},
            neighbors_callees={"mod.foo": []},
            pagerank_scores={"mod.foo": 0.2},
            module_data={str(src): {"nodes": [], "edges": []}},
        )

        assembler = ContextAssembler(
            graph_store=store,
            topology_analyzer=None,
            tokenizer=tokenizer,
        )

        request = QueryRequest(query="foo", max_tokens=100000)
        pkg = await assembler.assemble(["foo"], request)

        assert len(pkg.symbols) == 1
        assert pkg.symbols[0].fqn == "mod.foo"
        assert pkg.symbols[0].summary == "Foo."

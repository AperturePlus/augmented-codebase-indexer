"""
Tests for MCP graph tool handlers (get_symbol_context, query_graph).

Validates that the new graph-related MCP tools return correct JSON
structures, handle graph-disabled mode, and handle missing symbols.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aci.core.config import ACIConfig
from aci.core.graph_models import (
    ContextMetadata,
    ContextPackage,
    GraphEdge,
    GraphNeighborhood,
    GraphNode,
    SymbolDetail,
)
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.mcp.context import MCPContext
from aci.mcp.handlers import call_tool
from aci.services import SearchService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    tmp_path: Path,
    graph_store: Any = None,
    query_router: Any = None,
    context_assembler: Any = None,
) -> MCPContext:
    """Build a minimal MCPContext with optional graph components."""
    config = ACIConfig()
    vector_store = InMemoryVectorStore()
    embedding_client = LocalEmbeddingClient()
    metadata_store = IndexMetadataStore(":memory:")
    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=None,
        grep_searcher=None,
        default_limit=config.search.default_limit,
    )

    # Stub indexing service — not used by graph handlers
    indexing_service = MagicMock()

    return MCPContext(
        config=config,
        search_service=search_service,
        indexing_service=indexing_service,
        metadata_store=metadata_store,
        vector_store=vector_store,
        indexing_lock=asyncio.Lock(),
        workspace_root=tmp_path,
        embedding_client=embedding_client,
        graph_store=graph_store,
        query_router=query_router,
        context_assembler=context_assembler,
    )


def _sample_context_package() -> ContextPackage:
    """Return a representative ContextPackage for testing serialization."""
    return ContextPackage(
        query="my.module.MyClass.my_method",
        symbols=[
            SymbolDetail(
                fqn="my.module.MyClass.my_method",
                source_code="def my_method(self): pass",
                summary="A test method.",
                callers=["my.module.caller_func"],
                callees=["my.module.helper"],
                pagerank_score=0.42,
            ),
        ],
        graph_neighborhood=GraphNeighborhood(
            nodes=[
                GraphNode(
                    symbol_id="my.module.MyClass.my_method",
                    symbol_name="my_method",
                    symbol_type="method",
                    file_path="my/module.py",
                    start_line=10,
                    end_line=11,
                    language="python",
                    pagerank_score=0.42,
                ),
            ],
            edges=[
                GraphEdge(
                    source_id="my.module.caller_func",
                    target_id="my.module.MyClass.my_method",
                    edge_type="call",
                ),
            ],
            depth=1,
        ),
        file_summaries=[],
        metadata=ContextMetadata(
            query_params={"query": "my.module.MyClass.my_method"},
            symbol_count=1,
            total_tokens=50,
            pagerank_score_range=(0.42, 0.42),
            backends_used=["search", "graph"],
        ),
    )


# ---------------------------------------------------------------------------
# Tool listing
# ---------------------------------------------------------------------------


def test_list_tools_includes_graph_tools():
    """New graph tools appear in the tool list."""
    from aci.mcp.tools import list_tools

    tools = list_tools()
    names = {t.name for t in tools}
    assert "get_symbol_context" in names
    assert "query_graph" in names


def test_get_symbol_context_tool_schema():
    """get_symbol_context has the expected required params."""
    from aci.mcp.tools import list_tools

    tools = list_tools()
    tool = next(t for t in tools if t.name == "get_symbol_context")
    assert set(tool.inputSchema["required"]) == {"symbol", "path"}
    props = tool.inputSchema["properties"]
    assert "depth" in props
    assert "max_tokens" in props
    assert "include_graph_context" in props


def test_query_graph_tool_schema():
    """query_graph has the expected required params and enum."""
    from aci.mcp.tools import list_tools

    tools = list_tools()
    tool = next(t for t in tools if t.name == "query_graph")
    assert set(tool.inputSchema["required"]) == {"symbol_or_path", "path", "query_type"}
    qt = tool.inputSchema["properties"]["query_type"]
    assert set(qt["enum"]) == {"callers", "callees", "dependencies", "dependents"}


# ---------------------------------------------------------------------------
# get_symbol_context handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_symbol_context_returns_valid_json(tmp_path: Path):
    """Handler returns a valid ContextPackage JSON."""
    router = AsyncMock()
    router.query.return_value = _sample_context_package()

    ctx = _make_ctx(tmp_path=tmp_path, query_router=router)
    try:
        result = await call_tool(
            "get_symbol_context",
            {"symbol": "my.module.MyClass.my_method", "path": str(tmp_path)},
            ctx,
        )

        assert len(result) == 1
        body = json.loads(result[0].text)
        assert body["query"] == "my.module.MyClass.my_method"
        assert len(body["symbols"]) == 1
        assert body["symbols"][0]["fqn"] == "my.module.MyClass.my_method"
        assert body["metadata"]["symbol_count"] == 1
    finally:
        ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_get_symbol_context_passes_params_to_router(tmp_path: Path):
    """Handler forwards depth, max_tokens, include_graph_context to QueryRequest."""
    router = AsyncMock()
    router.query.return_value = _sample_context_package()

    ctx = _make_ctx(tmp_path=tmp_path, query_router=router)
    try:
        await call_tool(
            "get_symbol_context",
            {
                "symbol": "foo.bar",
                "path": str(tmp_path),
                "depth": 2,
                "max_tokens": 4096,
                "include_graph_context": True,
            },
            ctx,
        )

        router.query.assert_awaited_once()
        req = router.query.call_args[0][0]
        assert req.query == "foo.bar"
        assert req.query_type == "symbol"
        assert req.depth == 2
        assert req.max_tokens == 4096
        assert req.include_graph_context is True
    finally:
        ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_get_symbol_context_graph_disabled(tmp_path: Path):
    """Returns descriptive error when query_router is None (graph disabled)."""
    ctx = _make_ctx(tmp_path=tmp_path, query_router=None)
    try:
        result = await call_tool(
            "get_symbol_context",
            {"symbol": "anything", "path": str(tmp_path)},
            ctx,
        )

        body = json.loads(result[0].text)
        assert body["error"] == "graph feature is disabled"
        assert "ACI_GRAPH_ENABLED" in body["hint"]
    finally:
        ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_get_symbol_context_empty_result(tmp_path: Path):
    """Returns an empty ContextPackage when symbol is not found."""
    empty_pkg = ContextPackage(
        query="nonexistent.symbol",
        metadata=ContextMetadata(
            query_params={"query": "nonexistent.symbol"},
            backends_used=["search"],
        ),
    )
    router = AsyncMock()
    router.query.return_value = empty_pkg

    ctx = _make_ctx(tmp_path=tmp_path, query_router=router)
    try:
        result = await call_tool(
            "get_symbol_context",
            {"symbol": "nonexistent.symbol", "path": str(tmp_path)},
            ctx,
        )

        body = json.loads(result[0].text)
        assert body["symbols"] == []
        assert body["query"] == "nonexistent.symbol"
    finally:
        ctx.metadata_store.close()


# ---------------------------------------------------------------------------
# query_graph handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_graph_returns_valid_json(tmp_path: Path):
    """Handler returns a valid GraphQueryResult JSON."""
    store = MagicMock()
    store.get_neighbors.return_value = [
        GraphNode(
            symbol_id="mod.caller",
            symbol_name="caller",
            symbol_type="function",
            file_path="mod.py",
            start_line=1,
            end_line=5,
            language="python",
        ),
    ]
    store.get_edges.return_value = [
        GraphEdge(
            source_id="mod.caller",
            target_id="mod.target",
            edge_type="call",
        ),
    ]

    ctx = _make_ctx(tmp_path=tmp_path, graph_store=store)
    try:
        result = await call_tool(
            "query_graph",
            {
                "symbol_or_path": "mod.target",
                "path": str(tmp_path),
                "query_type": "callers",
            },
            ctx,
        )

        body = json.loads(result[0].text)
        assert body["symbol"] == "mod.target"
        assert body["query_type"] == "callers"
        assert len(body["nodes"]) == 1
        assert body["nodes"][0]["symbol_id"] == "mod.caller"
        assert len(body["edges"]) == 1
        assert body["depth"] == 1
    finally:
        ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_query_graph_callees_direction(tmp_path: Path):
    """callees query_type maps to 'callees' direction."""
    store = MagicMock()
    store.get_neighbors.return_value = []
    store.get_edges.return_value = []

    ctx = _make_ctx(tmp_path=tmp_path, graph_store=store)
    try:
        await call_tool(
            "query_graph",
            {
                "symbol_or_path": "mod.func",
                "path": str(tmp_path),
                "query_type": "callees",
                "depth": 2,
                "include_inferred": False,
            },
            ctx,
        )

        store.get_neighbors.assert_called_once_with(
            "mod.func", "callees", depth=2, include_inferred=False
        )
        store.get_edges.assert_called_once_with(
            "mod.func", "callees", depth=2, include_inferred=False
        )
    finally:
        ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_query_graph_dependents_uses_callers_direction(tmp_path: Path):
    """dependents query_type maps to 'callers' direction."""
    store = MagicMock()
    store.get_neighbors.return_value = []
    store.get_edges.return_value = []

    ctx = _make_ctx(tmp_path=tmp_path, graph_store=store)
    try:
        await call_tool(
            "query_graph",
            {
                "symbol_or_path": "mod.py",
                "path": str(tmp_path),
                "query_type": "dependents",
            },
            ctx,
        )

        store.get_neighbors.assert_called_once_with(
            "mod.py", "callers", depth=1, include_inferred=True
        )
    finally:
        ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_query_graph_graph_disabled(tmp_path: Path):
    """Returns descriptive error when graph_store is None."""
    ctx = _make_ctx(tmp_path=tmp_path, graph_store=None)
    try:
        result = await call_tool(
            "query_graph",
            {
                "symbol_or_path": "anything",
                "path": str(tmp_path),
                "query_type": "callers",
            },
            ctx,
        )

        body = json.loads(result[0].text)
        assert body["error"] == "graph feature is disabled"
        assert "ACI_GRAPH_ENABLED" in body["hint"]
    finally:
        ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_query_graph_missing_symbol_returns_empty(tmp_path: Path):
    """Returns empty nodes/edges when symbol is not in the graph."""
    store = MagicMock()
    store.get_neighbors.return_value = []
    store.get_edges.return_value = []

    ctx = _make_ctx(tmp_path=tmp_path, graph_store=store)
    try:
        result = await call_tool(
            "query_graph",
            {
                "symbol_or_path": "nonexistent.symbol",
                "path": str(tmp_path),
                "query_type": "callers",
            },
            ctx,
        )

        body = json.loads(result[0].text)
        assert body["nodes"] == []
        assert body["edges"] == []
        assert body["symbol"] == "nonexistent.symbol"
    finally:
        ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_query_graph_invalid_query_type(tmp_path: Path):
    """Returns error for invalid query_type."""
    store = MagicMock()
    ctx = _make_ctx(tmp_path=tmp_path, graph_store=store)
    try:
        result = await call_tool(
            "query_graph",
            {
                "symbol_or_path": "mod.func",
                "path": str(tmp_path),
                "query_type": "invalid",
            },
            ctx,
        )

        assert "Error" in result[0].text
        assert "query_type" in result[0].text
    finally:
        ctx.metadata_store.close()

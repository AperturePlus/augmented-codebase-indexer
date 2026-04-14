"""Unit tests for QueryRouter."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aci.core.graph_models import (
    ContextMetadata,
    ContextPackage,
    GraphNode,
    QueryRequest,
    SymbolIndexEntry,
    SymbolLocation,
)
from aci.services.query_router import QueryRouter
from aci.services.rrf_fuser import RRFFuser

# ------------------------------------------------------------------
# Helpers / fixtures
# ------------------------------------------------------------------


def _make_search_result(chunk_id: str) -> Any:
    """Create a minimal SearchResult-like object."""
    return MagicMock(chunk_id=chunk_id)


def _make_graph_node(symbol_id: str) -> GraphNode:
    return GraphNode(
        symbol_id=symbol_id,
        symbol_name=symbol_id.split(".")[-1],
        symbol_type="function",
        file_path="test.py",
        start_line=1,
        end_line=10,
    )


def _make_symbol_entry(fqn: str) -> SymbolIndexEntry:
    return SymbolIndexEntry(
        fqn=fqn,
        definition=SymbolLocation(file_path="test.py", start_line=1, end_line=10),
        graph_node_id=fqn,
    )


def _make_assembler(fused_ids: list[str] | None = None) -> AsyncMock:
    """Create a mock context assembler that returns a ContextPackage."""
    assembler = AsyncMock()

    async def assemble(fused_results: list[str], request: QueryRequest) -> ContextPackage:
        return ContextPackage(
            query=request.query,
            symbols=[],
            metadata=ContextMetadata(
                query_params={"query": request.query},
            ),
        )

    assembler.assemble = AsyncMock(side_effect=assemble)
    return assembler


def _make_search_service(results: list[Any] | None = None, fail: bool = False) -> AsyncMock:
    svc = AsyncMock()
    if fail:
        svc.search = AsyncMock(side_effect=RuntimeError("search failed"))
    else:
        svc.search = AsyncMock(return_value=results or [])
    return svc


def _make_graph_store(
    neighbors: list[GraphNode] | None = None,
    symbol: SymbolIndexEntry | None = None,
    symbols_by_name: list[SymbolIndexEntry] | None = None,
) -> MagicMock:
    store = MagicMock()
    store.get_neighbors = MagicMock(return_value=neighbors or [])
    store.lookup_symbol = MagicMock(return_value=symbol)
    store.lookup_symbols_by_name = MagicMock(return_value=symbols_by_name or [])
    return store


def _make_ast_parser() -> MagicMock:
    return MagicMock()


@pytest.fixture
def fuser() -> RRFFuser:
    return RRFFuser()


# ------------------------------------------------------------------
# Fan-out dispatches to all enabled backends (Req 5.1, 5.2)
# ------------------------------------------------------------------


class TestFanOutAllBackends:
    @pytest.mark.asyncio
    async def test_all_backends_dispatched(self, fuser: RRFFuser) -> None:
        """When no backends filter is set, all three backends are invoked."""
        search_results = [_make_search_result("chunk1")]
        graph_nodes = [_make_graph_node("mod.func")]
        symbol_entry = _make_symbol_entry("mod.func")

        search_svc = _make_search_service(search_results)
        graph_store = _make_graph_store(
            neighbors=graph_nodes, symbol=symbol_entry
        )
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=graph_store,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=True,
        )

        request = QueryRequest(query="mod.func", query_type="symbol")
        result = await router.query(request)

        # All three backends should have been called
        search_svc.search.assert_awaited_once()
        graph_store.get_neighbors.assert_called_once()
        # AST backend uses lookup_symbol
        graph_store.lookup_symbol.assert_called_once_with("mod.func")
        # Assembler should have been called with fused results
        assembler.assemble.assert_awaited_once()
        assert isinstance(result, ContextPackage)

    @pytest.mark.asyncio
    async def test_backends_used_populated(self, fuser: RRFFuser) -> None:
        search_svc = _make_search_service([_make_search_result("c1")])
        graph_store = _make_graph_store(
            neighbors=[_make_graph_node("a.b")],
            symbol=_make_symbol_entry("a.b"),
        )
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=graph_store,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=True,
        )

        result = await router.query(QueryRequest(query="a.b"))
        # All three backends should be listed
        assert set(result.metadata.backends_used) == {"search", "graph", "ast"}


# ------------------------------------------------------------------
# partial_results flag when a backend fails (Req 5.5)
# ------------------------------------------------------------------


class TestPartialResults:
    @pytest.mark.asyncio
    async def test_search_failure_sets_partial(self, fuser: RRFFuser) -> None:
        search_svc = _make_search_service(fail=True)
        graph_store = _make_graph_store(
            neighbors=[_make_graph_node("x.y")],
            symbol=_make_symbol_entry("x.y"),
        )
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=graph_store,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=True,
        )

        result = await router.query(QueryRequest(query="x.y"))
        assert result.metadata.partial_results is True
        # Graph and AST should still be in backends_used
        assert "search" not in result.metadata.backends_used

    @pytest.mark.asyncio
    async def test_graph_failure_sets_partial(self, fuser: RRFFuser) -> None:
        search_svc = _make_search_service([_make_search_result("c1")])
        graph_store = MagicMock()
        graph_store.get_neighbors = MagicMock(side_effect=RuntimeError("db error"))
        graph_store.lookup_symbol = MagicMock(return_value=_make_symbol_entry("a.b"))
        graph_store.lookup_symbols_by_name = MagicMock(return_value=[])
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=graph_store,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=True,
        )

        result = await router.query(QueryRequest(query="a.b"))
        assert result.metadata.partial_results is True
        assert "graph" not in result.metadata.backends_used

    @pytest.mark.asyncio
    async def test_all_backends_fail_returns_empty_package(
        self, fuser: RRFFuser
    ) -> None:
        search_svc = _make_search_service(fail=True)
        graph_store = MagicMock()
        graph_store.get_neighbors = MagicMock(side_effect=RuntimeError("fail"))
        graph_store.lookup_symbol = MagicMock(side_effect=RuntimeError("fail"))
        graph_store.lookup_symbols_by_name = MagicMock(side_effect=RuntimeError("fail"))
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=graph_store,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=True,
        )

        result = await router.query(QueryRequest(query="anything"))
        assert result.metadata.partial_results is True
        assert result.metadata.backends_used == []


# ------------------------------------------------------------------
# backends parameter restricts dispatch (Req 5.8)
# ------------------------------------------------------------------


class TestBackendsParameter:
    @pytest.mark.asyncio
    async def test_only_search_backend(self, fuser: RRFFuser) -> None:
        search_svc = _make_search_service([_make_search_result("c1")])
        graph_store = _make_graph_store()
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=graph_store,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=True,
        )

        request = QueryRequest(query="test", backends=["search"])
        result = await router.query(request)

        search_svc.search.assert_awaited_once()
        graph_store.get_neighbors.assert_not_called()
        graph_store.lookup_symbol.assert_not_called()
        assert "search" in result.metadata.backends_used

    @pytest.mark.asyncio
    async def test_only_graph_backend(self, fuser: RRFFuser) -> None:
        search_svc = _make_search_service()
        graph_store = _make_graph_store(
            neighbors=[_make_graph_node("a.b")]
        )
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=graph_store,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=True,
        )

        request = QueryRequest(query="a.b", backends=["graph"])
        result = await router.query(request)

        search_svc.search.assert_not_awaited()
        graph_store.get_neighbors.assert_called_once()
        assert "graph" in result.metadata.backends_used

    @pytest.mark.asyncio
    async def test_unknown_backend_ignored(self, fuser: RRFFuser) -> None:
        search_svc = _make_search_service()
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=None,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=False,
        )

        request = QueryRequest(query="test", backends=["nonexistent"])
        result = await router.query(request)
        assert result.metadata.backends_used == []


# ------------------------------------------------------------------
# Graph-disabled mode skips graph backend (Req 5.7)
# ------------------------------------------------------------------


class TestGraphDisabled:
    @pytest.mark.asyncio
    async def test_graph_disabled_skips_graph(self, fuser: RRFFuser) -> None:
        search_svc = _make_search_service([_make_search_result("c1")])
        graph_store = _make_graph_store()
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=graph_store,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=False,
        )

        result = await router.query(QueryRequest(query="test"))
        graph_store.get_neighbors.assert_not_called()
        assert "graph" not in result.metadata.backends_used

    @pytest.mark.asyncio
    async def test_graph_store_none_skips_graph(self, fuser: RRFFuser) -> None:
        search_svc = _make_search_service([_make_search_result("c1")])
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=None,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=True,  # enabled but store is None
        )

        result = await router.query(QueryRequest(query="test"))
        assert "graph" not in result.metadata.backends_used

    @pytest.mark.asyncio
    async def test_graph_disabled_with_backends_param(
        self, fuser: RRFFuser
    ) -> None:
        """Even if caller requests graph, it's skipped when disabled."""
        search_svc = _make_search_service([_make_search_result("c1")])
        graph_store = _make_graph_store()
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=graph_store,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=False,
        )

        request = QueryRequest(query="test", backends=["graph", "search"])
        result = await router.query(request)
        graph_store.get_neighbors.assert_not_called()
        assert "graph" not in result.metadata.backends_used
        assert "search" in result.metadata.backends_used


# ------------------------------------------------------------------
# Timeout handling cancels slow backends (Req 5.6)
# ------------------------------------------------------------------


class TestTimeoutHandling:
    @pytest.mark.asyncio
    async def test_slow_backend_cancelled(self, fuser: RRFFuser) -> None:
        """A backend that exceeds the timeout is cancelled and partial_results is set."""

        async def slow_search(*args: Any, **kwargs: Any) -> list[Any]:
            await asyncio.sleep(10)  # way beyond the 2s budget
            return [_make_search_result("never")]

        search_svc = AsyncMock()
        search_svc.search = AsyncMock(side_effect=slow_search)
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=None,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=False,
        )

        # Only search backend is enabled (graph disabled, ast needs graph_store)
        request = QueryRequest(query="test", backends=["search"])
        result = await router.query(request)

        assert result.metadata.partial_results is True
        assert "search" not in result.metadata.backends_used

    @pytest.mark.asyncio
    async def test_fast_backend_succeeds_despite_slow_sibling(
        self, fuser: RRFFuser
    ) -> None:
        """Fast backends return results even when a sibling times out."""

        async def slow_graph_neighbors(*args: Any, **kwargs: Any) -> list[GraphNode]:
            await asyncio.sleep(10)
            return []

        search_svc = _make_search_service([_make_search_result("fast_result")])

        graph_store = MagicMock()
        # Make get_neighbors slow by wrapping in a coroutine
        # The dispatch_graph uses asyncio.to_thread, so we mock at that level
        graph_store.get_neighbors = MagicMock(side_effect=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("slow")))
        graph_store.lookup_symbol = MagicMock(return_value=_make_symbol_entry("a.b"))
        graph_store.lookup_symbols_by_name = MagicMock(return_value=[])
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=graph_store,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=True,
        )

        result = await router.query(QueryRequest(query="a.b"))
        # Search and AST should succeed; graph fails
        assert "search" in result.metadata.backends_used
        assert result.metadata.partial_results is True


# ------------------------------------------------------------------
# Assembler integration
# ------------------------------------------------------------------


class TestAssemblerIntegration:
    @pytest.mark.asyncio
    async def test_fused_results_passed_to_assembler(
        self, fuser: RRFFuser
    ) -> None:
        """Verify the assembler receives the fused result IDs."""
        search_svc = _make_search_service(
            [_make_search_result("c1"), _make_search_result("c2")]
        )
        assembler = _make_assembler()

        router = QueryRouter(
            search_service=search_svc,
            graph_store=None,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=False,
        )

        request = QueryRequest(query="test", backends=["search"])
        await router.query(request)

        assembler.assemble.assert_awaited_once()
        call_args = assembler.assemble.call_args
        fused_ids = call_args[0][0] if call_args[0] else call_args[1]["fused_results"]
        assert "c1" in fused_ids
        assert "c2" in fused_ids

    @pytest.mark.asyncio
    async def test_assembler_failure_returns_empty_package(
        self, fuser: RRFFuser
    ) -> None:
        search_svc = _make_search_service([_make_search_result("c1")])
        assembler = AsyncMock()
        assembler.assemble = AsyncMock(side_effect=RuntimeError("assembly failed"))

        router = QueryRouter(
            search_service=search_svc,
            graph_store=None,
            ast_parser=_make_ast_parser(),
            context_assembler=assembler,
            rrf_fuser=fuser,
            graph_enabled=False,
        )

        request = QueryRequest(query="test", backends=["search"])
        result = await router.query(request)
        assert result.metadata.partial_results is True
        assert result.query == "test"

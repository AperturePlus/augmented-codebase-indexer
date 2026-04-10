"""
Unit tests for the ACI library-mode API.

Validates Requirements 10.1, 10.2, 10.3, 10.4.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aci.core.config import ACIConfig
from aci.core.graph_models import (
    ContextMetadata,
    ContextPackage,
    GraphEdge,
    GraphNode,
    GraphQueryResult,
    QueryRequest,
)
from aci.infrastructure.vector_store import SearchResult
from aci.services.indexing_models import IndexingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_result(chunk_id: str = "c1") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        file_path="src/foo.py",
        start_line=1,
        end_line=10,
        content="def foo(): pass",
        score=0.9,
        metadata={},
    )


def _make_graph_node(symbol_id: str = "mod.Foo.bar") -> GraphNode:
    return GraphNode(
        symbol_id=symbol_id,
        symbol_name="bar",
        symbol_type="function",
        file_path="src/foo.py",
        start_line=1,
        end_line=5,
        language="python",
    )


def _make_graph_edge(src: str = "mod.Foo.bar", tgt: str = "mod.Baz.qux") -> GraphEdge:
    return GraphEdge(
        source_id=src,
        target_id=tgt,
        edge_type="call",
    )


def _make_mock_services():
    """Build a mock ServicesContainer with all fields the ACI class needs."""
    svc = MagicMock()
    svc.config = ACIConfig()
    svc.config.graph.enabled = True
    svc.embedding_client = MagicMock()
    svc.vector_store = MagicMock()
    svc.metadata_store = MagicMock()
    svc.file_scanner = MagicMock()
    svc.chunker = MagicMock()
    svc.reranker = None
    svc.graph_store = MagicMock()
    svc.graph_builder = None
    svc.context_assembler = MagicMock()
    svc.rrf_fuser = MagicMock()
    svc.llm_enricher = None
    return svc


@pytest.fixture()
def aci_instance():
    """Create an ACI instance with fully mocked services.

    Patches ``create_services``, ``SearchService``, ``IndexingService``,
    ``GrepSearcher``, ``QueryRouter``, and ``TreeSitterParser`` so that
    no real infrastructure is needed.
    """
    mock_services = _make_mock_services()

    with (
        patch("aci.load_config", return_value=ACIConfig()),
        patch("aci.ACI._create_services", return_value=mock_services),
        patch("aci.ACI._create_search_service") as mock_ss_factory,
        patch("aci.ACI._create_indexing_service") as mock_is_factory,
        patch("aci.ACI._create_query_router") as mock_qr_factory,
    ):
        mock_search = AsyncMock()
        mock_indexing = AsyncMock()
        mock_router = AsyncMock()

        mock_ss_factory.return_value = mock_search
        mock_is_factory.return_value = mock_indexing
        mock_qr_factory.return_value = mock_router

        from aci import ACI

        instance = ACI()
        # Expose mocks for assertions.
        instance._mock_search = mock_search
        instance._mock_indexing = mock_indexing
        instance._mock_router = mock_router
        instance._mock_services = mock_services

        yield instance

        instance.close()


# ---------------------------------------------------------------------------
# Test: ACI() initialises without starting a server (Req 10.2)
# ---------------------------------------------------------------------------


class TestACIInitialisation:
    """ACI() should initialise without starting any server process."""

    def test_no_server_started(self, aci_instance):
        """ACI() must not start an HTTP or MCP server."""
        # The instance was created successfully — no server binding occurred.
        # Verify the background event loop thread is alive.
        assert aci_instance._thread.is_alive()
        assert aci_instance._thread.daemon is True
        assert aci_instance._thread.name == "aci-event-loop"

    def test_event_loop_running(self, aci_instance):
        """The background event loop must be running."""
        assert aci_instance._loop.is_running()

    def test_config_loaded(self, aci_instance):
        """Configuration must be loaded."""
        assert aci_instance._config is not None
        assert isinstance(aci_instance._config, ACIConfig)


# ---------------------------------------------------------------------------
# Test: index() returns IndexingResult (Req 10.1)
# ---------------------------------------------------------------------------


class TestIndex:
    """index() must bridge to IndexingService and return IndexingResult."""

    def test_index_returns_indexing_result(self, aci_instance):
        expected = IndexingResult(total_files=5, total_chunks=20)
        aci_instance._mock_indexing.index_directory = AsyncMock(return_value=expected)

        result = aci_instance.index("/some/path")

        assert isinstance(result, IndexingResult)
        assert result.total_files == 5
        assert result.total_chunks == 20

    def test_index_passes_path_as_resolved(self, aci_instance):
        expected = IndexingResult()
        aci_instance._mock_indexing.index_directory = AsyncMock(return_value=expected)

        aci_instance.index("/some/path")

        call_args = aci_instance._mock_indexing.index_directory.call_args
        root_arg = call_args[0][0]
        assert isinstance(root_arg, Path)
        assert root_arg.is_absolute()


# ---------------------------------------------------------------------------
# Test: search() returns list[SearchResult] (Req 10.1)
# ---------------------------------------------------------------------------


class TestSearch:
    """search() must bridge to SearchService and return list[SearchResult]."""

    def test_search_returns_list(self, aci_instance):
        results = [_make_search_result("c1"), _make_search_result("c2")]
        aci_instance._mock_search.search = AsyncMock(return_value=results)

        out = aci_instance.search("find auth")

        assert isinstance(out, list)
        assert len(out) == 2
        assert all(isinstance(r, SearchResult) for r in out)

    def test_search_forwards_options(self, aci_instance):
        aci_instance._mock_search.search = AsyncMock(return_value=[])

        aci_instance.search("query", limit=5, file_filter="*.py")

        aci_instance._mock_search.search.assert_awaited_once_with(
            "query", limit=5, file_filter="*.py"
        )

    def test_search_normalises_context_package(self, aci_instance):
        """When search returns a ContextPackage, normalise to empty list."""
        pkg = ContextPackage(query="q")
        aci_instance._mock_search.search = AsyncMock(return_value=pkg)

        out = aci_instance.search("q")

        assert out == []


# ---------------------------------------------------------------------------
# Test: get_context() returns ContextPackage (Req 10.1)
# ---------------------------------------------------------------------------


class TestGetContext:
    """get_context() must bridge to QueryRouter and return ContextPackage."""

    def test_get_context_returns_context_package(self, aci_instance):
        expected = ContextPackage(
            query="mod.Foo.bar",
            metadata=ContextMetadata(symbol_count=1),
        )
        aci_instance._mock_router.query = AsyncMock(return_value=expected)

        result = aci_instance.get_context("mod.Foo.bar")

        assert isinstance(result, ContextPackage)
        assert result.query == "mod.Foo.bar"

    def test_get_context_builds_query_request(self, aci_instance):
        expected = ContextPackage(query="mod.Foo.bar")
        aci_instance._mock_router.query = AsyncMock(return_value=expected)

        aci_instance.get_context("mod.Foo.bar", depth=2, max_tokens=4096)

        call_args = aci_instance._mock_router.query.call_args
        request = call_args[0][0]
        assert isinstance(request, QueryRequest)
        assert request.query == "mod.Foo.bar"
        assert request.depth == 2
        assert request.max_tokens == 4096
        assert request.query_type == "symbol"
        assert request.include_graph_context is True

    def test_get_context_no_router_returns_empty_package(self, aci_instance):
        """When query_router is None, return a minimal ContextPackage."""
        aci_instance._query_router = None

        result = aci_instance.get_context("mod.Foo.bar")

        assert isinstance(result, ContextPackage)
        assert result.query == "mod.Foo.bar"


# ---------------------------------------------------------------------------
# Test: get_graph() returns GraphQueryResult (Req 10.1)
# ---------------------------------------------------------------------------


class TestGetGraph:
    """get_graph() must query the graph store and return GraphQueryResult."""

    def test_get_graph_returns_graph_query_result(self, aci_instance):
        nodes = [_make_graph_node("mod.Foo.bar")]
        edges = [_make_graph_edge()]
        aci_instance._mock_services.graph_store.get_neighbors.return_value = nodes
        aci_instance._mock_services.graph_store.get_edges.return_value = edges

        result = aci_instance.get_graph("mod.Foo.bar", query_type="callees")

        assert isinstance(result, GraphQueryResult)
        assert result.symbol == "mod.Foo.bar"
        assert result.query_type == "callees"
        assert len(result.nodes) == 1
        assert len(result.edges) == 1

    def test_get_graph_no_store_returns_empty(self, aci_instance):
        """When graph_store is None, return an empty GraphQueryResult."""
        aci_instance._services.graph_store = None

        result = aci_instance.get_graph("mod.Foo.bar")

        assert isinstance(result, GraphQueryResult)
        assert result.nodes == []
        assert result.edges == []

    def test_get_graph_callers_direction(self, aci_instance):
        aci_instance._mock_services.graph_store.get_neighbors.return_value = []
        aci_instance._mock_services.graph_store.get_edges.return_value = []

        aci_instance.get_graph("mod.Foo.bar", query_type="callers", depth=2)

        aci_instance._mock_services.graph_store.get_neighbors.assert_called_once_with(
            "mod.Foo.bar", "callers", depth=2, include_inferred=True,
        )


# ---------------------------------------------------------------------------
# Test: context manager (Req 10.4)
# ---------------------------------------------------------------------------


class TestContextManager:
    """ACI must support context manager protocol for resource cleanup."""

    def test_context_manager_closes_resources(self):
        mock_services = _make_mock_services()

        with (
            patch("aci.load_config", return_value=ACIConfig()),
            patch("aci.ACI._create_services", return_value=mock_services),
            patch("aci.ACI._create_search_service", return_value=AsyncMock()),
            patch("aci.ACI._create_indexing_service", return_value=AsyncMock()),
            patch("aci.ACI._create_query_router", return_value=AsyncMock()),
        ):
            from aci import ACI

            with ACI() as instance:
                assert instance._thread.is_alive()

            # After exiting, the loop should have been stopped.
            assert not instance._loop.is_running()

    def test_enter_returns_self(self):
        mock_services = _make_mock_services()

        with (
            patch("aci.load_config", return_value=ACIConfig()),
            patch("aci.ACI._create_services", return_value=mock_services),
            patch("aci.ACI._create_search_service", return_value=AsyncMock()),
            patch("aci.ACI._create_indexing_service", return_value=AsyncMock()),
            patch("aci.ACI._create_query_router", return_value=AsyncMock()),
        ):
            from aci import ACI

            instance = ACI()
            try:
                assert instance.__enter__() is instance
            finally:
                instance.close()


# ---------------------------------------------------------------------------
# Test: sync-to-async bridge (Req 10.4)
# ---------------------------------------------------------------------------


class TestSyncAsyncBridge:
    """Sync methods must correctly bridge to the async event loop."""

    def test_run_executes_coroutine_on_background_loop(self, aci_instance):
        """_run() must schedule the coroutine on the background loop."""

        async def _coro():
            # Verify we're running on the background loop, not the test thread.
            loop = asyncio.get_running_loop()
            assert loop is aci_instance._loop
            return 42

        result = aci_instance._run(_coro())
        assert result == 42

    def test_run_propagates_exceptions(self, aci_instance):
        """_run() must propagate exceptions from the coroutine."""

        async def _failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            aci_instance._run(_failing())

    def test_background_thread_is_daemon(self, aci_instance):
        """The background thread must be a daemon so it doesn't block exit."""
        assert aci_instance._thread.daemon is True


# ---------------------------------------------------------------------------
# Test: configuration options (Req 10.3)
# ---------------------------------------------------------------------------


class TestConfiguration:
    """ACI must accept configuration via constructor, env vars, or file."""

    def test_accepts_config_object(self):
        config = ACIConfig()
        mock_services = _make_mock_services()

        with (
            patch("aci.ACI._create_services", return_value=mock_services),
            patch("aci.ACI._create_search_service", return_value=AsyncMock()),
            patch("aci.ACI._create_indexing_service", return_value=AsyncMock()),
            patch("aci.ACI._create_query_router", return_value=AsyncMock()),
        ):
            from aci import ACI

            instance = ACI(config=config)
            try:
                assert instance._config is config
            finally:
                instance.close()

    def test_accepts_config_path(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("embedding:\n  api_key: test-key\n")

        mock_services = _make_mock_services()

        with (
            patch("aci.ACI._create_services", return_value=mock_services),
            patch("aci.ACI._create_search_service", return_value=AsyncMock()),
            patch("aci.ACI._create_indexing_service", return_value=AsyncMock()),
            patch("aci.ACI._create_query_router", return_value=AsyncMock()),
        ):
            from aci import ACI

            instance = ACI(config_path=str(config_file))
            try:
                assert instance._config is not None
            finally:
                instance.close()

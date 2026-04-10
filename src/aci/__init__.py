"""
Augmented Codebase Indexer (Project ACI)

A Python tool for semantic code search with precise line-level location results.

Library usage::

    from aci import ACI

    with ACI() as aci:
        aci.index("/path/to/repo")
        results = aci.search("authentication logic")
        ctx = aci.get_context("my_module.MyClass.my_method")
        graph = aci.get_graph("my_module.MyClass.my_method", query_type="callees")
"""

from __future__ import annotations

__version__ = "0.2.0"

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

from aci.core.config import ACIConfig, load_config
from aci.core.graph_models import ContextPackage, GraphQueryResult, QueryRequest
from aci.infrastructure.vector_store import SearchResult
from aci.services.indexing_models import IndexingResult

logger = logging.getLogger(__name__)


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Target for the background daemon thread running the event loop."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ACI:
    """Public library API for ACI.

    Provides synchronous methods that bridge to the async service layer
    via a dedicated background event loop on a daemon thread.

    Usage::

        from aci import ACI

        aci = ACI()
        aci.index("/path/to/repo")
        results = aci.search("authentication logic")
        ctx = aci.get_context("my_module.MyClass.my_method")
        graph = aci.get_graph("my_module.MyClass", query_type="callees")
        aci.close()

    Or as a context manager::

        with ACI() as aci:
            aci.index("/path/to/repo")
            results = aci.search("find auth")
    """

    def __init__(
        self,
        config: ACIConfig | None = None,
        config_path: str | None = None,
    ) -> None:
        """Initialise ACI with configuration.

        Creates a dedicated event loop on a background daemon thread.
        The loop is started immediately and shut down in :meth:`close`.

        Args:
            config: Pre-built configuration object.  When *None*,
                configuration is loaded from *config_path* or the
                environment (same behaviour as ``load_config``).
            config_path: Path to a YAML/JSON configuration file.
                Ignored when *config* is provided.
        """
        # --- configuration ---------------------------------------------------
        if config is not None:
            self._config = config
        elif config_path is not None:
            self._config = load_config(config_path)
        else:
            self._config = load_config()

        # --- background event loop -------------------------------------------
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=_run_loop, args=(self._loop,), daemon=True, name="aci-event-loop"
        )
        self._thread.start()

        # --- service wiring (mirrors create_mcp_context) ---------------------
        self._services = self._create_services()
        self._search_service = self._create_search_service()
        self._indexing_service = self._create_indexing_service()
        self._query_router = self._create_query_router()

    # ------------------------------------------------------------------
    # Internal wiring helpers
    # ------------------------------------------------------------------

    def _create_services(self):
        """Create the shared services container."""
        from aci.services.container import create_services

        return create_services()

    def _create_search_service(self):
        """Build a SearchService wired to the services container."""
        from aci.infrastructure.grep_searcher import GrepSearcher
        from aci.services.search_service import SearchService

        grep_searcher = GrepSearcher(base_path=str(Path.cwd()))
        return SearchService(
            embedding_client=self._services.embedding_client,
            vector_store=self._services.vector_store,
            reranker=self._services.reranker,
            grep_searcher=grep_searcher,
            context_assembler=self._services.context_assembler,
            default_limit=self._services.config.search.default_limit,
        )

    def _create_indexing_service(self):
        """Build an IndexingService wired to the services container."""
        from aci.services.indexing_service import IndexingService

        return IndexingService(
            embedding_client=self._services.embedding_client,
            vector_store=self._services.vector_store,
            metadata_store=self._services.metadata_store,
            file_scanner=self._services.file_scanner,
            chunker=self._services.chunker,
            batch_size=self._services.config.embedding.batch_size,
            max_workers=self._services.config.indexing.max_workers,
            graph_builder=self._services.graph_builder,
        )

    def _create_query_router(self):
        """Build a QueryRouter if the required components are available."""
        svc = self._services
        if svc.context_assembler is None or svc.rrf_fuser is None:
            return None

        from aci.core.ast_parser import TreeSitterParser
        from aci.services.query_router import QueryRouter

        return QueryRouter(
            search_service=self._search_service,
            graph_store=svc.graph_store,
            ast_parser=TreeSitterParser(),
            context_assembler=svc.context_assembler,
            rrf_fuser=svc.rrf_fuser,
            graph_enabled=svc.config.graph.enabled,
        )

    # ------------------------------------------------------------------
    # Sync ↔ async bridge
    # ------------------------------------------------------------------

    def _run(self, coro: Any) -> Any:
        """Schedule *coro* on the background loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, path: str, **options: Any) -> IndexingResult:
        """Index a codebase directory.

        Args:
            path: Root directory to index.
            **options: Forwarded to
                :meth:`IndexingService.index_directory`.

        Returns:
            :class:`IndexingResult` with statistics.
        """
        root = Path(path).resolve()
        return self._run(
            self._indexing_service.index_directory(root, **options)
        )

    def search(self, query: str, **options: Any) -> list[SearchResult]:
        """Perform semantic search.

        Args:
            query: Natural language search query.
            **options: Forwarded to :meth:`SearchService.search`.

        Returns:
            List of :class:`SearchResult` sorted by relevance.
        """
        result = self._run(self._search_service.search(query, **options))
        # search() may return a ContextPackage when include_graph_context
        # is True; normalise to a plain list for the library API.
        if isinstance(result, list):
            return result
        return []

    def get_context(
        self, symbol_or_path: str, **options: Any
    ) -> ContextPackage:
        """Retrieve structured context for a symbol or file.

        Args:
            symbol_or_path: Fully-qualified symbol name or file path.
            **options: Override fields on the :class:`QueryRequest`
                (e.g. ``depth=2``, ``max_tokens=4096``).

        Returns:
            :class:`ContextPackage` with source, summaries, and graph
            neighbourhood.
        """
        if self._query_router is None:
            return ContextPackage(query=symbol_or_path)

        request = QueryRequest(
            query=symbol_or_path,
            query_type=options.pop("query_type", "symbol"),
            depth=options.pop("depth", 1),
            max_tokens=options.pop("max_tokens", 8192),
            include_graph_context=options.pop("include_graph_context", True),
            backends=options.pop("backends", None),
            rrf_k=options.pop("rrf_k", 60),
        )
        return self._run(self._query_router.query(request))

    def get_graph(
        self, symbol_or_path: str, **options: Any
    ) -> GraphQueryResult:
        """Query the code graph for a symbol or module.

        Args:
            symbol_or_path: Fully-qualified symbol name or module path.
            **options: ``query_type`` (``"callers"`` | ``"callees"`` |
                ``"dependencies"`` | ``"dependents"``), ``depth``,
                ``include_inferred``.

        Returns:
            :class:`GraphQueryResult` with nodes and edges.
        """
        graph_store = self._services.graph_store
        if graph_store is None:
            return GraphQueryResult(
                symbol=symbol_or_path,
                query_type=options.get("query_type", "callees"),
            )

        query_type = options.get("query_type", "callees")
        depth = options.get("depth", 1)
        include_inferred = options.get("include_inferred", True)

        direction = query_type
        if query_type in ("callers", "dependents"):
            direction = "callers"
        elif query_type in ("callees", "dependencies"):
            direction = "callees"

        nodes = graph_store.get_neighbors(
            symbol_or_path, direction, depth=depth,
            include_inferred=include_inferred,
        )
        edges = graph_store.get_edges(
            symbol_or_path, direction, depth=depth,
            include_inferred=include_inferred,
        )

        return GraphQueryResult(
            symbol=symbol_or_path,
            query_type=query_type,
            nodes=nodes,
            edges=edges,
            depth=depth,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the background event loop and release resources."""
        # Clean up services that need explicit closing.
        if self._services.graph_store is not None:
            try:
                self._services.graph_store.close()
            except Exception:
                logger.debug("Error closing graph store", exc_info=True)

        if self._services.llm_enricher is not None:
            try:
                self._run(self._services.llm_enricher.close())
            except Exception:
                logger.debug("Error closing LLM enricher", exc_info=True)

        # Stop the event loop and wait for the thread to exit.
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def __enter__(self) -> ACI:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# Re-export create_app for backward compatibility.
from aci.http_server import create_app  # noqa: E402

__all__ = ["ACI", "create_app"]

"""
Unified query router with parallel fan-out and RRF fusion.

Accepts a :class:`QueryRequest`, dispatches to enabled analysis backends
in parallel, fuses the ranked result lists via :class:`RRFFuser`, and
forwards the unified ranking to the context assembler for final
packaging into a :class:`ContextPackage`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Protocol

from aci.core.graph_models import (
    ContextMetadata,
    ContextPackage,
    QueryRequest,
)
from aci.services.rrf_fuser import RRFFuser

if TYPE_CHECKING:
    from aci.core.ast_parser import ASTParserInterface
    from aci.core.graph_store import GraphStoreInterface
    from aci.services.search_service import SearchService

logger = logging.getLogger(__name__)

# Total time budget for the full fan-out → fuse → assemble cycle (Req 5.6).
_TIMEOUT_SECONDS: float = 2.0


class ContextAssemblerProtocol(Protocol):
    """Minimal protocol the router expects from the context assembler."""

    async def assemble(
        self,
        fused_results: list[str],
        request: QueryRequest,
    ) -> ContextPackage: ...


class QueryRouter:
    """Fan-out coordinator for unified code queries.

    Dispatches to enabled backends in parallel via ``asyncio.gather``,
    collects results, fuses via :class:`RRFFuser`, then forwards to
    the context assembler.
    """

    # Recognised backend names used in ``QueryRequest.backends``.
    BACKEND_SEARCH = "search"
    BACKEND_GRAPH = "graph"
    BACKEND_AST = "ast"
    _ALL_BACKENDS = {BACKEND_SEARCH, BACKEND_GRAPH, BACKEND_AST}

    def __init__(
        self,
        search_service: SearchService,
        graph_store: GraphStoreInterface | None,
        ast_parser: ASTParserInterface,
        context_assembler: ContextAssemblerProtocol,
        rrf_fuser: RRFFuser,
        graph_enabled: bool = True,
    ) -> None:
        self._search = search_service
        self._graph_store = graph_store
        self._ast_parser = ast_parser
        self._assembler = context_assembler
        self._fuser = rrf_fuser
        self._graph_enabled = graph_enabled and graph_store is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(self, request: QueryRequest) -> ContextPackage:
        """Fan out to enabled backends, fuse results, assemble context.

        Backend dispatch:
        - **search**: always enabled (vector + grep via SearchService).
        - **graph**: skipped when ``graph_enabled=False`` or
          ``graph_store is None``.
        - **ast**: structural symbol lookup for ``query_type="symbol"``.

        If ``request.backends`` is set, only the listed backends are
        invoked.  Individual backend failures are caught; the
        ``partial_results`` flag is set in the returned
        :class:`ContextPackage` metadata.

        A 2-second timeout budget applies to the full fan-out phase
        (Req 5.6).  Backends that exceed their share are cancelled.
        """
        backends = self._resolve_backends(request)
        tasks: dict[str, asyncio.Task[list[str]]] = {}

        if self.BACKEND_SEARCH in backends:
            tasks[self.BACKEND_SEARCH] = asyncio.create_task(
                self._dispatch_search(request)
            )
        if self.BACKEND_GRAPH in backends:
            tasks[self.BACKEND_GRAPH] = asyncio.create_task(
                self._dispatch_graph(request)
            )
        if self.BACKEND_AST in backends:
            tasks[self.BACKEND_AST] = asyncio.create_task(
                self._dispatch_ast(request)
            )

        if not tasks:
            return self._empty_package(request, backends_used=[])

        # Await all tasks with a shared timeout budget.
        ranked_lists: list[list[str]] = []
        partial = False
        backends_used: list[str] = []

        try:
            done, pending = await asyncio.wait(
                tasks.values(),
                timeout=_TIMEOUT_SECONDS,
            )

            # Cancel anything still running after the timeout.
            for task in pending:
                task.cancel()
            # Suppress CancelledError from cancelled tasks.
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                partial = True

            # Map task objects back to backend names.
            task_to_name = {t: name for name, t in tasks.items()}
            for task in done:
                name = task_to_name[task]
                exc = task.exception()
                if exc is not None:
                    logger.warning("Backend %r failed: %s", name, exc)
                    partial = True
                else:
                    result = task.result()
                    if result:
                        ranked_lists.append(result)
                    backends_used.append(name)

        except Exception:
            logger.exception("Unexpected error during fan-out")
            partial = True

        # Fuse
        fused_pairs = self._fuser.fuse(ranked_lists, k=request.rrf_k)
        fused_ids = [item_id for item_id, _score in fused_pairs]

        # Assemble
        try:
            package = await self._assembler.assemble(fused_ids, request)
        except Exception:
            logger.exception("Context assembly failed")
            package = self._empty_package(request, backends_used)
            partial = True

        # Patch metadata
        package.metadata.partial_results = partial
        package.metadata.backends_used = backends_used
        return package

    # ------------------------------------------------------------------
    # Backend dispatchers
    # ------------------------------------------------------------------

    async def _dispatch_search(self, request: QueryRequest) -> list[str]:
        """Dispatch to SearchService and return ranked chunk/symbol IDs."""
        try:
            results = await self._search.search(query=request.query)
            return [r.chunk_id for r in results]
        except Exception:
            logger.exception("Search backend failed")
            raise

    async def _dispatch_graph(self, request: QueryRequest) -> list[str]:
        """Dispatch to GraphStore and return ranked symbol IDs."""
        try:
            store = self._graph_store
            if store is None:
                return []

            direction = "callees" if request.query_type != "symbol" else "callees"
            nodes = await asyncio.to_thread(
                store.get_neighbors,
                request.query,
                direction,
                depth=request.depth,
            )
            return [n.symbol_id for n in nodes]
        except Exception:
            logger.exception("Graph backend failed")
            raise

    async def _dispatch_ast(self, request: QueryRequest) -> list[str]:
        """Dispatch to AST parser for structural symbol lookup."""
        try:
            store = self._graph_store
            if store is None:
                return []

            # Use the symbol index for AST-based lookup.
            entry = await asyncio.to_thread(
                store.lookup_symbol, request.query
            )
            if entry is not None:
                return [entry.fqn]

            # Fall back to short-name lookup.
            entries = await asyncio.to_thread(
                store.lookup_symbols_by_name, request.query
            )
            return [e.fqn for e in entries]
        except Exception:
            logger.exception("AST backend failed")
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_backends(self, request: QueryRequest) -> set[str]:
        """Determine which backends to invoke for *request*."""
        if request.backends is not None:
            requested = set(request.backends) & self._ALL_BACKENDS
        else:
            requested = set(self._ALL_BACKENDS)

        # Always drop graph when disabled.
        if not self._graph_enabled:
            requested.discard(self.BACKEND_GRAPH)

        return requested

    @staticmethod
    def _empty_package(
        request: QueryRequest,
        backends_used: list[str],
    ) -> ContextPackage:
        """Return a minimal empty :class:`ContextPackage`."""
        return ContextPackage(
            query=request.query,
            metadata=ContextMetadata(
                query_params={
                    "query": request.query,
                    "query_type": request.query_type,
                    "depth": request.depth,
                    "max_tokens": request.max_tokens,
                },
                backends_used=backends_used,
            ),
        )

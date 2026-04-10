"""
MCP Context module for dependency injection.

Provides MCPContext dataclass that encapsulates all services needed by MCP handlers,
replacing the Service Locator pattern with explicit dependency injection.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from aci.core.config import ACIConfig
from aci.core.path_utils import RuntimePathMapping, parse_runtime_path_mappings
from aci.infrastructure.embedding import EmbeddingClientInterface
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.infrastructure.vector_store import VectorStoreInterface
from aci.services import IndexingService, SearchService
from aci.services.search_types import RerankerInterface

if TYPE_CHECKING:
    from aci.core.graph_store import GraphStoreInterface
    from aci.services.context_assembler import ContextAssembler
    from aci.services.llm_enricher import LLMEnricher
    from aci.services.query_router import QueryRouter


@dataclass
class MCPContext:
    """
    Container for all services needed by MCP handlers.

    This context is created once at MCP server startup and passed to all handlers,
    enabling explicit dependency injection instead of global service lookup.

    Attributes:
        config: Application configuration
        search_service: Service for semantic code search
        indexing_service: Service for codebase indexing
        metadata_store: SQLite store for index metadata
        vector_store: Vector database for similarity search
        indexing_lock: Lock to prevent concurrent indexing operations
        reranker: Optional reranker for cleanup (not used by handlers directly)
        embedding_client: Embedding client for cleanup (not used by handlers directly)
        graph_store: Optional graph store for code-relationship graphs
        query_router: Optional unified query router
        context_assembler: Optional context assembler for structured context
    """

    config: ACIConfig
    search_service: SearchService
    indexing_service: IndexingService
    metadata_store: IndexMetadataStore
    vector_store: VectorStoreInterface
    indexing_lock: asyncio.Lock
    indexing_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    workspace_root: Path | None = None
    path_mappings: tuple[RuntimePathMapping, ...] = field(default_factory=tuple)
    # These are stored for cleanup purposes only
    reranker: RerankerInterface | None = None
    embedding_client: EmbeddingClientInterface | None = None
    # Graph and semantic intelligence components
    graph_store: GraphStoreInterface | None = None
    query_router: QueryRouter | None = None
    context_assembler: ContextAssembler | None = None
    # Stored for cleanup
    llm_enricher: LLMEnricher | None = None


def create_mcp_context() -> MCPContext:
    """
    Create MCPContext with all services initialized.

    Uses the centralized create_services() factory to initialize infrastructure,
    then constructs SearchService and IndexingService with proper dependency injection.
    Wires graph_store, query_router, and context_assembler from ServicesContainer.

    Returns:
        MCPContext with all services initialized and ready for use.

    Raises:
        ValueError: If required configuration is missing (e.g., API key).
        ConnectionError: If unable to connect to required services.
    """
    from aci.infrastructure.grep_searcher import GrepSearcher
    from aci.services.container import create_services

    workspace_root_env = (
        os.environ.get("ACI_MCP_WORKSPACE_ROOT") or os.environ.get("ACI_WORKSPACE_ROOT")
    )
    raw_path_mappings = (
        os.environ.get("ACI_MCP_PATH_MAPPINGS") or os.environ.get("ACI_PATH_MAPPINGS")
    )
    workspace_root = Path(workspace_root_env).resolve() if workspace_root_env else None
    path_mappings = tuple(parse_runtime_path_mappings(raw_path_mappings))

    # Create infrastructure services using centralized factory
    services = create_services()

    # Create GrepSearcher with base path from current directory
    grep_searcher = GrepSearcher(base_path=str(Path.cwd()))

    # Create SearchService with injected dependencies (including context_assembler)
    search_service = SearchService(
        embedding_client=services.embedding_client,
        vector_store=services.vector_store,
        reranker=services.reranker,
        grep_searcher=grep_searcher,
        context_assembler=services.context_assembler,
        default_limit=services.config.search.default_limit,
    )

    # Create IndexingService with injected dependencies (including graph_builder)
    indexing_service = IndexingService(
        embedding_client=services.embedding_client,
        vector_store=services.vector_store,
        metadata_store=services.metadata_store,
        file_scanner=services.file_scanner,
        chunker=services.chunker,
        batch_size=services.config.embedding.batch_size,
        max_workers=services.config.indexing.max_workers,
        graph_builder=services.graph_builder,
    )

    # Build QueryRouter now that we have a SearchService
    query_router: QueryRouter | None = None
    if services.context_assembler is not None and services.rrf_fuser is not None:
        from aci.core.ast_parser import TreeSitterParser
        from aci.services.query_router import QueryRouter as _QueryRouter

        query_router = _QueryRouter(
            search_service=search_service,
            graph_store=services.graph_store,
            ast_parser=TreeSitterParser(),
            context_assembler=services.context_assembler,
            rrf_fuser=services.rrf_fuser,
            graph_enabled=services.config.graph.enabled,
        )

    return MCPContext(
        config=services.config,
        search_service=search_service,
        indexing_service=indexing_service,
        metadata_store=services.metadata_store,
        vector_store=services.vector_store,
        indexing_lock=asyncio.Lock(),
        workspace_root=workspace_root,
        path_mappings=path_mappings,
        reranker=services.reranker,
        embedding_client=services.embedding_client,
        graph_store=services.graph_store,
        query_router=query_router,
        context_assembler=services.context_assembler,
        llm_enricher=services.llm_enricher,
    )


async def cleanup_context(ctx: MCPContext | None) -> None:
    """
    Clean up MCPContext and close all service connections.

    Handles both sync and async close methods gracefully. Errors during
    cleanup are logged but not re-raised to ensure all resources are
    attempted to be cleaned up.

    Args:
        ctx: The MCPContext to clean up, or None (no-op if None).
    """
    if ctx is None:
        return

    # Close vector store connection
    close_fn = getattr(ctx.vector_store, "close", None)
    if close_fn:
        try:
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass  # Log errors but don't re-raise during cleanup

    # Close reranker connection
    if ctx.reranker:
        aclose_fn = getattr(ctx.reranker, "aclose", None)
        if aclose_fn:
            try:
                result = aclose_fn()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    # Close embedding client connection
    if ctx.embedding_client:
        close_fn = getattr(ctx.embedding_client, "close", None)
        if close_fn:
            try:
                result = close_fn()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    # Close metadata store
    if ctx.metadata_store:
        try:
            ctx.metadata_store.close()
        except Exception:
            pass

    # Close graph store
    if ctx.graph_store:
        close_fn = getattr(ctx.graph_store, "close", None)
        if close_fn:
            try:
                result = close_fn()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    # Close LLM enricher (async)
    if ctx.llm_enricher:
        close_fn = getattr(ctx.llm_enricher, "close", None)
        if close_fn:
            try:
                result = close_fn()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

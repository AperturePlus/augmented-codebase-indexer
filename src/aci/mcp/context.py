"""
MCP Context module for dependency injection.

Provides MCPContext dataclass that encapsulates all services needed by MCP handlers,
replacing the Service Locator pattern with explicit dependency injection.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aci.core.config import ACIConfig
from aci.infrastructure.embedding import EmbeddingClientInterface
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.infrastructure.vector_store import VectorStoreInterface
from aci.services import IndexingService, SearchService
from aci.services.search_types import RerankerInterface


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
    """

    config: ACIConfig
    search_service: SearchService
    indexing_service: IndexingService
    metadata_store: IndexMetadataStore
    vector_store: VectorStoreInterface
    indexing_lock: asyncio.Lock
    indexing_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    # These are stored for cleanup purposes only
    reranker: Optional[RerankerInterface] = None
    embedding_client: Optional[EmbeddingClientInterface] = None


def create_mcp_context() -> MCPContext:
    """
    Create MCPContext with all services initialized.

    Uses the centralized create_services() factory to initialize infrastructure,
    then constructs SearchService and IndexingService with proper dependency injection.

    Returns:
        MCPContext with all services initialized and ready for use.

    Raises:
        ValueError: If required configuration is missing (e.g., API key).
        ConnectionError: If unable to connect to required services.
    """
    from aci.infrastructure.grep_searcher import GrepSearcher
    from aci.services.container import create_services

    # Create infrastructure services using centralized factory
    services = create_services()

    # Create GrepSearcher with base path from current directory
    grep_searcher = GrepSearcher(base_path=str(Path.cwd()))

    # Create SearchService with injected dependencies
    search_service = SearchService(
        embedding_client=services.embedding_client,
        vector_store=services.vector_store,
        reranker=services.reranker,
        grep_searcher=grep_searcher,
        default_limit=services.config.search.default_limit,
    )

    # Create IndexingService with injected dependencies
    indexing_service = IndexingService(
        embedding_client=services.embedding_client,
        vector_store=services.vector_store,
        metadata_store=services.metadata_store,
        file_scanner=services.file_scanner,
        chunker=services.chunker,
        batch_size=services.config.embedding.batch_size,
        max_workers=services.config.indexing.max_workers,
    )

    return MCPContext(
        config=services.config,
        search_service=search_service,
        indexing_service=indexing_service,
        metadata_store=services.metadata_store,
        vector_store=services.vector_store,
        indexing_lock=asyncio.Lock(),
        reranker=services.reranker,
        embedding_client=services.embedding_client,
    )


async def cleanup_context(ctx: Optional[MCPContext]) -> None:
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

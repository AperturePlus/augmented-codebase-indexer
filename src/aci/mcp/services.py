"""
Service initialization and caching for MCP server.

Manages process-level service instances to avoid creating new connections
on each tool call.
"""

import asyncio
from pathlib import Path
from typing import Optional

from aci.cli import get_services
from aci.infrastructure.grep_searcher import GrepSearcher
from aci.services import IndexingService, SearchService


# Process-level service cache to avoid creating new connections on each call
_services_cache: Optional[tuple] = None

# Lock to prevent concurrent indexing operations from corrupting shared state
_indexing_lock = asyncio.Lock()

# Maximum allowed workers (matches HTTP API limit)
MAX_WORKERS = 32


def get_initialized_services():
    """
    Get initialized services for ACI operations.
    
    Services are cached at process level to avoid creating new connections
    (AsyncQdrantClient, httpx.AsyncClient) on each tool call.
    
    Returns:
        Tuple of (cfg, search_service, indexing_service, metadata_store, vector_store)
    """
    global _services_cache
    
    if _services_cache is not None:
        return _services_cache[:5]  # Return original 5-tuple for compatibility
    
    (
        cfg,
        embedding_client,
        vector_store,
        metadata_store,
        file_scanner,
        chunker,
        reranker,
    ) = get_services()

    # Create GrepSearcher with base path from current directory
    grep_searcher = GrepSearcher(base_path=str(Path.cwd()))

    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=reranker,
        grep_searcher=grep_searcher,
        default_limit=cfg.search.default_limit,
    )

    indexing_service = IndexingService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        chunker=chunker,
        batch_size=cfg.embedding.batch_size,
        max_workers=cfg.indexing.max_workers,
    )

    _services_cache = (
        cfg,
        search_service,
        indexing_service,
        metadata_store,
        vector_store,
        reranker,
        embedding_client,
    )
    return _services_cache[:5]


async def cleanup_services():
    """Clean up cached services and close connections."""
    global _services_cache
    
    if _services_cache is None:
        return
    
    (
        cfg,
        search_service,
        indexing_service,
        metadata_store,
        vector_store,
        reranker,
        embedding_client,
    ) = _services_cache
    
    # Close vector store connection
    close_fn = getattr(vector_store, "close", None)
    if close_fn:
        result = close_fn()
        if asyncio.iscoroutine(result):
            await result
    
    # Close reranker connection
    if reranker:
        aclose_fn = getattr(reranker, "aclose", None)
        if aclose_fn:
            result = aclose_fn()
            if asyncio.iscoroutine(result):
                await result
    
    # Close embedding client connection (method is 'close', not 'aclose')
    close_fn = getattr(embedding_client, "close", None)
    if close_fn:
        result = close_fn()
        if asyncio.iscoroutine(result):
            await result
    
    # Close metadata store
    if metadata_store:
        metadata_store.close()
    
    _services_cache = None


def get_indexing_lock() -> asyncio.Lock:
    """Get the indexing lock for preventing concurrent indexing operations."""
    return _indexing_lock

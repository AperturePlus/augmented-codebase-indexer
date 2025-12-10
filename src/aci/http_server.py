"""
HTTP layer for Project ACI.

Provides a lightweight FastAPI server to expose indexing and search endpoints.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from aci.cli import get_services
from aci.core.path_utils import is_system_directory
from aci.infrastructure.grep_searcher import GrepSearcher
from aci.infrastructure.vector_store import SearchResult
from aci.services import IndexingService, SearchMode, SearchService

logger = logging.getLogger(__name__)

# Lock to prevent concurrent indexing operations from corrupting shared state
_indexing_lock = asyncio.Lock()


class IndexRequest(BaseModel):
    path: str
    workers: Optional[int] = None


class SearchResponseItem(BaseModel):
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float
    metadata: dict


def _to_response_item(result: SearchResult) -> SearchResponseItem:
    """Convert SearchResult to API response model."""
    return SearchResponseItem(
        chunk_id=result.chunk_id,
        file_path=result.file_path,
        start_line=result.start_line,
        end_line=result.end_line,
        content=result.content,
        score=result.score,
        metadata=result.metadata,
    )


def create_app() -> FastAPI:
    """FastAPI application factory (config sourced from .env)."""
    (
        cfg,
        embedding_client,
        vector_store,
        metadata_store,
        file_scanner,
        chunker,
        reranker,
    ) = get_services()

    # Create GrepSearcher with base path from config or current directory
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

    app = FastAPI(
        title="Augmented Codebase Indexer",
        version="0.1.0",
        description="HTTP interface for semantic code search and indexing.",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/status")
    async def status(path: Optional[str] = None):
        try:
            metadata_stats = metadata_store.get_stats()

            # If path provided, get collection-specific stats
            collection_name = None
            if path:
                status_path = Path(path)
                if status_path.exists() and status_path.is_dir():
                    status_path_abs = str(status_path.resolve())
                    index_info = metadata_store.get_index_info(status_path_abs)
                    if index_info:
                        collection_name = index_info.get("collection_name")
                        if not collection_name:
                            from aci.core.path_utils import get_collection_name_for_path
                            collection_name = get_collection_name_for_path(status_path_abs)

            vector_stats = await vector_store.get_stats(collection_name=collection_name)
            return {
                "metadata": metadata_stats,
                "vector_store": vector_stats,
                "embedding_model": cfg.embedding.model,
                "collection_name": collection_name,
            }
        except Exception as exc:
            logger.error(f"Error in /status: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal Server Error")

    @app.post("/index")
    async def index(req: IndexRequest):
        try:
            # Security: Validate path
            target_path = Path(req.path).resolve()
            if not target_path.exists():
                raise HTTPException(status_code=400, detail="Path does not exist")
            if not target_path.is_dir():
                raise HTTPException(status_code=400, detail="Path is not a directory")

            # Security: Block sensitive system directories (platform-aware)
            if is_system_directory(target_path):
                raise HTTPException(status_code=403, detail="Indexing system directories is forbidden")

            # Security: Cap workers
            max_allowed_workers = 32
            requested_workers = req.workers if req.workers is not None else cfg.indexing.max_workers
            workers = min(requested_workers, max_allowed_workers)

            # Use lock to prevent concurrent indexing operations
            async with _indexing_lock:
                indexing_service._max_workers = workers
                result = await indexing_service.index_directory(target_path)
                return result.__dict__
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error in /index: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal Server Error")

    @app.post("/update")
    async def update(req: IndexRequest):
        try:
            # Security: Validate path
            target_path = Path(req.path).resolve()
            if not target_path.exists():
                raise HTTPException(status_code=400, detail="Path does not exist")
            if not target_path.is_dir():
                raise HTTPException(status_code=400, detail="Path is not a directory")

            # Security: Block sensitive system directories (platform-aware)
            if is_system_directory(target_path):
                raise HTTPException(status_code=403, detail="Indexing system directories is forbidden")

            # Security: Cap workers
            max_allowed_workers = 32
            requested_workers = req.workers if req.workers is not None else cfg.indexing.max_workers
            workers = min(requested_workers, max_allowed_workers)

            # Use lock to prevent concurrent indexing operations
            async with _indexing_lock:
                indexing_service._max_workers = workers
                result = await indexing_service.update_incremental(target_path)
                return result.__dict__
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error in /update: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal Server Error")

    @app.get("/search")
    async def search(
        q: str,
        path: str,
        limit: Optional[int] = None,
        file_filter: Optional[str] = None,
        use_rerank: Optional[bool] = None,
        mode: Optional[str] = None,
    ):
        try:
            # Validate and resolve path
            search_path = Path(path)
            if not search_path.exists():
                raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
            if not search_path.is_dir():
                raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")

            # Check if path is indexed and get collection name
            search_path_abs = str(search_path.resolve())
            index_info = metadata_store.get_index_info(search_path_abs)
            if index_info is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Path has not been indexed: {path}. Run /index first.",
                )

            # Get collection name for this codebase
            # For backward compatibility, generate collection name if not stored
            collection_name = index_info.get("collection_name")
            if not collection_name:
                from aci.core.path_utils import get_collection_name_for_path
                collection_name = get_collection_name_for_path(search_path_abs)
                metadata_store.register_repository(search_path_abs, collection_name)

            apply_rerank = cfg.search.use_rerank if use_rerank is None else use_rerank

            # Parse search mode (default to hybrid)
            search_mode = SearchMode.HYBRID
            if mode:
                mode_lower = mode.lower()
                if mode_lower == "vector":
                    search_mode = SearchMode.VECTOR
                elif mode_lower == "grep":
                    search_mode = SearchMode.GREP
                elif mode_lower == "hybrid":
                    search_mode = SearchMode.HYBRID

            # Pass collection_name explicitly to avoid shared state mutation
            results = await search_service.search(
                query=q,
                limit=limit,
                file_filter=file_filter,
                use_rerank=apply_rerank,
                search_mode=search_mode,
                collection_name=collection_name,
            )
            return {"results": [_to_response_item(r) for r in results]}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error in /search: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal Server Error")

    @app.on_event("shutdown")
    async def shutdown_event():
        # Close vector store if supported
        close = getattr(vector_store, "close", None)
        if close:
            maybe_coro = close()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

        # Close reranker if supported
        if reranker:
            aclose = getattr(reranker, "aclose", None)
            if aclose:
                maybe_coro = aclose()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro

        # Close embedding client (method is 'close', not 'aclose')
        close_embed = getattr(embedding_client, "close", None)
        if close_embed:
            maybe_coro = close_embed()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

        # Close metadata store
        if metadata_store:
            close_meta = getattr(metadata_store, "close", None)
            if close_meta:
                close_meta()

    return app

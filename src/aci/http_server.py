"""
HTTP layer for Project ACI.

Provides a lightweight FastAPI server to expose indexing and search endpoints.
"""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from aci.cli import get_services
from aci.infrastructure.vector_store import SearchResult
from aci.services import IndexingService, SearchService


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

    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=reranker,
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
    async def status():
        try:
            metadata_stats = metadata_store.get_stats()
            vector_stats = await vector_store.get_stats()
            return {
                "metadata": metadata_stats,
                "vector_store": vector_stats,
                "embedding_model": cfg.embedding.model,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/index")
    async def index(req: IndexRequest):
        try:
            workers = req.workers if req.workers is not None else cfg.indexing.max_workers
            indexing_service._max_workers = workers  # reuse service instance safely
            result = await indexing_service.index_directory(Path(req.path))
            return result.__dict__
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/update")
    async def update(req: IndexRequest):
        try:
            workers = req.workers if req.workers is not None else cfg.indexing.max_workers
            indexing_service._max_workers = workers
            result = await indexing_service.update_incremental(Path(req.path))
            return result.__dict__
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/search")
    async def search(
        q: str,
        limit: Optional[int] = None,
        file_filter: Optional[str] = None,
        use_rerank: Optional[bool] = None,
    ):
        try:
            apply_rerank = cfg.search.use_rerank if use_rerank is None else use_rerank
            results = await search_service.search(
                query=q,
                limit=limit,
                file_filter=file_filter,
                use_rerank=apply_rerank,
            )
            return {"results": [_to_response_item(r) for r in results]}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.on_event("shutdown")
    async def shutdown_event():
        # Close vector store if supported
        close = getattr(vector_store, "close", None)
        if close:
            maybe_coro = close()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        if reranker:
            aclose = getattr(reranker, "aclose", None)
            if aclose:
                maybe_coro = aclose()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro

    return app

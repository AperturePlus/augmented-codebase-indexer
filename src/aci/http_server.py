"""
HTTP layer for Project ACI.

Provides a lightweight FastAPI server to expose indexing and search endpoints.
"""

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from aci.core.path_utils import get_collection_name_for_path, is_system_directory
from aci.core.watch_config import WatchConfig
from aci.infrastructure.codebase_registry import best_effort_update_registry
from aci.infrastructure.file_watcher import FileWatcher
from aci.infrastructure.grep_searcher import GrepSearcher
from aci.infrastructure.vector_store import SearchResult
from aci.services import IndexingService, SearchMode, SearchService, WatchService
from aci.services.container import create_services
from aci.services.metrics_collector import MetricsCollector
from aci.services.repository_resolver import resolve_repository

logger = logging.getLogger(__name__)

# Lock to prevent concurrent indexing operations from corrupting shared state
_indexing_lock = asyncio.Lock()


class IndexRequest(BaseModel):
    path: str
    workers: int | None = None


class WatchRequest(BaseModel):
    """Request model for starting file watch."""

    path: str
    debounce_ms: int | None = None
    ignore_patterns: list[str] | None = None
    verbose: bool | None = None


class WatchStatusResponse(BaseModel):
    """Response model for watch service status."""

    running: bool
    watch_path: str | None = None
    debounce_ms: int | None = None
    started_at: str | None = None
    events_received: int = 0
    updates_triggered: int = 0
    last_update_at: str | None = None
    last_update_duration_ms: float = 0.0
    errors: int = 0
    pending_events: int = 0


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


def create_app(
    watch_path: str | Path | None = None,
    watch_debounce_ms: int = 2000,
) -> FastAPI:
    """
    FastAPI application factory (config sourced from .env).

    Args:
        watch_path: Optional path to watch for file changes. If provided,
                   the watch service will start automatically with the HTTP server.
        watch_debounce_ms: Debounce delay in milliseconds for file watching.
    """
    services = create_services()
    cfg = services.config
    embedding_client = services.embedding_client
    vector_store = services.vector_store
    metadata_store = services.metadata_store
    file_scanner = services.file_scanner
    chunker = services.chunker
    reranker = services.reranker

    # Create GrepSearcher with base path from config or current directory
    grep_searcher = GrepSearcher(base_path=str(Path.cwd()))

    # Create shared MetricsCollector instance
    metrics_collector = MetricsCollector()

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
        metrics_collector=metrics_collector,
    )

    # Watch service state (initialized lazily when /watch/start is called or via watch_path)
    watch_service_state: dict = {
        "service": None,
        "config": cfg,
        "initial_watch_path": Path(watch_path).resolve() if watch_path else None,
        "initial_debounce_ms": watch_debounce_ms,
    }

    app = FastAPI(
        title="Augmented Codebase Indexer",
        version="0.1.0",
        description="HTTP interface for semantic code search and indexing.",
    )

    @app.on_event("startup")
    async def startup_event():
        """Start watch service if watch_path was provided."""
        initial_path = watch_service_state.get("initial_watch_path")
        if initial_path is not None:
            try:
                # Create watch configuration
                watch_config = WatchConfig(
                    watch_path=initial_path,
                    debounce_ms=watch_service_state["initial_debounce_ms"],
                    ignore_patterns=[],
                    verbose=False,
                )

                # Create file watcher with config-driven extensions and ignore patterns
                file_watcher = FileWatcher(
                    extensions=set(cfg.indexing.file_extensions),
                    ignore_patterns=list(cfg.indexing.ignore_patterns),
                )

                # Create watch service with metrics collector
                watch_service = WatchService(
                    indexing_service=indexing_service,
                    file_watcher=file_watcher,
                    config=watch_config,
                    metrics_collector=metrics_collector,
                )

                # Start the watch service
                await watch_service.start()
                watch_service_state["service"] = watch_service

                logger.info(f"Watch service started for: {initial_path}")
            except Exception as e:
                logger.error(f"Failed to start watch service: {e}")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        """Return current operational metrics in JSON format."""
        return metrics_collector.get_metrics()

    @app.get("/status")
    async def status(path: str | None = None):
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

            # Get staleness information
            stale_files = metadata_store.get_stale_files(limit=5)
            all_stale_files = metadata_store.get_stale_files()
            stale_file_count = len(all_stale_files)

            # Format stale files sample with staleness in hours
            stale_files_sample = [
                {"path": file_path, "staleness_hours": round(staleness_seconds / 3600, 2)}
                for file_path, staleness_seconds in stale_files
            ]

            return {
                "metadata": metadata_stats,
                "vector_store": vector_stats,
                "embedding_model": cfg.embedding.model,
                "collection_name": collection_name,
                "vector_count": vector_stats.get("total_vectors", 0),
                "file_count": metadata_stats.get("total_files", 0),
                "staleness": {
                    "stale_file_count": stale_file_count,
                    "stale_files_sample": stale_files_sample,
                },
                "stale_file_count": stale_file_count,
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
                best_effort_update_registry(
                    root_path=target_path,
                    metadata_db_path=metadata_store.db_path,
                    collection_name=get_collection_name_for_path(target_path),
                )
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
                best_effort_update_registry(
                    root_path=target_path,
                    metadata_db_path=metadata_store.db_path,
                    collection_name=get_collection_name_for_path(target_path),
                )
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
        limit: int | None = None,
        file_filter: str | None = None,
        use_rerank: bool | None = None,
        mode: str | None = None,
        artifact_type: list[str] | None = Query(None),
    ):
        try:
            # Use centralized repository resolution
            resolution = resolve_repository(path, metadata_store)
            if not resolution.valid:
                raise HTTPException(status_code=400, detail=resolution.error_message)

            collection_name = resolution.collection_name

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
                elif mode_lower == "summary":
                    search_mode = SearchMode.SUMMARY

            # Validate artifact types if provided
            valid_artifact_types = ["chunk", "function_summary", "class_summary", "file_summary"]
            if artifact_type:
                invalid_types = [t for t in artifact_type if t not in valid_artifact_types]
                if invalid_types:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid artifact type(s): {invalid_types}. "
                               f"Valid types are: {valid_artifact_types}",
                    )

            # Pass collection_name explicitly to avoid shared state mutation
            results = await search_service.search(
                query=q,
                limit=limit,
                file_filter=file_filter,
                use_rerank=apply_rerank,
                search_mode=search_mode,
                collection_name=collection_name,
                artifact_types=artifact_type,
            )
            return {"results": [_to_response_item(r) for r in results]}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error in /search: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal Server Error")

    @app.post("/watch/start")
    async def watch_start(req: WatchRequest):
        """Start watching a directory for file changes."""
        try:
            # Check if already watching
            if watch_service_state["service"] is not None:
                current_service: WatchService = watch_service_state["service"]
                if current_service.is_running():
                    raise HTTPException(
                        status_code=409,
                        detail="Watch service is already running. Stop it first.",
                    )

            # Validate path
            target_path = Path(req.path).resolve()
            if not target_path.exists():
                raise HTTPException(status_code=400, detail="Path does not exist")
            if not target_path.is_dir():
                raise HTTPException(status_code=400, detail="Path is not a directory")

            # Security: Block sensitive system directories
            if is_system_directory(target_path):
                raise HTTPException(
                    status_code=403, detail="Watching system directories is forbidden"
                )

            # Create watch configuration
            watch_config = WatchConfig(
                watch_path=target_path,
                debounce_ms=req.debounce_ms or 2000,
                ignore_patterns=req.ignore_patterns or [],
                verbose=req.verbose or False,
            )

            # Create file watcher with config-driven extensions and ignore patterns
            combined_ignore_patterns = list(cfg.indexing.ignore_patterns)
            if req.ignore_patterns:
                combined_ignore_patterns.extend(req.ignore_patterns)

            file_watcher = FileWatcher(
                extensions=set(cfg.indexing.file_extensions),
                ignore_patterns=combined_ignore_patterns,
            )

            # Create watch service with metrics collector
            watch_service = WatchService(
                indexing_service=indexing_service,
                file_watcher=file_watcher,
                config=watch_config,
                metrics_collector=metrics_collector,
            )

            # Start the watch service
            await watch_service.start()
            watch_service_state["service"] = watch_service

            logger.info(f"Watch service started for: {target_path}")

            return {
                "status": "started",
                "watch_path": str(target_path),
                "debounce_ms": watch_config.debounce_ms,
            }

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error in /watch/start: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/watch/stop")
    async def watch_stop():
        """Stop the current file watch."""
        try:
            watch_service: WatchService | None = watch_service_state["service"]

            if watch_service is None or not watch_service.is_running():
                raise HTTPException(
                    status_code=400, detail="Watch service is not running"
                )

            # Get stats before stopping
            stats = watch_service.get_stats()

            # Stop the watch service
            await watch_service.stop()
            watch_service_state["service"] = None

            logger.info("Watch service stopped")

            return {
                "status": "stopped",
                "events_received": stats.events_received,
                "updates_triggered": stats.updates_triggered,
                "errors": stats.errors,
            }

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Error in /watch/stop: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/watch/status", response_model=WatchStatusResponse)
    async def watch_status():
        """Get the current watch service status."""
        try:
            watch_service: WatchService | None = watch_service_state["service"]

            if watch_service is None or not watch_service.is_running():
                return WatchStatusResponse(running=False)

            stats = watch_service.get_stats()
            config = watch_service.config

            return WatchStatusResponse(
                running=True,
                watch_path=str(config.watch_path),
                debounce_ms=config.debounce_ms,
                started_at=stats.started_at.isoformat(),
                events_received=stats.events_received,
                updates_triggered=stats.updates_triggered,
                last_update_at=(
                    stats.last_update_at.isoformat() if stats.last_update_at else None
                ),
                last_update_duration_ms=stats.last_update_duration_ms,
                errors=stats.errors,
                pending_events=watch_service.get_pending_count(),
            )

        except Exception as exc:
            logger.error(f"Error in /watch/status: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.on_event("shutdown")
    async def shutdown_event():
        # Stop watch service if running
        watch_service: WatchService | None = watch_service_state["service"]
        if watch_service is not None and watch_service.is_running():
            try:
                await watch_service.stop()
                watch_service_state["service"] = None
                logger.info("Watch service stopped during shutdown")
            except Exception as e:
                logger.error(f"Error stopping watch service during shutdown: {e}")

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

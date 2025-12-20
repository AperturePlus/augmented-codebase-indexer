"""MCP tool handlers for ACI."""
import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mcp.types import TextContent

from aci.core.path_utils import get_collection_name_for_path, validate_indexable_path
from aci.infrastructure.codebase_registry import best_effort_update_registry
from aci.mcp.context import MCPContext
from aci.mcp.services import MAX_WORKERS
from aci.services import SearchMode
from aci.services.repository_resolver import resolve_repository

# Handler type: takes arguments dict and MCPContext, returns list of TextContent
_HANDLERS: dict[str, Callable[[dict, MCPContext], Awaitable[list[TextContent]]]] = {}


def _is_debug() -> bool:
    """Check if debug mode is enabled (reads env each time for flexibility)."""
    return os.environ.get("ACI_ENV", "production").lower() == "development"


def _debug(msg: str):
    """Print debug message to stderr if in development mode."""
    if _is_debug():
        pass


def _register(name: str):
    def decorator(fn):
        _HANDLERS[name] = fn
        return fn
    return decorator


def _get_repo_index_lock(ctx: MCPContext, repo_key: str) -> asyncio.Lock:
    lock = ctx.indexing_locks.get(repo_key)
    if lock is None:
        lock = asyncio.Lock()
        ctx.indexing_locks[repo_key] = lock
    return lock


async def call_tool(name: str, arguments: Any, ctx: MCPContext) -> list[TextContent]:
    """
    Handle tool calls from MCP clients.

    Args:
        name: The tool name to invoke.
        arguments: Tool arguments as a dictionary.
        ctx: MCPContext containing all required services.

    Returns:
        List of TextContent with the tool result.
    """
    if ctx is None:
        return [TextContent(type="text", text="Error: MCPContext not initialized")]
    try:
        handler = _HANDLERS.get(name)
        if handler:
            return await handler(arguments, ctx)
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]


@_register("index_codebase")
async def _handle_index_codebase(arguments: dict, ctx: MCPContext) -> list[TextContent]:
    path_str = arguments["path"]
    workers = arguments.get("workers")
    start_time = time.time()
    _debug(f"index_codebase called with path: {path_str}, workers: {workers}")

    # Validate path before any indexing operation
    validation = validate_indexable_path(path_str)
    if not validation.valid:
        _debug(f"Path validation failed: {validation.error_message}")
        return [TextContent(
            type="text",
            text=f"Error: {validation.error_message} (path: {path_str})"
        )]

    path = Path(path_str)
    _debug(f"Resolved path: {path.resolve()}")

    cfg = ctx.config
    indexing_service = ctx.indexing_service
    _debug(f"Services initialized, embedding_url={cfg.embedding.api_url}, model={cfg.embedding.model}")
    _debug(f"API key present: {bool(cfg.embedding.api_key)}")

    # Respect user input/config while keeping a hard cap (matches HTTP API limit).
    if workers is None:
        workers = cfg.indexing.max_workers
    else:
        try:
            workers = int(workers)
        except (TypeError, ValueError):
            return [TextContent(type="text", text="Error: 'workers' must be an integer")]
        if workers < 1:
            return [TextContent(type="text", text="Error: 'workers' must be >= 1")]

    if workers < 1:
        workers = 1
    if workers > MAX_WORKERS:
        _debug(f"Requested workers ({workers}) exceeds MAX_WORKERS ({MAX_WORKERS}); capping.")
        workers = MAX_WORKERS

    _debug(f"Using {workers} workers")

    repo_key = str(path.resolve())
    repo_lock = _get_repo_index_lock(ctx, repo_key)

    _debug("Acquiring indexing lock...")
    async with repo_lock:
        _debug("Lock acquired, starting indexing...")
        result = await indexing_service.index_directory(path, max_workers=workers)
        _debug(f"Indexing completed in {time.time() - start_time:.2f}s")

    best_effort_update_registry(
        root_path=path,
        metadata_db_path=ctx.metadata_store.db_path,
        collection_name=get_collection_name_for_path(path),
    )

    response = {"status": "success", "total_files": result.total_files,
        "total_chunks": result.total_chunks, "duration_seconds": result.duration_seconds,
        "failed_files": result.failed_files[:10] if result.failed_files else []}
    if result.failed_files and len(result.failed_files) > 10:
        response["failed_files_truncated"] = len(result.failed_files) - 10
    _debug(f"Result: files={result.total_files}, chunks={result.total_chunks}")
    return [TextContent(type="text", text=json.dumps(response, indent=2))]


@_register("search_code")
async def _handle_search_code(arguments: dict, ctx: MCPContext) -> list[TextContent]:
    query = arguments["query"]
    search_path = Path(arguments["path"])
    limit = arguments.get("limit")
    file_filter = arguments.get("file_filter")
    use_rerank = arguments.get("use_rerank")
    mode = arguments.get("mode")
    artifact_types = arguments.get("artifact_types")

    cfg = ctx.config
    search_service = ctx.search_service
    metadata_store = ctx.metadata_store

    # Use centralized repository resolution
    resolution = resolve_repository(search_path, metadata_store)
    if not resolution.valid:
        return [TextContent(type="text", text=f"Error: {resolution.error_message}")]

    collection_name = resolution.collection_name
    indexed_root = resolution.indexed_root

    # If searching a subdirectory, build a path prefix filter
    search_path_abs = str(search_path.resolve())
    path_prefix_filter = None
    if search_path_abs != indexed_root:
        # Normalize to forward slashes for consistent matching
        path_prefix_filter = search_path_abs.replace("\\", "/")

    # Use config defaults if not specified
    if limit is None:
        limit = cfg.search.default_limit
    if use_rerank is None:
        use_rerank = cfg.search.use_rerank

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

    # Validate artifact_types if provided
    valid_artifact_types = {"chunk", "function_summary", "class_summary", "file_summary"}
    if artifact_types:
        invalid_types = [t for t in artifact_types if t not in valid_artifact_types]
        if invalid_types:
            return [TextContent(
                type="text",
                text=f"Error: Invalid artifact type(s): {', '.join(invalid_types)}. "
                     f"Valid types: {', '.join(sorted(valid_artifact_types))}"
            )]

    # Request more results if filtering by subdirectory (to ensure enough after filtering)
    fetch_limit = limit * 3 if path_prefix_filter else limit

    # Pass collection_name explicitly to avoid shared state mutation
    results = await search_service.search(
        query=query,
        limit=fetch_limit,
        file_filter=file_filter,
        use_rerank=use_rerank,
        search_mode=search_mode,
        collection_name=collection_name,
        artifact_types=artifact_types,
    )

    # Filter results to subdirectory if searching in a subdirectory
    if path_prefix_filter and results:
        results = [
            r for r in results
            if r.file_path.replace("\\", "/").startswith(path_prefix_filter)
        ][:limit]

    if not results:
        return [TextContent(type="text", text="No results found.")]

    # Format results for LLM consumption
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append({
            "rank": i,
            "file_path": result.file_path,
            "start_line": result.start_line,
            "end_line": result.end_line,
            "score": round(result.score, 4),
            "language": result.metadata.get("language", "unknown"),
            "artifact_type": result.metadata.get("artifact_type", "chunk"),
            "content": result.content,
        })

    response = {
        "query": query,
        "total_results": len(results),
        "results": formatted_results,
    }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


@_register("get_index_status")
async def _handle_get_status(arguments: dict, ctx: MCPContext) -> list[TextContent]:
    cfg = ctx.config
    metadata_store = ctx.metadata_store
    vector_store = ctx.vector_store

    # Check if a specific path was requested
    path_str = arguments.get("path")
    collection_name = None

    if path_str:
        # Get collection name for the specific repository
        search_path = Path(path_str)
        if not search_path.exists():
            return [TextContent(type="text", text=f"Error: Path does not exist: {search_path}")]
        if not search_path.is_dir():
            return [TextContent(type="text", text=f"Error: Path is not a directory: {search_path}")]

        search_path_abs = str(search_path.resolve())
        index_info = metadata_store.get_index_info(search_path_abs)
        if index_info is None:
            return [TextContent(type="text", text=f"Error: Path has not been indexed: {search_path}")]

        # Get collection name for this repository
        collection_name = index_info.get("collection_name")
        if not collection_name:
            from aci.core.path_utils import get_collection_name_for_path
            collection_name = get_collection_name_for_path(search_path_abs)

    # Get metadata stats
    stats = metadata_store.get_stats()

    # Get vector store stats for the specified collection (or default)
    try:
        vector_stats = await vector_store.get_stats(collection_name=collection_name)
        vector_count = vector_stats.get("total_vectors", 0)
        vector_status = "connected"
    except Exception as e:
        vector_count = 0
        vector_status = f"error: {str(e)}"

    response = {
        "metadata": {
            "total_files": stats["total_files"],
            "total_chunks": stats["total_chunks"],
            "total_lines": stats["total_lines"],
            "languages": stats["languages"],
        },
        "vector_store": {
            "status": vector_status,
            "total_vectors": vector_count,
        },
        "configuration": {
            "embedding_model": cfg.embedding.model,
            "embedding_dimension": cfg.embedding.dimension,
            "max_chunk_tokens": cfg.indexing.max_chunk_tokens,
            "rerank_enabled": cfg.search.use_rerank,
            "file_extensions": list(cfg.indexing.file_extensions),
        },
    }

    # Add repository info if a specific path was requested
    if path_str:
        response["repository"] = {
            "path": path_str,
            "collection_name": collection_name,
        }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


@_register("update_index")
async def _handle_update_index(arguments: dict, ctx: MCPContext) -> list[TextContent]:
    path_str = arguments["path"]
    start_time = time.time()
    _debug(f"update_index called with path: {path_str}")

    # Validate path before any indexing operation
    validation = validate_indexable_path(path_str)
    if not validation.valid:
        _debug(f"Path validation failed: {validation.error_message}")
        return [TextContent(
            type="text",
            text=f"Error: {validation.error_message} (path: {path_str})"
        )]

    path = Path(path_str)
    _debug(f"Resolved path: {path.resolve()}")

    indexing_service = ctx.indexing_service
    metadata_store = ctx.metadata_store
    _debug("Services initialized")

    # Check if path is indexed
    abs_path = str(path.resolve())
    index_info = metadata_store.get_index_info(abs_path)
    if index_info is None:
        _debug(f"Path not indexed: {path}")
        return [TextContent(
            type="text",
            text=f"Error: Path has not been indexed: {path}. Run index_codebase first."
        )]
    _debug(f"Index info: {index_info}")

    # Check if we have file hashes (required for incremental update)
    existing_hashes = metadata_store.get_all_file_hashes()
    _debug(f"Existing file hashes count: {len(existing_hashes)}")

    if len(existing_hashes) == 0:
        _debug("No file hashes found - index metadata is incomplete")
        return [TextContent(
            type="text",
            text=(
                f"Error: Index metadata is incomplete for {path}. "
                "File hashes are missing. Please run index_codebase again to rebuild the index."
            )
        )]

    repo_key = str(path.resolve())
    repo_lock = _get_repo_index_lock(ctx, repo_key)

    _debug("Acquiring indexing lock...")
    async with repo_lock:
        _debug("Lock acquired, starting incremental update...")
        result = await indexing_service.update_incremental(path)
        _debug(f"Update completed in {time.time() - start_time:.2f}s")

        best_effort_update_registry(
            root_path=path,
            metadata_db_path=metadata_store.db_path,
            collection_name=get_collection_name_for_path(path),
        )

        response = {
            "status": "success",
            "new_files": result.new_files,
            "modified_files": result.modified_files,
            "deleted_files": result.deleted_files,
            "duration_seconds": result.duration_seconds,
        }
        _debug(f"Result: {response}")

        return [TextContent(type="text", text=json.dumps(response, indent=2))]


@_register("list_indexed_repos")
async def _handle_list_repos(arguments: dict, ctx: MCPContext) -> list[TextContent]:
    metadata_store = ctx.metadata_store

    repos = metadata_store.get_repositories()

    if not repos:
        return [TextContent(type="text", text="No repositories indexed.")]

    formatted_repos = [
        {
            "root_path": repo["root_path"],
            "last_updated": str(repo["updated_at"]),
        }
        for repo in repos
    ]

    response = {
        "total_repositories": len(repos),
        "repositories": formatted_repos,
    }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]

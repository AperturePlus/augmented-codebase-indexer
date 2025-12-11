"""MCP tool handlers for ACI."""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from mcp.types import TextContent
from aci.core.path_utils import validate_indexable_path
from aci.mcp.services import get_initialized_services, get_indexing_lock, MAX_WORKERS
from aci.services import SearchMode

_HANDLERS = {}


def _is_debug() -> bool:
    """Check if debug mode is enabled (reads env each time for flexibility)."""
    return os.environ.get("ACI_ENV", "production").lower() == "development"


def _debug(msg: str):
    """Print debug message to stderr if in development mode."""
    if _is_debug():
        print(f"[ACI-DEBUG] {msg}", file=sys.stderr, flush=True)

def _register(name: str):
    def decorator(fn):
        _HANDLERS[name] = fn
        return fn
    return decorator

async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls from MCP clients."""
    try:
        handler = _HANDLERS.get(name)
        if handler:
            return await handler(arguments)
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]


@_register("index_codebase")
async def _handle_index_codebase(arguments: dict) -> list[TextContent]:
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
    
    cfg, _, indexing_service, _, _ = get_initialized_services()
    _debug(f"Services initialized, embedding_url={cfg.embedding.api_url}, model={cfg.embedding.model}")
    #_debug(f"API key present: {bool(cfg.embedding.api_key)}, key prefix: {cfg.embedding.api_key[:8] if cfg.embedding.api_key else 'NONE'}...")

    # IMPORTANT: Force single-threaded mode for MCP
    # ProcessPoolExecutor conflicts with MCP's stdio event loop, causing hangs.
    # Sequential processing is slower but reliable in stdio context.
    workers = 1
    _debug(f"Using {workers} workers (forced single-threaded for MCP stdio compatibility)")

    # Use lock to prevent concurrent indexing operations
    indexing_lock = get_indexing_lock()
    _debug("Acquiring indexing lock...")
    async with indexing_lock:
        _debug("Lock acquired, starting indexing...")
        indexing_service._max_workers = workers
        result = await indexing_service.index_directory(path)
        _debug(f"Indexing completed in {time.time() - start_time:.2f}s")

    response = {"status": "success", "total_files": result.total_files,
        "total_chunks": result.total_chunks, "duration_seconds": result.duration_seconds,
        "failed_files": result.failed_files[:10] if result.failed_files else []}
    if result.failed_files and len(result.failed_files) > 10:
        response["failed_files_truncated"] = len(result.failed_files) - 10
    _debug(f"Result: files={result.total_files}, chunks={result.total_chunks}")
    return [TextContent(type="text", text=json.dumps(response, indent=2))]


@_register("search_code")
async def _handle_search_code(arguments: dict) -> list[TextContent]:
    query = arguments["query"]
    search_path = Path(arguments["path"])
    limit = arguments.get("limit")
    file_filter = arguments.get("file_filter")
    use_rerank = arguments.get("use_rerank")
    mode = arguments.get("mode")

    cfg, search_service, _, metadata_store, _ = get_initialized_services()

    # Validate and resolve path
    if not search_path.exists():
        return [TextContent(type="text", text=f"Error: Path does not exist: {search_path}")]
    if not search_path.is_dir():
        return [TextContent(type="text", text=f"Error: Path is not a directory: {search_path}")]

    # Check if path is indexed and get collection name
    search_path_abs = str(search_path.resolve())
    index_info = metadata_store.get_index_info(search_path_abs)
    if index_info is None:
        return [TextContent(type="text", text=f"Error: Path has not been indexed: {search_path}. Run index_codebase first.")]

    # Get collection name for this codebase (no shared state mutation)
    # For backward compatibility, generate collection name if not stored
    collection_name = index_info.get("collection_name")
    if not collection_name:
        from aci.core.path_utils import get_collection_name_for_path
        collection_name = get_collection_name_for_path(search_path_abs)
        metadata_store.register_repository(search_path_abs, collection_name)

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

    # Pass collection_name explicitly to avoid shared state mutation
    results = await search_service.search(
        query=query,
        limit=limit,
        file_filter=file_filter,
        use_rerank=use_rerank,
        search_mode=search_mode,
        collection_name=collection_name,
    )

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
            "content": result.content,
        })

    response = {
        "query": query,
        "total_results": len(results),
        "results": formatted_results,
    }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


@_register("get_index_status")
async def _handle_get_status(arguments: dict) -> list[TextContent]:
    cfg, _, _, metadata_store, vector_store = get_initialized_services()

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
async def _handle_update_index(arguments: dict) -> list[TextContent]:
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
    
    cfg, _, indexing_service, metadata_store, _ = get_initialized_services()
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

    # Use lock to prevent concurrent indexing operations
    indexing_lock = get_indexing_lock()
    _debug("Acquiring indexing lock...")
    async with indexing_lock:
        _debug("Lock acquired, starting incremental update...")
        result = await indexing_service.update_incremental(path)
        _debug(f"Update completed in {time.time() - start_time:.2f}s")

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
async def _handle_list_repos(arguments: dict) -> list[TextContent]:
    _, _, _, metadata_store, _ = get_initialized_services()

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

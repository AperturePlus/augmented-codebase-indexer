"""
MCP Server for Project ACI.

Provides Model Context Protocol interface for semantic code search and indexing.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from aci.cli import get_services
from aci.infrastructure.grep_searcher import GrepSearcher
from aci.services import IndexingService, SearchMode, SearchService


# Initialize MCP server
app = Server("aci-mcp-server")

# Process-level service cache to avoid creating new connections on each call
_services_cache: Optional[tuple] = None


def _get_initialized_services():
    """
    Get initialized services for ACI operations.
    
    Services are cached at process level to avoid creating new connections
    (AsyncQdrantClient, httpx.AsyncClient) on each tool call.
    """
    global _services_cache
    
    if _services_cache is not None:
        return _services_cache
    
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
    return _services_cache[:5]  # Return original 5-tuple for compatibility


async def _cleanup_services():
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
    
    # Close embedding client connection
    aclose_fn = getattr(embedding_client, "aclose", None)
    if aclose_fn:
        result = aclose_fn()
        if asyncio.iscoroutine(result):
            await result
    
    # Close metadata store
    if metadata_store:
        metadata_store.close()
    
    _services_cache = None


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="index_codebase",
            description="Index a codebase directory for semantic search. This will scan all supported files and create embeddings for code chunks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the directory to index",
                    },
                    "workers": {
                        "type": "integer",
                        "description": "Number of parallel workers (optional, defaults to config)",
                        "minimum": 1,
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="search_code",
            description="Search the indexed codebase using semantic search, keyword search, or both. Returns relevant code chunks with file paths and line numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing the code you're looking for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (optional, defaults to config)",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "file_filter": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.py' or 'src/**/*.js')",
                    },
                    "use_rerank": {
                        "type": "boolean",
                        "description": "Whether to use reranking for better results (optional, defaults to config)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "vector", "grep"],
                        "description": "Search mode: 'hybrid' (default) combines semantic and keyword search, 'vector' for semantic only, 'grep' for keyword only",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_index_status",
            description="Get the current status and statistics of the indexed codebase, including total files, chunks, languages, and health information.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="update_index",
            description="Incrementally update the index by detecting new, modified, or deleted files since the last indexing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the directory to update",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="list_indexed_repos",
            description="List all indexed repositories with their root paths and last update times.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls from MCP clients."""
    try:
        if name == "index_codebase":
            return await _handle_index_codebase(arguments)
        elif name == "search_code":
            return await _handle_search_code(arguments)
        elif name == "get_index_status":
            return await _handle_get_status(arguments)
        elif name == "update_index":
            return await _handle_update_index(arguments)
        elif name == "list_indexed_repos":
            return await _handle_list_repos(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        return [TextContent(type="text", text=error_msg)]


async def _handle_index_codebase(arguments: dict) -> list[TextContent]:
    """Handle index_codebase tool call."""
    path = Path(arguments["path"])
    workers = arguments.get("workers")

    if not path.exists():
        return [TextContent(type="text", text=f"Error: Path does not exist: {path}")]

    if not path.is_dir():
        return [TextContent(type="text", text=f"Error: Path is not a directory: {path}")]

    cfg, _, indexing_service, _, _ = _get_initialized_services()

    if workers is not None:
        indexing_service._max_workers = workers

    result = await indexing_service.index_directory(path)

    response = {
        "status": "success",
        "total_files": result.total_files,
        "total_chunks": result.total_chunks,
        "duration_seconds": result.duration_seconds,
        "failed_files": result.failed_files[:10] if result.failed_files else [],
    }

    if result.failed_files and len(result.failed_files) > 10:
        response["failed_files_truncated"] = len(result.failed_files) - 10

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def _handle_search_code(arguments: dict) -> list[TextContent]:
    """Handle search_code tool call."""
    query = arguments["query"]
    limit = arguments.get("limit")
    file_filter = arguments.get("file_filter")
    use_rerank = arguments.get("use_rerank")
    mode = arguments.get("mode")

    cfg, search_service, _, _, _ = _get_initialized_services()

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

    results = await search_service.search(
        query=query,
        limit=limit,
        file_filter=file_filter,
        use_rerank=use_rerank,
        search_mode=search_mode,
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


async def _handle_get_status(arguments: dict) -> list[TextContent]:
    """Handle get_index_status tool call."""
    cfg, _, _, metadata_store, vector_store = _get_initialized_services()

    # Get metadata stats
    stats = metadata_store.get_stats()

    # Get vector store stats
    try:
        vector_stats = await vector_store.get_stats()
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

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def _handle_update_index(arguments: dict) -> list[TextContent]:
    """Handle update_index tool call."""
    path = Path(arguments["path"])

    if not path.exists():
        return [TextContent(type="text", text=f"Error: Path does not exist: {path}")]

    if not path.is_dir():
        return [TextContent(type="text", text=f"Error: Path is not a directory: {path}")]

    _, _, indexing_service, _, _ = _get_initialized_services()

    result = await indexing_service.update_incremental(path)

    response = {
        "status": "success",
        "new_files": result.new_files,
        "modified_files": result.modified_files,
        "deleted_files": result.deleted_files,
        "duration_seconds": result.duration_seconds,
    }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def _handle_list_repos(arguments: dict) -> list[TextContent]:
    """Handle list_indexed_repos tool call."""
    _, _, _, metadata_store, _ = _get_initialized_services()

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


async def main():
    """Run the MCP server."""
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        # Clean up connections on shutdown
        await _cleanup_services()


if __name__ == "__main__":
    asyncio.run(main())

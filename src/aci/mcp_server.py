"""
MCP Server for Project ACI.

Provides Model Context Protocol interface for semantic code search and indexing.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from aci.cli import get_services
from aci.services import IndexingService, SearchService


# Initialize MCP server
app = Server("aci-mcp-server")


def _get_initialized_services():
    """Get initialized services for ACI operations."""
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

    return cfg, search_service, indexing_service, metadata_store, vector_store


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
            description="Search the indexed codebase using semantic search. Returns relevant code chunks with file paths and line numbers.",
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

    cfg, search_service, _, _, _ = _get_initialized_services()

    # Use config defaults if not specified
    if limit is None:
        limit = cfg.search.default_limit
    if use_rerank is None:
        use_rerank = cfg.search.use_rerank

    results = await search_service.search(
        query=query,
        limit=limit,
        file_filter=file_filter,
        use_rerank=use_rerank,
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
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())

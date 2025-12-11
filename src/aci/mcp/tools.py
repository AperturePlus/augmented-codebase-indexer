"""
MCP tool definitions for ACI.

Defines the available tools and their schemas for the MCP interface.
"""

from mcp.types import Tool


def list_tools() -> list[Tool]:
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
            description="Search the indexed codebase using semantic search, keyword search, or both. Returns relevant code chunks and summaries with file paths and line numbers. Supports filtering by artifact type to search at different granularity levels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing the code you're looking for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the indexed codebase to search (required for isolation between codebases)",
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
                    "artifact_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["chunk", "function_summary", "class_summary", "file_summary"],
                        },
                        "description": "Filter results by artifact type. Options: 'chunk' (code chunks), 'function_summary' (function descriptions), 'class_summary' (class descriptions), 'file_summary' (file overviews). If not specified, returns all types.",
                    },
                },
                "required": ["query", "path"],
            },
        ),
        Tool(
            name="get_index_status",
            description="Get the current status and statistics of the indexed codebase, including total files, chunks, languages, and health information. Optionally specify a path to get stats for a specific repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path to a specific indexed repository to get stats for. If not provided, returns aggregate stats.",
                    },
                },
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

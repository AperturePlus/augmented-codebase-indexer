# Project ACI - Augmented Codebase Indexer

A Python tool for semantic code search with precise line-level location results.

## Features

- Semantic code search using embeddings
- Precise line-level location results
- Support for Python, JavaScript/TypeScript, and Go
- Tree-sitter based AST parsing
- Qdrant vector database integration
- Incremental indexing

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Index a codebase
aci index /path/to/codebase

# Search for code
aci search "function that handles authentication"

# Check index status
aci status

# Update index incrementally
aci update

# Reset index (drop collection & metadata)
aci reset

# Start HTTP server (FastAPI)
aci serve --host 0.0.0.0 --port 8000

# Also available via python -m entrypoint
uv run python -m aci serve  # when using uv
```

## Configuration

Use a `.env` file (required). The app auto-loads `.env`; YAML configs are disabled:

```
ACI_EMBEDDING_API_URL=https://api.openai.com/v1/embeddings
ACI_EMBEDDING_API_KEY=your_embedding_api_key   # required
ACI_EMBEDDING_MODEL=text-embedding-3-small
ACI_EMBEDDING_DIMENSION=1024                  # must match vector size below
ACI_VECTOR_STORE_HOST=localhost
ACI_VECTOR_STORE_PORT=6333
ACI_VECTOR_STORE_VECTOR_SIZE=1024
ACI_INDEXING_MAX_WORKERS=4

# Optional reranker
ACI_SEARCH_USE_RERANK=false
ACI_SEARCH_RERANK_API_URL=https://your-rerank-endpoint.example.com
ACI_SEARCH_RERANK_API_KEY=your_rerank_api_key
ACI_SEARCH_RERANK_MODEL=bge-reranker-large
ACI_SEARCH_RERANK_TIMEOUT=30
ACI_SEARCH_RERANK_ENDPOINT=/v1/rerank  # override if provider uses a different path

# Optional: overrides for chunking/ignores (comma-separated)
# ACI_INDEXING_FILE_EXTENSIONS=.py,.js,.ts,.go
# ACI_INDEXING_IGNORE_PATTERNS=__pycache__,*.pyc,.git,node_modules
```

All settings must come from `.env`/environment variables; YAML/JSON configs are not used.

The CLI and HTTP server will attempt to auto-start a local Qdrant Docker container on port `6333`
if one is not already running.

## License

MIT

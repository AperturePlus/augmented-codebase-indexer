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
```

## Configuration

Create a `config.yaml` file or use environment variables:

```yaml
embedding:
  api_key: "your-api-key"
  model: "text-embedding-3-small"

vector_store:
  host: "localhost"
  port: 6333

indexing:
  file_extensions: [".py", ".js", ".ts", ".go"]
  max_workers: 4
```

Environment variables override config file settings:
- `ACI_EMBEDDING_API_KEY`
- `ACI_VECTOR_STORE_HOST`
- `ACI_LOGGING_LEVEL`

## License

MIT

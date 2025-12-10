# Project ACI - Augmented Codebase Indexer

A Python tool for semantic code search with precise line-level location results.

## Features

- Semantic code search using embeddings
- Precise line-level location results
- Support for Python, JavaScript/TypeScript, and Go
- Tree-sitter based AST parsing
- Qdrant vector database integration
- Incremental indexing
- **MCP (Model Context Protocol) interface for LLM integration**

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

# Search with file path filter
aci search "parse config path:*.py"

# Search excluding certain paths
aci search "database connection -path:tests"

# Check index status
aci status

# Update index incrementally
aci update

# Reset index (drop collection & metadata)
aci reset

# Start interactive shell mode
aci shell

# Start HTTP server (FastAPI)
aci serve --host 0.0.0.0 --port 8000

# Also available via python -m entrypoint
uv run python -m aci serve  # when using uv

# Start MCP server (for LLM integration)
aci-mcp
# or
uv run aci-mcp
```

## Interactive Shell Mode

ACI provides an interactive shell mode that allows you to execute multiple commands without restarting the program each time. This is especially useful for iterative workflows like indexing, searching, and refining queries.

### Starting the Shell

```bash
aci shell
```

This launches an interactive REPL (Read-Eval-Print Loop) with:
- Command history (up/down arrows to navigate)
- Tab completion for commands
- Persistent history across sessions

### Available Commands

| Command | Description |
|---------|-------------|
| `index <path>` | Index a directory for semantic search |
| `search <query>` | Search the indexed codebase (supports modifiers) |
| `status` | Show index status and statistics |
| `update <path>` | Incrementally update the index |
| `list` | List all indexed repositories |
| `reset` | Clear the index (requires confirmation) |
| `help` or `?` | Display available commands |
| `exit`, `quit`, or `q` | Exit the shell |

### Example Session

```
$ aci shell

    _    ____ ___   ____  _          _ _ 
   / \  / ___|_ _| / ___|| |__   ___| | |
  / _ \| |    | |  \___ \| '_ \ / _ \ | |
 / ___ \ |___ | |   ___) | | | |  __/ | |
/_/   \_\____|___| |____/|_| |_|\___|_|_|

Welcome to ACI Interactive Shell
Type 'help' for available commands, 'exit' to quit

aci> index ./src
Indexing ./src...
✓ Indexed 42 files, 156 chunks

aci> search "authentication handler"
Found 3 results:
...

aci> search "config parser path:src/*.py -path:tests"
Found 2 results:
...

aci> exit
Goodbye!
```

## Search Query Modifiers

Search queries support inline modifiers to filter results:

| Modifier | Description | Example |
|----------|-------------|---------|
| `path:<pattern>` | Include only files matching pattern | `path:*.py`, `path:src/**` |
| `file:<pattern>` | Alias for `path:` | `file:handlers.py` |
| `-path:<pattern>` | Exclude files matching pattern | `-path:tests` |
| `exclude:<pattern>` | Alias for `-path:` | `exclude:fixtures` |

Multiple exclusions can be combined:
```bash
aci search "database query -path:tests -path:fixtures"
```

## MCP Integration

ACI now supports the Model Context Protocol (MCP), allowing large language models to directly interact with your codebase indexing and search capabilities.

### Quick Start with MCP

1. Install ACI with dependencies:
```bash
pip install -e ".[dev]"
```

2. Configure your MCP client (e.g., Kiro, Claude Desktop) to use ACI:
```json
{
  "mcpServers": {
    "aci": {
      "command": "uv",
      "args": ["run", "aci-mcp"],
      "env": {
        "ACI_EMBEDDING_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

3. Start using natural language to interact with your codebase:
   - "Index the /path/to/project directory"
   - "Search for authentication functions"
   - "Show me the index status"

For detailed MCP setup and usage, see [MCP_USAGE.md](./MCP_USAGE.md).

### Available MCP Tools

- `index_codebase` - Index a directory for semantic search
- `search_code` - Search code using natural language queries
- `get_index_status` - Get indexing statistics and health info
- `update_index` - Incrementally update the index
- `list_indexed_repos` - List all indexed repositories
```

## Security

ACI includes built-in security protections:

- **System directory protection**: Indexing system directories (`/etc`, `/var`, `C:\Windows`, etc.) is blocked across all interfaces (CLI, HTTP, MCP)
- **Sensitive file denylist**: The following files are automatically excluded from indexing regardless of configuration:
  - SSH keys and directories (`.ssh`, `id_rsa`, `id_ed25519`, etc.)
  - GPG directories (`.gnupg`)
  - Certificates and private keys (`*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.crt`)
  - Environment files (`.env`, `.env.*`)
  - Credential files (`.netrc`, `.npmrc`, `.pypirc`)

These protections cannot be overridden by user configuration.

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

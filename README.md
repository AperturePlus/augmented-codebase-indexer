# Project ACI - Augmented Codebase Indexer

A Python tool for semantic code search with precise line-level location results.

## Features

- Semantic code search using embeddings (OpenAI-compatible API)
- Precise line-level location results
- Support for Python, JavaScript/TypeScript, Go, Java, C, C++
- Tree-sitter based AST parsing for accurate code chunking
- Hybrid search (semantic + keyword/grep)
- Qdrant vector database integration
- Incremental indexing for efficient updates
- Multiple interfaces: CLI, HTTP API, MCP (for LLM integration)
- Auto-detection of local timezone for timestamps

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e ".[dev]"
```

## Requirements

- Python 3.10+
- Qdrant (auto-started via Docker if not running)
- OpenAI-compatible embedding API (OpenAI, SiliconFlow, etc.)

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

ACI supports the Model Context Protocol (MCP), allowing LLMs to directly interact with your codebase indexing and search capabilities.

### Quick Start with MCP

1. Configure your MCP client (e.g., Kiro, Claude Desktop, Cursor):

```json
{
  "mcpServers": {
    "aci": {
      "command": "uv",
      "args": ["run", "aci-mcp"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

2. Ensure `.env` exists in the working directory with required settings (see `.env.example`)

3. Use natural language to interact with your codebase:
   - "Index the current directory"
   - "Search for authentication functions"
   - "Show me the index status"

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `index_codebase` | Index a directory for semantic search |
| `search_code` | Search code using natural language queries |
| `get_index_status` | Get indexing statistics and health info |
| `update_index` | Incrementally update the index |
| `list_indexed_repos` | List all indexed repositories |

### Testing MCP

```bash
# Test with MCP Inspector (Web UI)
npx @modelcontextprotocol/inspector uv run aci-mcp

# Test via Python script
uv run python tests/test_mcp_call/test_stdio.py

# Test indexing
uv run python tests/test_mcp_call/test_index_codebase.py
```

### Debug Mode

Set `ACI_ENV=development` in `.env` to enable debug logging:
```
ACI_ENV=development
```

Debug messages are printed to stderr and visible in MCP Inspector's notifications.

> **Note**: MCP uses single-threaded indexing for stdio compatibility. For faster indexing of large codebases, use the CLI: `uv run aci index .`

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

Configuration is done via `.env` file or environment variables. Copy `.env.example` to `.env` and fill in your settings:

```bash
cp .env.example .env
```

Key settings:
| Variable | Description | Required |
|----------|-------------|----------|
| `ACI_EMBEDDING_API_KEY` | API key for embedding service | Yes |
| `ACI_EMBEDDING_API_URL` | Embedding API endpoint | No (defaults to OpenAI) |
| `ACI_EMBEDDING_MODEL` | Model name | No |
| `ACI_VECTOR_STORE_HOST` | Qdrant host | No (defaults to localhost) |
| `ACI_VECTOR_STORE_PORT` | Qdrant port | No (defaults to 6333) |
| `ACI_SERVER_HOST` | HTTP server host | No (defaults to 0.0.0.0) |
| `ACI_SERVER_PORT` | HTTP server port | No (defaults to 8000) |
| `ACI_ENV` | Environment (development/production) | No |

See `.env.example` for the full list of options.

The CLI and HTTP server will attempt to auto-start a local Qdrant Docker container on port `6333`
if one is not already running.



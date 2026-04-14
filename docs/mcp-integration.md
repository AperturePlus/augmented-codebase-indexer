# MCP Integration

ACI supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), letting LLMs directly drive codebase indexing and search.

## Quick Start

Start the MCP server:

```bash
aci-mcp
# or
uv run aci-mcp
```

Configure your MCP client (Kiro, Claude Desktop, Cursor, etc.):

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

Ensure `.env` exists in the working directory (see `.env.example`), then use natural language:

- "Index the current directory"
- "Search for authentication functions"
- "Show me the index status"

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `index_codebase` | Index a directory for semantic search |
| `search_code` | Search code using natural language queries |
| `get_index_status` | Get indexing statistics and health info |
| `update_index` | Incrementally update the index |
| `list_indexed_repos` | List all indexed repositories |

## Docker Sidecar Deployment

The recommended production model for agentic coding tools:

- Code repository stays on the user's machine
- MCP server runs in a local container
- Qdrant runs as another local container or as a cloud endpoint
- Embedding API uses the user's own API key

```bash
# Build the image
docker build -t aci-mcp:latest .

# Start a local Qdrant container (if not using cloud)
docker compose -f docker/qdrant/docker-compose.yaml up -d
```

Configure your MCP client to launch ACI through Docker. A complete template is in `mcp-config.docker.example.json`.

### Runtime Rules

- Mount the host source tree **read-only** into the container (e.g. `/workspace`)
- Persist `/data` as a Docker volume so `.aci/index.db` survives restarts
- Set `ACI_MCP_WORKSPACE_ROOT` for relative path resolution
- Set `ACI_MCP_PATH_MAPPINGS` when the client sends host-native absolute paths

```text
ACI_MCP_WORKSPACE_ROOT=/workspace
ACI_MCP_PATH_MAPPINGS=D:\repo=/workspace
ACI_MCP_PATH_MAPPINGS=/Users/alice/repo=/workspace
```

With path mappings configured, MCP tools accept host paths and resolve them to container paths automatically.

## Testing MCP

```bash
# Interactive Web UI via MCP Inspector
npx @modelcontextprotocol/inspector uv run aci-mcp

# Scripted tests
uv run python tests/test_mcp_call/test_stdio.py
uv run python tests/test_mcp_call/test_index_codebase.py
```

## Debug Mode

Set `ACI_ENV=development` in `.env` to enable debug logging to stderr (visible in MCP Inspector notifications):

```
ACI_ENV=development
```

> **Note:** MCP uses single-threaded indexing for stdio compatibility. For large codebases, prefer the CLI: `aci index .`

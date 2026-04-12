# ACI — Augmented Codebase Indexer

[![Tests](https://github.com/AperturePlus/augmented-codebase-indexer/actions/workflows/test.yml/badge.svg)](https://github.com/AperturePlus/augmented-codebase-indexer/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![linux.do](https://img.shields.io/badge/linux.do-%E6%8E%A8%E5%B9%BF%E9%93%BE%E6%8E%A5-orange)](https://linux.do)

Language: **English** | [简体中文](doc/README.zh-CN.md)

---

> **Ask your codebase a question. Get a precise answer — down to the line.**

ACI indexes your code with embeddings and Tree-sitter AST parsing, then lets you search it with natural language. Results come back with exact file paths and line numbers, not just fuzzy matches.

```bash
$ aci search "function that validates JWT tokens"

src/auth/middleware.py:42  verify_token(token: str) -> Claims
src/auth/utils.py:118      decode_and_validate(raw: str) -> dict
```

---

## Why ACI?

Most code search tools give you grep or a fuzzy filename match. ACI gives you **semantic understanding**:

- You describe intent, it finds the implementation
- Hybrid search combines embeddings with keyword/grep for precision
- Multi-level indexing: raw chunks, function summaries, class summaries, file summaries
- Incremental updates — only re-indexes what changed
- Works with Python, JavaScript/TypeScript, Go, Java, C, C++

---

## Get Started

```bash
# Install
uv sync

# Configure (add your embedding API key)
cp .env.example .env

# Index your codebase
aci index /path/to/your/project

# Search
aci search "error handling in the HTTP layer"
```

That's it. See [Installation](docs/installation.md) for full setup details.

---

## Interfaces

| Interface | Command | Use case |
|-----------|---------|----------|
| CLI | `aci <command>` | Day-to-day search and indexing |
| Interactive shell | `aci shell` | Iterative exploration sessions |
| HTTP API | `aci serve` | Integrate with other tools |
| MCP server | `aci-mcp` | LLM / agent integration |

---

## Documentation

- [Installation & Configuration](docs/installation.md)
- [CLI Usage & Search Syntax](docs/cli-usage.md)
- [MCP Integration](docs/mcp-integration.md)
- [Security](docs/security.md)
- [Chunking Algorithm](doc/CHUNKING_ALGORITHM.zh-CN.md)

---

## MCP — Let Your LLM Search the Code

ACI ships a first-class MCP server so agents can index and search your codebase directly.

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

For Docker-based deployment (recommended for agentic tools), see [MCP Integration](docs/mcp-integration.md).

---

## Requirements

- Python 3.10+
- Qdrant (auto-started locally via Docker, or point to Qdrant Cloud)
- Any OpenAI-compatible embedding API (OpenAI, SiliconFlow, etc.)

---

Development governance: [AGENTS.md](AGENTS.md)

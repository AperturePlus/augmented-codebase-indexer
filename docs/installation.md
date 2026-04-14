# Installation

## Requirements

- Python 3.10+
- Qdrant (local via Docker auto-start, or cloud via URL + API key)
- OpenAI-compatible embedding API (OpenAI, SiliconFlow, etc.)

## Install

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and fill in your settings:

```bash
cp .env.example .env
```

| Variable | Description | Required |
|----------|-------------|----------|
| `ACI_EMBEDDING_API_KEY` | API key for embedding service | Yes |
| `ACI_EMBEDDING_API_URL` | Embedding API endpoint | No (defaults to OpenAI) |
| `ACI_EMBEDDING_MODEL` | Model name | No |
| `ACI_VECTOR_STORE_URL` | Qdrant base URL (takes precedence over host/port) | No |
| `ACI_VECTOR_STORE_API_KEY` | Qdrant API key (for Qdrant Cloud) | No |
| `ACI_VECTOR_STORE_HOST` | Qdrant host | No (defaults to localhost) |
| `ACI_VECTOR_STORE_PORT` | Qdrant port | No (defaults to 6333) |
| `ACI_SERVER_HOST` | HTTP server host | No (defaults to 0.0.0.0) |
| `ACI_SERVER_PORT` | HTTP server port | No (defaults to 8000) |
| `ACI_ENV` | Environment (`development`/`production`) | No |

See `.env.example` for the full list of options.

## Qdrant Auto-Start

When targeting a local endpoint (`localhost` / `127.0.0.1`) and Qdrant is not reachable, ACI automatically runs:

```bash
docker compose -f docker/qdrant/docker-compose.yaml up -d
```

This starts a named `aci-qdrant` container with a persistent `qdrant_data` volume. You can also start it manually:

```bash
docker compose -f docker/qdrant/docker-compose.yaml up -d
```

For cloud Qdrant (`ACI_VECTOR_STORE_URL`), Docker is not launched.

When ACI itself runs inside a container, it will not attempt to launch nested Docker for Qdrant. In that setup, run Qdrant as a separate container or point `ACI_VECTOR_STORE_URL` to Qdrant Cloud.

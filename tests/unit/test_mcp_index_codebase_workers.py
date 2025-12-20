import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aci.core.config import ACIConfig
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.mcp.context import MCPContext
from aci.mcp.handlers import call_tool
from aci.mcp.services import MAX_WORKERS
from aci.services import SearchService


@dataclass
class DummyIndexingResult:
    total_files: int = 0
    total_chunks: int = 0
    duration_seconds: float = 0.0
    failed_files: list[str] = field(default_factory=list)


class StubIndexingService:
    def __init__(self) -> None:
        self.seen_workers: list[int | None] = []
        self.paths: list[Path] = []

    async def index_directory(
        self, path: Path, *, max_workers: int | None = None
    ) -> DummyIndexingResult:
        self.paths.append(path)
        self.seen_workers.append(max_workers)
        return DummyIndexingResult(total_files=1, total_chunks=2, duration_seconds=0.01)


def _make_ctx(config: ACIConfig, indexing_service: StubIndexingService) -> MCPContext:
    vector_store = InMemoryVectorStore()
    embedding_client = LocalEmbeddingClient()
    metadata_store = IndexMetadataStore(":memory:")
    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=None,
        grep_searcher=None,
        default_limit=config.search.default_limit,
    )
    return MCPContext(
        config=config,
        search_service=search_service,
        indexing_service=indexing_service,  # type: ignore[arg-type]
        metadata_store=metadata_store,
        vector_store=vector_store,
        indexing_lock=asyncio.Lock(),
        embedding_client=embedding_client,
    )


@pytest.mark.asyncio
async def test_mcp_index_codebase_respects_workers_argument(tmp_path: Path):
    indexing_service = StubIndexingService()
    ctx = _make_ctx(ACIConfig(), indexing_service)

    result = await call_tool(
        "index_codebase",
        {"path": str(tmp_path), "workers": 4},
        ctx,
    )

    data = json.loads(result[0].text)
    assert data["status"] == "success"
    assert indexing_service.seen_workers == [4]

    ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_mcp_index_codebase_caps_workers(tmp_path: Path):
    indexing_service = StubIndexingService()
    ctx = _make_ctx(ACIConfig(), indexing_service)

    requested = MAX_WORKERS + 10
    await call_tool(
        "index_codebase",
        {"path": str(tmp_path), "workers": requested},
        ctx,
    )

    assert indexing_service.seen_workers == [MAX_WORKERS]
    ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_mcp_index_codebase_defaults_to_config_workers(tmp_path: Path):
    config = ACIConfig()
    config.indexing.max_workers = 7
    indexing_service = StubIndexingService()
    ctx = _make_ctx(config, indexing_service)

    await call_tool("index_codebase", {"path": str(tmp_path)}, ctx)

    assert indexing_service.seen_workers == [7]
    ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_mcp_index_codebase_invalid_workers_returns_error(tmp_path: Path):
    indexing_service = StubIndexingService()
    ctx = _make_ctx(ACIConfig(), indexing_service)

    result = await call_tool(
        "index_codebase",
        {"path": str(tmp_path), "workers": "nope"},
        ctx,
    )

    assert "Error" in result[0].text
    assert "workers" in result[0].text
    assert indexing_service.seen_workers == []
    ctx.metadata_store.close()

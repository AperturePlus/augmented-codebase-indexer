import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aci.core.config import ACIConfig
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.mcp.context import MCPContext
from aci.mcp.handlers import call_tool
from aci.services import SearchService


@dataclass
class DummyIndexingResult:
    total_files: int = 0
    total_chunks: int = 0
    duration_seconds: float = 0.0
    failed_files: list[str] = field(default_factory=list)


def _make_ctx(indexing_service) -> MCPContext:
    config = ACIConfig()
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
async def test_mcp_index_codebase_allows_concurrent_different_paths(tmp_path: Path):
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()

    started: dict[str, asyncio.Event] = {
        str(repo_a.resolve()): asyncio.Event(),
        str(repo_b.resolve()): asyncio.Event(),
    }
    release = asyncio.Event()

    class BlockingIndexingService:
        async def index_directory(
            self, path: Path, *, max_workers: int | None = None
        ) -> DummyIndexingResult:
            started[str(path.resolve())].set()
            await release.wait()
            return DummyIndexingResult(total_files=1, total_chunks=1, duration_seconds=0.0)

    ctx = _make_ctx(BlockingIndexingService())

    task_a = asyncio.create_task(
        call_tool("index_codebase", {"path": str(repo_a), "workers": 2}, ctx)
    )
    task_b = asyncio.create_task(
        call_tool("index_codebase", {"path": str(repo_b), "workers": 2}, ctx)
    )

    try:
        await asyncio.wait_for(
            asyncio.gather(
                started[str(repo_a.resolve())].wait(),
                started[str(repo_b.resolve())].wait(),
            ),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(task_a, task_b, return_exceptions=True)
        ctx.metadata_store.close()


@pytest.mark.asyncio
async def test_mcp_index_codebase_serializes_same_path(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    started_first = asyncio.Event()
    started_second = asyncio.Event()
    release = asyncio.Event()

    class BlockingIndexingService:
        def __init__(self) -> None:
            self.call_count = 0

        async def index_directory(
            self, path: Path, *, max_workers: int | None = None
        ) -> DummyIndexingResult:
            self.call_count += 1
            if self.call_count == 1:
                started_first.set()
                await release.wait()
            else:
                started_second.set()
            return DummyIndexingResult(total_files=1, total_chunks=1, duration_seconds=0.0)

    ctx = _make_ctx(BlockingIndexingService())

    task_1 = asyncio.create_task(
        call_tool("index_codebase", {"path": str(repo), "workers": 2}, ctx)
    )
    task_2 = asyncio.create_task(
        call_tool("index_codebase", {"path": str(repo), "workers": 2}, ctx)
    )

    try:
        await asyncio.wait_for(started_first.wait(), timeout=1.0)
        assert not started_second.is_set()

        release.set()
        await asyncio.wait_for(started_second.wait(), timeout=1.0)
    finally:
        release.set()
        await asyncio.gather(task_1, task_2, return_exceptions=True)
        ctx.metadata_store.close()


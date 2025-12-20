import asyncio
import tempfile
from pathlib import Path

from aci.core.file_scanner import FileScanner
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.services.indexing_service import IndexingService


def run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_update_incremental_does_not_delete_other_repo_metadata():
    """
    Regression test: incremental updates must be scoped to the target root.

    When a metadata DB contains multiple indexed repositories, calling
    update_incremental(repo_a) must not treat repo_b files as "deleted".
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repo_a = root / "repo_a"
        repo_b = root / "repo_b"
        repo_a.mkdir(parents=True, exist_ok=True)
        repo_b.mkdir(parents=True, exist_ok=True)

        file_a = repo_a / "a.py"
        file_b = repo_b / "b.py"
        file_a.write_text("def a():\n    return 1\n", encoding="utf-8")
        file_b.write_text("def b():\n    return 2\n", encoding="utf-8")

        vector_store = InMemoryVectorStore()
        embedding_client = LocalEmbeddingClient()
        metadata_store = IndexMetadataStore(root / "metadata.db")
        file_scanner = FileScanner(extensions={".py"})

        service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            file_scanner=file_scanner,
            max_workers=1,
        )

        # Index both repositories into the same metadata store.
        run_async(service.index_directory(repo_a))
        run_async(service.index_directory(repo_b))

        file_a_abs = str(file_a.resolve())
        file_b_abs = str(file_b.resolve())
        assert metadata_store.get_file_info(file_a_abs) is not None
        assert metadata_store.get_file_info(file_b_abs) is not None

        # Update only repo_a (no changes) - repo_b metadata must remain intact.
        run_async(service.update_incremental(repo_a))

        assert metadata_store.get_file_info(file_b_abs) is not None

        metadata_store.close()


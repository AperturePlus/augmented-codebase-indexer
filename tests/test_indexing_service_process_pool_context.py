from pathlib import Path

from aci.core.file_scanner import FileScanner
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.services.indexing_service import IndexingService


def _make_service(tmp_path: Path) -> IndexingService:
    return IndexingService(
        embedding_client=LocalEmbeddingClient(),
        vector_store=InMemoryVectorStore(),
        metadata_store=IndexMetadataStore(tmp_path / "index.db"),
        file_scanner=FileScanner(extensions={".py"}),
        max_workers=2,
    )


def test_process_pool_context_is_none_when_single_thread(tmp_path, monkeypatch):
    import aci.services.indexing_service as mod

    monkeypatch.delenv("ACI_PROCESS_START_METHOD", raising=False)
    monkeypatch.setattr(mod.multiprocessing, "get_start_method", lambda *a, **k: "fork")
    monkeypatch.setattr(mod.threading, "active_count", lambda: 1)

    service = _make_service(tmp_path)
    try:
        assert service._get_process_pool_mp_context() is None
    finally:
        service._metadata_store.close()


def test_process_pool_context_prefers_forkserver_when_threads_active(tmp_path, monkeypatch):
    import aci.services.indexing_service as mod

    monkeypatch.delenv("ACI_PROCESS_START_METHOD", raising=False)
    monkeypatch.setattr(mod.multiprocessing, "get_start_method", lambda *a, **k: "fork")
    monkeypatch.setattr(mod.threading, "active_count", lambda: 2)

    calls: list[str] = []

    def fake_get_context(method: str):
        calls.append(method)

        class DummyCtx:
            def get_start_method(self) -> str:
                return method

        return DummyCtx()

    monkeypatch.setattr(mod.multiprocessing, "get_context", fake_get_context)

    service = _make_service(tmp_path)
    try:
        ctx = service._get_process_pool_mp_context()
        assert ctx is not None
        assert ctx.get_start_method() in {"forkserver", "spawn"}
        assert calls and calls[0] == "forkserver"
    finally:
        service._metadata_store.close()


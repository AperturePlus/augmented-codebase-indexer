"""
Property-based tests for Structured Logging in Indexing Observability.

Tests for logging completeness (Properties 3, 4, 5).
"""

import asyncio
import hashlib
import logging
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.core.file_scanner import ScannedFile
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.services.indexing_service import IndexingService


class LogCapture(logging.Handler):
    """Handler that captures log records for testing."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord):
        self.records.append(record)

    def clear(self):
        self.records.clear()


simple_file_path_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=3,
    max_size=20,
).map(lambda x: f"src/{x}.py")


@given(
    file_paths=st.lists(simple_file_path_strategy, min_size=1, max_size=3, unique=True),
)
@settings(max_examples=100, deadline=None)
def test_embedding_latency_logging_completeness(file_paths: list[str]):
    """
    **Feature: indexing-observability, Property 3: Embedding latency logging completeness**
    **Validates: Requirements 2.1**

    *For any* embedding API call that completes (success or failure), a structured
    log entry should be emitted containing a `latency_ms` field with a non-negative
    numeric value.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        metadata_store = IndexMetadataStore(db_path)
        vector_store = InMemoryVectorStore()
        embedding_client = LocalEmbeddingClient(dimension=1536)

        service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            batch_size=10,
            max_workers=1,
        )

        log_capture = LogCapture()
        logger = logging.getLogger("aci.services.indexing_service")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(log_capture)

        try:
            scanned_files = []
            for fp in file_paths:
                content = f"# File: {fp}\ndef hello(): pass"
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                scanned_files.append(
                    ScannedFile(
                        path=Path(fp),
                        content=content,
                        language="python",
                        size_bytes=len(content),
                        modified_time=1700000000.0,
                        content_hash=content_hash,
                    )
                )

            async def run_indexing():
                all_chunks = []
                for scanned_file in scanned_files:
                    processed = service._process_file(scanned_file)
                    for chunk in processed.chunks:
                        chunk.metadata["_pending_file_info"] = {
                            "file_path": str(scanned_file.path),
                            "content_hash": scanned_file.content_hash,
                            "language": scanned_file.language,
                            "line_count": scanned_file.content.count("\n") + 1,
                            "chunk_count": len(processed.chunks),
                            "modified_time": scanned_file.modified_time,
                        }
                    all_chunks.extend(processed.chunks)
                await service._embed_and_store_chunks(all_chunks)

            asyncio.run(run_indexing())

            embedding_logs = [
                r for r in log_capture.records
                if "Embedding batch completed" in r.getMessage()
            ]

            assert len(embedding_logs) >= 1, "Should have at least one embedding log entry"

            for record in embedding_logs:
                assert hasattr(record, "latency_ms"), "Embedding log should have latency_ms field"
                assert isinstance(record.latency_ms, (int, float)), (
                    f"latency_ms should be numeric, got {type(record.latency_ms)}"
                )
                assert record.latency_ms >= 0, (
                    f"latency_ms should be non-negative, got {record.latency_ms}"
                )

        finally:
            logger.removeHandler(log_capture)
            logger.setLevel(original_level)
            metadata_store.close()


@given(
    file_paths=st.lists(simple_file_path_strategy, min_size=1, max_size=3, unique=True),
)
@settings(max_examples=100, deadline=None)
def test_qdrant_operation_logging_completeness(file_paths: list[str]):
    """
    **Feature: indexing-observability, Property 4: Qdrant operation logging completeness**
    **Validates: Requirements 2.2**

    *For any* Qdrant upsert operation that completes, a structured log entry should
    be emitted containing `duration_ms` and `chunk_count` fields.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        metadata_store = IndexMetadataStore(db_path)
        vector_store = InMemoryVectorStore()
        embedding_client = LocalEmbeddingClient(dimension=1536)

        service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            batch_size=10,
            max_workers=1,
        )

        log_capture = LogCapture()
        logger = logging.getLogger("aci.services.indexing_service")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(log_capture)

        try:
            scanned_files = []
            for fp in file_paths:
                content = f"# File: {fp}\ndef hello(): pass"
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                scanned_files.append(
                    ScannedFile(
                        path=Path(fp),
                        content=content,
                        language="python",
                        size_bytes=len(content),
                        modified_time=1700000000.0,
                        content_hash=content_hash,
                    )
                )

            async def run_indexing():
                all_chunks = []
                for scanned_file in scanned_files:
                    processed = service._process_file(scanned_file)
                    for chunk in processed.chunks:
                        chunk.metadata["_pending_file_info"] = {
                            "file_path": str(scanned_file.path),
                            "content_hash": scanned_file.content_hash,
                            "language": scanned_file.language,
                            "line_count": scanned_file.content.count("\n") + 1,
                            "chunk_count": len(processed.chunks),
                            "modified_time": scanned_file.modified_time,
                        }
                    all_chunks.extend(processed.chunks)
                await service._embed_and_store_chunks(all_chunks)

            asyncio.run(run_indexing())

            qdrant_logs = [
                r for r in log_capture.records
                if "Qdrant upsert completed" in r.getMessage()
            ]

            assert len(qdrant_logs) >= 1, "Should have at least one Qdrant log entry"

            for record in qdrant_logs:
                assert hasattr(record, "duration_ms"), "Qdrant log should have duration_ms field"
                assert hasattr(record, "chunk_count"), "Qdrant log should have chunk_count field"
                assert isinstance(record.duration_ms, (int, float)), (
                    f"duration_ms should be numeric, got {type(record.duration_ms)}"
                )
                assert isinstance(record.chunk_count, int), (
                    f"chunk_count should be int, got {type(record.chunk_count)}"
                )
                assert record.duration_ms >= 0, (
                    f"duration_ms should be non-negative, got {record.duration_ms}"
                )
                assert record.chunk_count >= 0, (
                    f"chunk_count should be non-negative, got {record.chunk_count}"
                )

        finally:
            logger.removeHandler(log_capture)
            logger.setLevel(original_level)
            metadata_store.close()


@given(
    file_paths=st.lists(simple_file_path_strategy, min_size=1, max_size=3, unique=True),
)
@settings(max_examples=100, deadline=None)
def test_indexing_summary_logging_completeness(file_paths: list[str]):
    """
    **Feature: indexing-observability, Property 5: Indexing summary logging completeness**
    **Validates: Requirements 2.3**

    *For any* indexing run that completes, a structured log entry should be emitted
    containing `total_chunks`, `total_files`, and `duration_seconds` fields.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        root_path = Path(tmpdir) / "repo"
        root_path.mkdir()

        metadata_store = IndexMetadataStore(db_path)
        vector_store = InMemoryVectorStore()
        embedding_client = LocalEmbeddingClient(dimension=1536)

        for fp in file_paths:
            file_path = root_path / fp
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = f"# File: {fp}\ndef hello(): pass"
            file_path.write_text(content)

        service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            batch_size=10,
            max_workers=1,
        )

        log_capture = LogCapture()
        logger = logging.getLogger("aci.services.indexing_service")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(log_capture)

        try:
            async def run_indexing():
                await service.index_directory(root_path)

            asyncio.run(run_indexing())

            summary_logs = [
                r for r in log_capture.records
                if "Indexing completed" in r.getMessage()
            ]

            assert len(summary_logs) >= 1, "Should have at least one indexing summary log entry"

            for record in summary_logs:
                assert hasattr(record, "total_chunks"), "Summary log should have total_chunks field"
                assert hasattr(record, "total_files"), "Summary log should have total_files field"
                assert hasattr(record, "duration_seconds"), (
                    "Summary log should have duration_seconds field"
                )
                assert isinstance(record.total_chunks, int), (
                    f"total_chunks should be int, got {type(record.total_chunks)}"
                )
                assert isinstance(record.total_files, int), (
                    f"total_files should be int, got {type(record.total_files)}"
                )
                assert isinstance(record.duration_seconds, (int, float)), (
                    f"duration_seconds should be numeric, got {type(record.duration_seconds)}"
                )
                assert record.total_chunks >= 0, (
                    f"total_chunks should be non-negative, got {record.total_chunks}"
                )
                assert record.total_files >= 0, (
                    f"total_files should be non-negative, got {record.total_files}"
                )
                assert record.duration_seconds >= 0, (
                    f"duration_seconds should be non-negative, got {record.duration_seconds}"
                )

        finally:
            logger.removeHandler(log_capture)
            logger.setLevel(original_level)
            metadata_store.close()

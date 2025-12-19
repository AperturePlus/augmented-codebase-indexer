"""
Property-based tests for Pending Batch tracking in Indexing Observability.

Tests for batch lifecycle and failure rollback (Properties 1, 2).
"""

import asyncio
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.core.file_scanner import ScannedFile
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.services.indexing_service import IndexingService


# Strategies for generating test data
batch_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=1,
    max_size=50,
).filter(lambda x: x.strip() != "")

file_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/._-"),
    min_size=1,
    max_size=100,
).filter(lambda x: x.strip() != "")

chunk_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=1,
    max_size=50,
).filter(lambda x: x.strip() != "")

simple_file_path_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=3,
    max_size=20,
).map(lambda x: f"src/{x}.py")


class FailingVectorStore(InMemoryVectorStore):
    """Vector store that fails after a configurable number of upserts."""

    def __init__(self, fail_after: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._fail_after = fail_after
        self._upsert_count = 0

    async def upsert(
        self,
        chunk_id: str,
        vector,
        payload: dict,
        collection_name: str | None = None,
    ) -> None:
        self._upsert_count += 1
        if self._upsert_count > self._fail_after:
            raise RuntimeError("Simulated Qdrant failure")
        await super().upsert(chunk_id, vector, payload, collection_name=collection_name)


@given(
    batch_id=batch_id_strategy,
    file_paths=st.lists(file_path_strategy, min_size=0, max_size=10, unique=True),
    chunk_ids=st.lists(chunk_id_strategy, min_size=0, max_size=20, unique=True),
)
@settings(max_examples=100, deadline=None)
def test_pending_batch_lifecycle_round_trip(
    batch_id: str, file_paths: list[str], chunk_ids: list[str]
):
    """
    **Feature: indexing-observability, Property 2: Pending batch lifecycle round-trip**
    **Validates: Requirements 1.2, 1.3**

    *For any* batch operation, if a pending batch is created at the start and
    the operation completes successfully, then no pending batch with that ID
    should exist afterward.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = IndexMetadataStore(db_path)

        try:
            # Create a pending batch
            store.create_pending_batch(batch_id, file_paths, chunk_ids)

            # Verify the batch exists
            pending = store.get_pending_batches()
            batch_ids = [b.batch_id for b in pending]
            assert batch_id in batch_ids, (
                f"Pending batch {batch_id} should exist after creation"
            )

            # Verify the batch data is correct
            batch = next(b for b in pending if b.batch_id == batch_id)
            assert batch.file_paths == file_paths
            assert batch.chunk_ids == chunk_ids
            assert isinstance(batch.created_at, datetime)

            # Complete the batch (simulating successful operation)
            completed = store.complete_pending_batch(batch_id)
            assert completed is True, "complete_pending_batch should return True"

            # Verify the batch no longer exists
            pending_after = store.get_pending_batches()
            batch_ids_after = [b.batch_id for b in pending_after]
            assert batch_id not in batch_ids_after, (
                f"Pending batch {batch_id} should not exist after completion"
            )

            # Completing again should return False
            completed_again = store.complete_pending_batch(batch_id)
            assert completed_again is False, (
                "complete_pending_batch should return False for non-existent batch"
            )

        finally:
            store.close()


@given(
    file_paths=st.lists(simple_file_path_strategy, min_size=1, max_size=5, unique=True),
)
@settings(max_examples=100, deadline=None)
def test_batch_failure_rollback_preserves_consistency(file_paths: list[str]):
    """
    **Feature: indexing-observability, Property 1: Batch failure rollback preserves consistency**
    **Validates: Requirements 1.1**

    *For any* batch of chunks where embedding succeeds but Qdrant storage fails,
    the metadata store should contain no file entries for that batch after the
    failure is handled.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        metadata_store = IndexMetadataStore(db_path)

        # Create a vector store that fails after first upsert
        failing_vector_store = FailingVectorStore(fail_after=0)
        embedding_client = LocalEmbeddingClient(dimension=1536)

        service = IndexingService(
            embedding_client=embedding_client,
            vector_store=failing_vector_store,
            metadata_store=metadata_store,
            batch_size=10,
            max_workers=1,
        )

        # Create scanned files
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

        try:
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
            assert False, "Expected RuntimeError from failing vector store"

        except RuntimeError as e:
            assert "Simulated Qdrant failure" in str(e)

        # Verify no file metadata was persisted for the failed batch
        all_files = metadata_store.get_all_files()
        persisted_paths = {f.file_path for f in all_files}

        for fp in file_paths:
            assert fp not in persisted_paths, (
                f"File {fp} should not be in metadata store after batch failure rollback"
            )

        # Verify no pending batches remain
        pending = metadata_store.get_pending_batches()
        assert len(pending) == 0, (
            f"No pending batches should remain after rollback, found {len(pending)}"
        )

        metadata_store.close()

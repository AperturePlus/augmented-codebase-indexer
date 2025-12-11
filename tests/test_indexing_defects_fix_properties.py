"""
Property-based tests for indexing service defects fix.

Tests Fix 2: Embedding count validation
"""

import shutil
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from aci.core.chunker import CodeChunk
from aci.services.indexing_service import IndexingError, IndexingService
from tests.search_service_test_utils import run_async


# =============================================================================
# Test Helpers
# =============================================================================


class PartialEmbeddingClient:
    """Embedding client that returns fewer embeddings than requested."""

    def __init__(self, return_ratio: float = 0.5):
        """
        Args:
            return_ratio: Fraction of embeddings to return (0.0 to 1.0)
        """
        self.return_ratio = return_ratio
        self.call_count = 0
        self.last_input_count = 0

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        self.last_input_count = len(texts)

        # Return fewer embeddings than requested
        return_count = max(1, int(len(texts) * self.return_ratio))
        if return_count >= len(texts):
            return_count = len(texts) - 1  # Always return at least 1 fewer

        return [[0.1] * 384 for _ in range(return_count)]


class CorrectEmbeddingClient:
    """Embedding client that returns correct number of embeddings."""

    def __init__(self):
        self.call_count = 0
        self.total_embedded = 0

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        self.total_embedded += len(texts)
        return [[0.1] * 384 for _ in range(len(texts))]


class TrackingVectorStore:
    """Vector store that tracks upsert calls."""

    def __init__(self):
        self.upserted_chunks: List[str] = []
        self.upsert_count = 0

    async def upsert(self, chunk_id: str, vector: List[float], payload: dict) -> None:
        self.upserted_chunks.append(chunk_id)
        self.upsert_count += 1

    async def search(self, *args, **kwargs):
        return []

    async def get_all_file_paths(self, collection_name=None):
        return []

    async def delete_by_file(self, file_path: str) -> int:
        return 0

    async def get_stats(self):
        return {"total_vectors": 0}

    def set_collection(self, name: str):
        pass


class TrackingMetadataStore:
    """Metadata store that tracks writes."""

    def __init__(self):
        self.written_files: List[str] = []
        self.write_count = 0
        self.pending_batches: list[str] = []

    def upsert_file(self, file_info):
        self.written_files.append(file_info.file_path)
        self.write_count += 1

    def register_repository(self, path: str, collection_name: str):
        pass

    def get_all_file_hashes(self):
        return {}

    def delete_file(self, file_path: str):
        pass

    def create_pending_batch(self, batch_id: str, file_paths: list[str], chunk_ids: list[str]):
        self.pending_batches.append(batch_id)

    def complete_pending_batch(self, batch_id: str):
        if batch_id in self.pending_batches:
            self.pending_batches.remove(batch_id)

    def rollback_pending_batch(self, batch_id: str):
        if batch_id in self.pending_batches:
            self.pending_batches.remove(batch_id)

    def close(self):
        pass


# =============================================================================
# Fix 2: Embedding Count Validation Tests
# =============================================================================


class TestEmbeddingCountMismatchRaisesException:
    """
    **Feature: search-indexing-defects-fix, Property 3: Embedding count mismatch raises exception**
    **Validates: Requirements 2.1**

    *For any* batch of chunks where embed_batch returns fewer embeddings than
    input texts, the IndexingService SHALL raise an IndexingError exception.
    """

    @given(
        num_chunks=st.integers(min_value=2, max_value=10),
        return_ratio=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=50)
    def test_mismatch_raises_indexing_error(self, num_chunks, return_ratio):
        """Embedding count mismatch should raise IndexingError."""
        # Create mock chunks
        chunks = [
            CodeChunk(
                chunk_id=f"chunk_{i}",
                file_path=f"file_{i}.py",
                start_line=1,
                end_line=10,
                content=f"def func_{i}(): pass",
                language="python",
                chunk_type="function",
                metadata={"_pending_file_info": {
                    "file_path": f"file_{i}.py",
                    "content_hash": "abc123",
                    "language": "python",
                    "line_count": 10,
                    "chunk_count": 1,
                    "modified_time": 0.0,
                }},
            )
            for i in range(num_chunks)
        ]

        # Create service with partial embedding client
        embedding_client = PartialEmbeddingClient(return_ratio=return_ratio)
        vector_store = TrackingVectorStore()
        metadata_store = TrackingMetadataStore()

        service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            batch_size=num_chunks + 1,  # Process all in one batch
        )

        # Should raise IndexingError
        with pytest.raises(IndexingError) as exc_info:
            run_async(service._embed_and_store_chunks(chunks))

        # Verify error contains useful information
        error = exc_info.value
        assert error.expected == num_chunks
        assert error.actual < num_chunks
        assert error.batch_index == 0
        assert "mismatch" in str(error).lower()


class TestMatchingEmbeddingCountsAllowStorage:
    """
    **Feature: search-indexing-defects-fix, Property 4: Matching counts allow storage**
    **Validates: Requirements 2.2**

    *For any* batch of chunks where embed_batch returns exactly the expected
    number of embeddings, all chunk-embedding pairs SHALL be stored.
    """

    @given(num_chunks=st.integers(min_value=1, max_value=10))
    @settings(max_examples=30)
    def test_matching_counts_stores_all_chunks(self, num_chunks):
        """Correct embedding count should store all chunks."""
        chunks = [
            CodeChunk(
                chunk_id=f"chunk_{i}",
                file_path="file.py",
                start_line=i * 10 + 1,
                end_line=(i + 1) * 10,
                content=f"def func_{i}(): pass",
                language="python",
                chunk_type="function",
                metadata={"_pending_file_info": {
                    "file_path": "file.py",
                    "content_hash": "abc123",
                    "language": "python",
                    "line_count": 100,
                    "chunk_count": num_chunks,
                    "modified_time": 0.0,
                }},
            )
            for i in range(num_chunks)
        ]

        embedding_client = CorrectEmbeddingClient()
        vector_store = TrackingVectorStore()
        metadata_store = TrackingMetadataStore()

        service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            batch_size=num_chunks + 1,
        )

        # Should succeed without exception
        run_async(service._embed_and_store_chunks(chunks))

        # All chunks should be stored
        assert vector_store.upsert_count == num_chunks
        assert len(vector_store.upserted_chunks) == num_chunks


class TestNoMetadataWrittenOnMismatch:
    """
    **Feature: search-indexing-defects-fix, Property 5: No metadata on mismatch**
    **Validates: Requirements 2.4**

    *For any* batch where embedding count mismatch occurs, no metadata
    SHALL be written to the metadata store for chunks in that batch.
    """

    @given(num_chunks=st.integers(min_value=2, max_value=8))
    @settings(max_examples=30)
    def test_no_metadata_written_on_mismatch(self, num_chunks):
        """Metadata should not be written when embedding fails."""
        chunks = [
            CodeChunk(
                chunk_id=f"chunk_{i}",
                file_path=f"file_{i}.py",
                start_line=1,
                end_line=10,
                content=f"def func_{i}(): pass",
                language="python",
                chunk_type="function",
                metadata={"_pending_file_info": {
                    "file_path": f"file_{i}.py",
                    "content_hash": "abc123",
                    "language": "python",
                    "line_count": 10,
                    "chunk_count": 1,
                    "modified_time": 0.0,
                }},
            )
            for i in range(num_chunks)
        ]

        embedding_client = PartialEmbeddingClient(return_ratio=0.5)
        vector_store = TrackingVectorStore()
        metadata_store = TrackingMetadataStore()

        service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            batch_size=num_chunks + 1,
        )

        # Should raise exception
        with pytest.raises(IndexingError):
            run_async(service._embed_and_store_chunks(chunks))

        # No metadata should be written
        assert metadata_store.write_count == 0
        assert len(metadata_store.written_files) == 0

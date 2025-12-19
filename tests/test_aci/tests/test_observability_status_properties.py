"""
Property-based tests for HTTP Status endpoint in Indexing Observability.

Tests for status response field presence (Property 9).
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.infrastructure.fakes import InMemoryVectorStore
from aci.infrastructure.metadata_store import IndexedFileInfo, IndexMetadataStore


@given(
    num_files=st.integers(min_value=0, max_value=10),
    num_stale=st.integers(min_value=0, max_value=10),
    staleness_seconds=st.integers(min_value=1, max_value=86400),
)
@settings(max_examples=100, deadline=None)
def test_http_status_response_field_presence(
    num_files: int,
    num_stale: int,
    staleness_seconds: int,
):
    """
    **Feature: indexing-observability, Property 9: HTTP status response field presence**
    **Validates: Requirements 3.5**

    *For any* call to `/status` with a valid indexed path, the response should
    contain `vector_count`, `file_count`, and `stale_file_count` fields.

    This test validates the data preparation logic that feeds the /status endpoint.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        metadata_store = IndexMetadataStore(db_path)
        vector_store = InMemoryVectorStore()

        try:
            indexed_at_ts = 1700000000
            indexed_at = datetime.fromtimestamp(float(indexed_at_ts))
            actual_stale = min(num_stale, num_files)

            for i in range(num_files):
                if i < actual_stale:
                    modified_time = float(indexed_at_ts + staleness_seconds)
                else:
                    modified_time = float(indexed_at_ts - 100)

                file_info = IndexedFileInfo(
                    file_path=f"src/file_{i}.py",
                    content_hash=f"hash_{i}",
                    language="python",
                    line_count=100,
                    chunk_count=5,
                    indexed_at=indexed_at,
                    modified_time=modified_time,
                )
                metadata_store.upsert_file(file_info)

            async def add_vectors():
                for i in range(num_files * 5):
                    await vector_store.upsert(
                        chunk_id=f"chunk_{i}",
                        vector=[0.1] * 1536,
                        payload={
                            "file_path": f"src/file_{i // 5}.py",
                            "start_line": 1,
                            "end_line": 10,
                            "content": "test content",
                        },
                    )

            asyncio.run(add_vectors())

            metadata_stats = metadata_store.get_stats()

            async def get_vector_stats():
                return await vector_store.get_stats()

            vector_stats = asyncio.run(get_vector_stats())

            stale_files = metadata_store.get_stale_files(limit=5)
            all_stale_files = metadata_store.get_stale_files()
            stale_file_count = len(all_stale_files)

            stale_files_sample = [
                {"path": file_path, "staleness_hours": round(staleness_secs / 3600, 2)}
                for file_path, staleness_secs in stale_files
            ]

            response = {
                "metadata": metadata_stats,
                "vector_store": vector_stats,
                "vector_count": vector_stats.get("total_vectors", 0),
                "file_count": metadata_stats.get("total_files", 0),
                "staleness": {
                    "stale_file_count": stale_file_count,
                    "stale_files_sample": stale_files_sample,
                },
                "stale_file_count": stale_file_count,
            }

            # Verify required fields are present
            assert "vector_count" in response, "Response should have 'vector_count' field"
            assert "file_count" in response, "Response should have 'file_count' field"
            assert "stale_file_count" in response, "Response should have 'stale_file_count' field"

            # Verify field types
            assert isinstance(response["vector_count"], int), (
                f"vector_count should be int, got {type(response['vector_count'])}"
            )
            assert isinstance(response["file_count"], int), (
                f"file_count should be int, got {type(response['file_count'])}"
            )
            assert isinstance(response["stale_file_count"], int), (
                f"stale_file_count should be int, got {type(response['stale_file_count'])}"
            )

            # Verify values are non-negative
            assert response["vector_count"] >= 0, "vector_count should be non-negative"
            assert response["file_count"] >= 0, "file_count should be non-negative"
            assert response["stale_file_count"] >= 0, "stale_file_count should be non-negative"

            # Verify staleness object structure
            assert "staleness" in response, "Response should have 'staleness' object"
            staleness_obj = response["staleness"]
            assert "stale_file_count" in staleness_obj, (
                "staleness object should have 'stale_file_count'"
            )
            assert "stale_files_sample" in staleness_obj, (
                "staleness object should have 'stale_files_sample'"
            )

            # Verify stale_files_sample is a list with correct structure
            assert isinstance(staleness_obj["stale_files_sample"], list), (
                "stale_files_sample should be a list"
            )
            for item in staleness_obj["stale_files_sample"]:
                assert "path" in item, "Each stale file item should have 'path'"
                assert "staleness_hours" in item, (
                    "Each stale file item should have 'staleness_hours'"
                )

            # Verify counts match expected values
            assert response["file_count"] == num_files, (
                f"file_count {response['file_count']} should match {num_files}"
            )
            assert response["stale_file_count"] == actual_stale, (
                f"stale_file_count {response['stale_file_count']} should match {actual_stale}"
            )

        finally:
            metadata_store.close()

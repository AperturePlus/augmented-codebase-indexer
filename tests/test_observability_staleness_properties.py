"""
Property-based tests for Staleness detection in Indexing Observability.

Tests for staleness computation and display limits (Properties 7, 8, 10).
"""

import tempfile
from datetime import datetime
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.infrastructure.metadata_store import IndexedFileInfo, IndexMetadataStore


# Strategies for generating test data
file_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/._-"),
    min_size=1,
    max_size=100,
).filter(lambda x: x.strip() != "")

language_strategy = st.sampled_from(["python", "javascript", "go", "java", "c", "cpp"])


@given(
    file_path=file_path_strategy,
    content_hash=st.text(min_size=8, max_size=64),
    language=language_strategy,
    line_count=st.integers(min_value=1, max_value=10000),
    chunk_count=st.integers(min_value=1, max_value=100),
    indexed_at_ts=st.integers(min_value=1000000000, max_value=2000000000),
    staleness_delta=st.integers(min_value=-86400, max_value=86400),
)
@settings(max_examples=100, deadline=None)
def test_staleness_computation_correctness(
    file_path: str,
    content_hash: str,
    language: str,
    line_count: int,
    chunk_count: int,
    indexed_at_ts: int,
    staleness_delta: int,
):
    """
    **Feature: indexing-observability, Property 7: Staleness computation correctness**
    **Validates: Requirements 3.3, 4.1, 4.2**

    *For any* file in the metadata store, if its `modified_time` (as Unix timestamp)
    is greater than its `indexed_at` timestamp, then `get_stale_files()` should
    include that file in its results.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = IndexMetadataStore(db_path)

        try:
            indexed_at = datetime.fromtimestamp(float(indexed_at_ts))
            modified_time = float(indexed_at_ts + staleness_delta)

            file_info = IndexedFileInfo(
                file_path=file_path,
                content_hash=content_hash,
                language=language,
                line_count=line_count,
                chunk_count=chunk_count,
                indexed_at=indexed_at,
                modified_time=modified_time,
            )

            store.upsert_file(file_info)
            stale_files = store.get_stale_files()
            stale_paths = [path for path, _ in stale_files]
            is_stale = staleness_delta > 0

            if is_stale:
                assert file_path in stale_paths, (
                    f"File {file_path} should be in stale files list "
                    f"(modified_time={modified_time}, indexed_at_ts={indexed_at_ts}, "
                    f"delta={staleness_delta})"
                )
            else:
                assert file_path not in stale_paths, (
                    f"File {file_path} should NOT be in stale files list "
                    f"(modified_time={modified_time}, indexed_at_ts={indexed_at_ts}, "
                    f"delta={staleness_delta})"
                )

        finally:
            store.close()


@given(
    file_path=file_path_strategy,
    content_hash=st.text(min_size=8, max_size=64),
    language=language_strategy,
    line_count=st.integers(min_value=1, max_value=10000),
    chunk_count=st.integers(min_value=1, max_value=100),
    indexed_at_ts=st.integers(min_value=1000000000, max_value=2000000000),
    staleness_seconds=st.integers(min_value=1, max_value=86400),
)
@settings(max_examples=100, deadline=None)
def test_stale_files_query_returns_duration(
    file_path: str,
    content_hash: str,
    language: str,
    line_count: int,
    chunk_count: int,
    indexed_at_ts: int,
    staleness_seconds: int,
):
    """
    **Feature: indexing-observability, Property 10: Stale files query returns duration**
    **Validates: Requirements 4.3**

    *For any* stale file returned by `get_stale_files()`, the result should include
    both the file path and a non-negative staleness duration value.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = IndexMetadataStore(db_path)

        try:
            indexed_at = datetime.fromtimestamp(float(indexed_at_ts))
            modified_time = float(indexed_at_ts + staleness_seconds)

            file_info = IndexedFileInfo(
                file_path=file_path,
                content_hash=content_hash,
                language=language,
                line_count=line_count,
                chunk_count=chunk_count,
                indexed_at=indexed_at,
                modified_time=modified_time,
            )

            store.upsert_file(file_info)
            stale_files = store.get_stale_files()

            assert len(stale_files) >= 1, "Should have at least one stale file"

            file_result = next(
                ((path, duration) for path, duration in stale_files if path == file_path),
                None,
            )
            assert file_result is not None, f"File {file_path} should be in stale files"

            returned_path, returned_duration = file_result

            assert isinstance(returned_path, str), "File path should be a string"
            assert isinstance(returned_duration, float), "Duration should be a float"
            assert returned_duration >= 0, (
                f"Staleness duration should be non-negative, got {returned_duration}"
            )
            assert abs(returned_duration - staleness_seconds) < 2.0, (
                f"Staleness duration {returned_duration} should be close to "
                f"expected {staleness_seconds}"
            )

        finally:
            store.close()


@given(
    num_stale_files=st.integers(min_value=0, max_value=20),
    staleness_seconds=st.integers(min_value=1, max_value=86400),
)
@settings(max_examples=100, deadline=None)
def test_staleness_display_limit_enforcement(
    num_stale_files: int,
    staleness_seconds: int,
):
    """
    **Feature: indexing-observability, Property 8: Staleness display limit enforcement**
    **Validates: Requirements 3.4**

    *For any* status request, the `stale_files_sample` list should contain at most
    5 entries, regardless of how many stale files exist.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = IndexMetadataStore(db_path)

        try:
            indexed_at_ts = 1700000000
            indexed_at = datetime.fromtimestamp(float(indexed_at_ts))
            modified_time = float(indexed_at_ts + staleness_seconds)

            for i in range(num_stale_files):
                file_info = IndexedFileInfo(
                    file_path=f"src/file_{i}.py",
                    content_hash=f"hash_{i}",
                    language="python",
                    line_count=100,
                    chunk_count=5,
                    indexed_at=indexed_at,
                    modified_time=modified_time,
                )
                store.upsert_file(file_info)

            stale_files_sample = store.get_stale_files(limit=5)

            assert len(stale_files_sample) <= 5, (
                f"stale_files_sample should have at most 5 entries, "
                f"got {len(stale_files_sample)}"
            )

            expected_count = min(num_stale_files, 5)
            assert len(stale_files_sample) == expected_count, (
                f"Expected {expected_count} stale files, got {len(stale_files_sample)}"
            )

        finally:
            store.close()

"""
Property-based tests for IndexMetadataStore.

**Feature: codebase-semantic-search, Property 18a: Metadata Query Correctness**
**Validates: Requirements 5.5**
"""

import tempfile
from datetime import datetime, timedelta
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

content_hash_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=64,
    max_size=64,
)

language_strategy = st.sampled_from(["python", "javascript", "go", "typescript", "unknown"])

line_count_strategy = st.integers(min_value=1, max_value=10000)
chunk_count_strategy = st.integers(min_value=1, max_value=100)
modified_time_strategy = st.floats(min_value=0, max_value=2000000000, allow_nan=False)


@st.composite
def indexed_file_info_strategy(draw, days_ago: int = 0):
    """Generate a valid IndexedFileInfo."""
    indexed_at = datetime.now() - timedelta(days=days_ago)

    return IndexedFileInfo(
        file_path=draw(file_path_strategy),
        content_hash=draw(content_hash_strategy),
        language=draw(language_strategy),
        line_count=draw(line_count_strategy),
        chunk_count=draw(chunk_count_strategy),
        indexed_at=indexed_at,
        modified_time=draw(modified_time_strategy),
    )


@given(file_info=indexed_file_info_strategy())
@settings(max_examples=100, deadline=None)
def test_metadata_round_trip(file_info: IndexedFileInfo):
    """
    *For any* IndexedFileInfo, storing and retrieving should
    return the same data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = IndexMetadataStore(db_path)

        try:
            # Store the file info
            store.upsert_file(file_info)

            # Retrieve it
            result = store.get_file_info(file_info.file_path)

            # Verify round-trip
            assert result is not None
            assert result.file_path == file_info.file_path
            assert result.content_hash == file_info.content_hash
            assert result.language == file_info.language
            assert result.line_count == file_info.line_count
            assert result.chunk_count == file_info.chunk_count
            assert result.modified_time == file_info.modified_time
        finally:
            store.close()


@given(
    old_days=st.integers(min_value=5, max_value=30),
    query_days=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100, deadline=None)
def test_get_files_older_than(old_days: int, query_days: int):
    """
    **Feature: codebase-semantic-search, Property 18a: Metadata Query Correctness**
    **Validates: Requirements 5.5**

    *For any* call to get_files_older_than(days), all returned files
    should have indexed_at earlier than (now - days).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = IndexMetadataStore(db_path)

        try:
            # Create old files (should be returned)
            old_file = IndexedFileInfo(
                file_path="old/file.py",
                content_hash="a" * 64,
                language="python",
                line_count=100,
                chunk_count=5,
                indexed_at=datetime.now() - timedelta(days=old_days),
                modified_time=1000.0,
            )
            store.upsert_file(old_file)

            # Create recent files (should NOT be returned)
            recent_file = IndexedFileInfo(
                file_path="recent/file.py",
                content_hash="b" * 64,
                language="python",
                line_count=50,
                chunk_count=2,
                indexed_at=datetime.now(),
                modified_time=2000.0,
            )
            store.upsert_file(recent_file)

            # Query for files older than query_days
            old_files = store.get_files_older_than(query_days)

            # Old file should be in results (old_days > query_days)
            assert "old/file.py" in old_files

            # Recent file should NOT be in results
            assert "recent/file.py" not in old_files

            # Verify all returned files are actually old
            cutoff = datetime.now() - timedelta(days=query_days)
            for file_path in old_files:
                info = store.get_file_info(file_path)
                assert info is not None
                assert info.indexed_at < cutoff, (
                    f"File {file_path} indexed at {info.indexed_at} is not older than {cutoff}"
                )
        finally:
            store.close()


@given(
    files=st.lists(
        indexed_file_info_strategy(),
        min_size=1,
        max_size=10,
        unique_by=lambda x: x.file_path,
    )
)
@settings(max_examples=100, deadline=None)
def test_get_all_file_hashes(files: list[IndexedFileInfo]):
    """
    *For any* set of stored files, get_all_file_hashes should
    return a dict mapping each file_path to its content_hash.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = IndexMetadataStore(db_path)

        try:
            # Store all files
            for file_info in files:
                store.upsert_file(file_info)

            # Get all hashes
            hashes = store.get_all_file_hashes()

            # Verify all files are present with correct hashes
            assert len(hashes) == len(files)
            for file_info in files:
                assert file_info.file_path in hashes
                assert hashes[file_info.file_path] == file_info.content_hash
        finally:
            store.close()


@given(
    files=st.lists(
        indexed_file_info_strategy(),
        min_size=1,
        max_size=10,
        unique_by=lambda x: x.file_path,
    )
)
@settings(max_examples=100, deadline=None)
def test_stats_consistency(files: list[IndexedFileInfo]):
    """
    *For any* set of stored files, get_stats should return
    consistent totals matching the stored data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = IndexMetadataStore(db_path)

        try:
            # Store all files
            for file_info in files:
                store.upsert_file(file_info)

            # Get stats
            stats = store.get_stats()

            # Verify totals
            expected_files = len(files)
            expected_chunks = sum(f.chunk_count for f in files)
            expected_lines = sum(f.line_count for f in files)

            assert stats["total_files"] == expected_files
            assert stats["total_chunks"] == expected_chunks
            assert stats["total_lines"] == expected_lines

            # Verify language breakdown
            language_counts = {}
            for f in files:
                language_counts[f.language] = language_counts.get(f.language, 0) + 1

            assert stats["languages"] == language_counts
        finally:
            store.close()


@given(file_info=indexed_file_info_strategy())
@settings(max_examples=100, deadline=None)
def test_delete_file(file_info: IndexedFileInfo):
    """
    *For any* stored file, delete_file should remove it
    and subsequent get_file_info should return None.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = IndexMetadataStore(db_path)

        try:
            # Store the file
            store.upsert_file(file_info)

            # Verify it exists
            assert store.get_file_info(file_info.file_path) is not None

            # Delete it
            deleted = store.delete_file(file_info.file_path)
            assert deleted is True

            # Verify it's gone
            assert store.get_file_info(file_info.file_path) is None

            # Delete again should return False
            deleted_again = store.delete_file(file_info.file_path)
            assert deleted_again is False
        finally:
            store.close()



@given(
    repos=st.lists(
        st.tuples(
            file_path_strategy,  # root_path
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=20,
            ),  # collection_name
        ),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0],  # Unique root paths
    ),
    files_per_repo=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_reset_clears_all_metadata(
    repos: list[tuple[str, str]],
    files_per_repo: int,
):
    """
    **Feature: vector-store-isolation, Property 4: Reset clears all metadata**
    **Validates: Requirements 2.2, 2.4**

    *For any* metadata store containing repository registrations and indexed file
    records, after reset completes, the metadata store SHALL contain zero
    repositories and zero indexed files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = IndexMetadataStore(db_path)

        try:
            # Register repositories
            for root_path, collection_name in repos:
                store.register_repository(root_path, collection_name)

            # Add indexed files for each repository
            for root_path, _ in repos:
                for i in range(files_per_repo):
                    file_info = IndexedFileInfo(
                        file_path=f"{root_path}/file_{i}.py",
                        content_hash="a" * 64,
                        language="python",
                        line_count=100,
                        chunk_count=5,
                        indexed_at=datetime.now(),
                        modified_time=1000.0 + i,
                    )
                    store.upsert_file(file_info)

            # Verify data exists before reset
            repos_before = store.get_repositories()
            stats_before = store.get_stats()
            assert len(repos_before) == len(repos), (
                f"Expected {len(repos)} repositories before reset, got {len(repos_before)}"
            )
            assert stats_before["total_files"] == len(repos) * files_per_repo, (
                f"Expected {len(repos) * files_per_repo} files before reset, "
                f"got {stats_before['total_files']}"
            )

            # Execute reset (clear_all)
            store.clear_all()

            # Verify all data is cleared after reset
            repos_after = store.get_repositories()
            stats_after = store.get_stats()
            all_files_after = store.get_all_files()

            # Property: Zero repositories after reset
            assert len(repos_after) == 0, (
                f"Expected 0 repositories after reset, got {len(repos_after)}"
            )

            # Property: Zero indexed files after reset
            assert stats_after["total_files"] == 0, (
                f"Expected 0 files after reset, got {stats_after['total_files']}"
            )
            assert len(all_files_after) == 0, (
                f"Expected 0 files in get_all_files after reset, got {len(all_files_after)}"
            )

            # Property: Zero chunks after reset
            assert stats_after["total_chunks"] == 0, (
                f"Expected 0 chunks after reset, got {stats_after['total_chunks']}"
            )

            # Property: Zero lines after reset
            assert stats_after["total_lines"] == 0, (
                f"Expected 0 lines after reset, got {stats_after['total_lines']}"
            )

        finally:
            store.close()

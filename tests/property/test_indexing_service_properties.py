"""
Property-based tests for IndexingService.

Tests the correctness properties for indexing operations including
incremental updates and parallel processing determinism.

Uses InMemoryVectorStore + LocalEmbeddingClient for testing without
external dependencies.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.core.file_scanner import FileScanner
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.services.indexing_service import IndexingService


# Strategies for generating test data
@st.composite
def python_function_code(draw):
    """Generate valid Python function code."""
    func_name = draw(st.from_regex(r"[a-z][a-z0-9_]{2,10}", fullmatch=True))
    body_lines = draw(st.integers(min_value=1, max_value=5))
    body = "\n".join([f"    x = {i}" for i in range(body_lines)])
    return f"def {func_name}():\n{body}\n    return x\n"


@st.composite
def python_file_content(draw):
    """Generate valid Python file content with multiple functions."""
    num_functions = draw(st.integers(min_value=1, max_value=3))
    functions = [draw(python_function_code()) for _ in range(num_functions)]
    return "\n\n".join(functions)


def create_test_file(directory: Path, filename: str, content: str) -> Path:
    """Create a test file with given content."""
    file_path = directory / filename
    file_path.write_text(content, encoding="utf-8")
    return file_path


def run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def create_indexing_components(temp_dir: Path, db_name: str = "metadata.db"):
    """Create indexing service components."""
    vector_store = InMemoryVectorStore()
    embedding_client = LocalEmbeddingClient()
    metadata_store = IndexMetadataStore(temp_dir / db_name)
    file_scanner = FileScanner(extensions={".py"})

    # Point default vector store operations at the repository collection so
    # tests that omit collection_name still observe indexed data.
    from aci.core.path_utils import get_collection_name_for_path

    collection_name = get_collection_name_for_path(str(temp_dir.resolve()))
    if hasattr(vector_store, "set_collection"):
        vector_store.set_collection(collection_name)

    service = IndexingService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        max_workers=1,
    )
    return service, vector_store, metadata_store


class TestIncrementalUpdateModifiedFile:
    """
    **Feature: codebase-semantic-search, Property 15: Incremental Update - Modified File**
    **Validates: Requirements 5.1**

    *For any* initially indexed file that is modified, after incremental update,
    the Vector_Store should only contain the new chunks (old chunks deleted).
    """

    @given(
        original_content=python_file_content(),
        modified_content=python_file_content(),
    )
    @settings(
        max_examples=20,
        deadline=30000,
    )
    def test_modified_file_updates_chunks(self, original_content, modified_content):
        """Modified files should have old chunks removed and new chunks added."""
        assume(original_content != modified_content)

        temp_dir = Path(tempfile.mkdtemp())
        try:
            service, vector_store, metadata_store = create_indexing_components(temp_dir)

            # Create initial file
            test_file = create_test_file(temp_dir, "test_module.py", original_content)

            # Initial indexing
            result1 = run_async(service.index_directory(temp_dir))
            assert result1.total_files >= 1

            # Get original chunks
            original_stats = run_async(vector_store.get_stats())
            original_stats["total_vectors"]

            # Modify the file
            test_file.write_text(modified_content, encoding="utf-8")

            # Incremental update
            result2 = run_async(service.update_incremental(temp_dir))

            # Verify modified file was detected
            assert result2.modified_files == 1
            assert result2.new_files == 0
            assert result2.deleted_files == 0

            # Verify chunks were updated (old removed, new added)
            new_stats = run_async(vector_store.get_stats())

            # All chunks should be for the modified file
            assert new_stats["total_files"] == 1
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestIncrementalUpdateDeletedFile:
    """
    **Feature: codebase-semantic-search, Property 16: Incremental Update - Deleted File**
    **Validates: Requirements 5.2**

    *For any* initially indexed file that is deleted, after incremental update,
    the Vector_Store should not contain any chunks for that file path.
    """

    @given(content=python_file_content())
    @settings(
        max_examples=20,
        deadline=30000,
    )
    def test_deleted_file_removes_chunks(self, content):
        """Deleted files should have all their chunks removed."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            service, vector_store, metadata_store = create_indexing_components(temp_dir)

            # Create initial file
            test_file = create_test_file(temp_dir, "to_delete.py", content)

            # Initial indexing
            result1 = run_async(service.index_directory(temp_dir))
            assert result1.total_files >= 1

            # Verify chunks exist
            stats_before = run_async(vector_store.get_stats())
            assert stats_before["total_vectors"] > 0

            # Delete the file
            test_file.unlink()

            # Incremental update
            result2 = run_async(service.update_incremental(temp_dir))

            # Verify deleted file was detected
            assert result2.deleted_files == 1
            assert result2.new_files == 0
            assert result2.modified_files == 0

            # Verify all chunks were removed
            stats_after = run_async(vector_store.get_stats())
            assert stats_after["total_vectors"] == 0
            assert stats_after["total_files"] == 0
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestIncrementalUpdateNewFile:
    """
    **Feature: codebase-semantic-search, Property 17: Incremental Update - New File**
    **Validates: Requirements 5.3**

    *For any* file added after initial indexing, after incremental update,
    the Vector_Store should contain chunks for the new file.
    """

    @given(
        initial_content=python_file_content(),
        new_content=python_file_content(),
    )
    @settings(
        max_examples=20,
        deadline=30000,
    )
    def test_new_file_adds_chunks(self, initial_content, new_content):
        """New files should have their chunks added to the index."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            service, vector_store, metadata_store = create_indexing_components(temp_dir)

            # Create initial file
            create_test_file(temp_dir, "initial.py", initial_content)

            # Initial indexing
            run_async(service.index_directory(temp_dir))

            stats_before = run_async(vector_store.get_stats())
            initial_chunks = stats_before["total_vectors"]

            # Add new file
            create_test_file(temp_dir, "new_module.py", new_content)

            # Incremental update
            result2 = run_async(service.update_incremental(temp_dir))

            # Verify new file was detected
            assert result2.new_files == 1
            assert result2.modified_files == 0
            assert result2.deleted_files == 0

            # Verify chunks were added
            stats_after = run_async(vector_store.get_stats())
            assert stats_after["total_vectors"] > initial_chunks
            assert stats_after["total_files"] == 2
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestMetadataStatisticsConsistency:
    """
    **Feature: codebase-semantic-search, Property 18: Metadata Statistics Consistency**
    **Validates: Requirements 5.5**

    *For any* completed indexing or update operation, the IndexMetadataStore
    statistics should match the actual indexed files and VectorStore chunks.
    """

    @given(
        file_contents=st.lists(
            python_file_content(),
            min_size=1,
            max_size=3,
        )
    )
    @settings(
        max_examples=15,
        deadline=60000,
    )
    def test_metadata_matches_vector_store(self, file_contents):
        """Metadata statistics should match actual VectorStore state."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            service, vector_store, metadata_store = create_indexing_components(temp_dir)

            # Create test files
            for i, content in enumerate(file_contents):
                create_test_file(temp_dir, f"module_{i}.py", content)

            # Index directory
            run_async(service.index_directory(temp_dir))

            # Get metadata stats
            metadata_stats = metadata_store.get_stats()

            # Get vector store stats
            vector_stats = run_async(vector_store.get_stats())

            # Verify file count consistency
            assert metadata_stats["total_files"] == vector_stats["total_files"]
            assert metadata_stats["total_files"] == len(file_contents)

            # Verify chunk count consistency
            # Note: metadata_stats["total_chunks"] is the sum of chunk_count from indexed_files table
            # vector_stats["total_vectors"] is the actual count in vector store
            assert metadata_stats["total_chunks"] == vector_stats["total_vectors"]
        finally:
            metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestParallelProcessingDeterminism:
    """
    **Feature: codebase-semantic-search, Property 19: Parallel Processing Determinism**
    **Validates: Requirements 6.1**

    *For any* codebase, indexing with different worker counts should produce
    the same set of chunks (same content, same line numbers), though order may differ.
    """

    def _create_indexing_service(self, temp_dir: Path, max_workers: int, db_suffix: str):
        """Create an IndexingService with specified worker count."""
        vector_store = InMemoryVectorStore()
        embedding_client = LocalEmbeddingClient()
        metadata_store = IndexMetadataStore(temp_dir / f"metadata_{db_suffix}.db")
        file_scanner = FileScanner(extensions={".py"})

        service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            file_scanner=file_scanner,
            max_workers=max_workers,
        )
        return service, vector_store, metadata_store

    @given(
        file_contents=st.lists(
            python_file_content(),
            min_size=2,
            max_size=4,
        )
    )
    @settings(
        max_examples=10,
        deadline=120000,
    )
    def test_different_worker_counts_same_results(self, file_contents):
        """Different worker counts should produce identical chunk sets."""
        temp_dir = Path(tempfile.mkdtemp())
        metadata_stores = []
        try:
            # Create test files
            for i, content in enumerate(file_contents):
                create_test_file(temp_dir, f"module_{i}.py", content)

            # Index with 1 worker
            service1, vector_store1, metadata_store1 = self._create_indexing_service(
                temp_dir, max_workers=1, db_suffix="1"
            )
            metadata_stores.append(metadata_store1)
            result1 = run_async(service1.index_directory(temp_dir))

            # Index with 2 workers
            service2, vector_store2, metadata_store2 = self._create_indexing_service(
                temp_dir, max_workers=2, db_suffix="2"
            )
            metadata_stores.append(metadata_store2)
            result2 = run_async(service2.index_directory(temp_dir))

            # Compare results
            assert result1.total_files == result2.total_files
            assert result1.total_chunks == result2.total_chunks

            # Get all chunks from both stores
            stats1 = run_async(vector_store1.get_stats())
            stats2 = run_async(vector_store2.get_stats())

            assert stats1["total_vectors"] == stats2["total_vectors"]
            assert stats1["total_files"] == stats2["total_files"]

            # Extract chunk contents for comparison
            chunks1_contents: set[tuple] = set()
            chunks2_contents: set[tuple] = set()

            for chunk_id in vector_store1._payloads:
                payload = vector_store1._payloads[chunk_id]
                chunks1_contents.add(
                    (
                        payload["file_path"],
                        payload["start_line"],
                        payload["end_line"],
                        payload["content"],
                    )
                )

            for chunk_id in vector_store2._payloads:
                payload = vector_store2._payloads[chunk_id]
                chunks2_contents.add(
                    (
                        payload["file_path"],
                        payload["start_line"],
                        payload["end_line"],
                        payload["content"],
                    )
                )

            # Chunks should be identical (ignoring order and chunk_id)
            assert chunks1_contents == chunks2_contents
        finally:
            for ms in metadata_stores:
                ms.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

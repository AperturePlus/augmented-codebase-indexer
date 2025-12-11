"""
Property-based tests for IndexingService summary handling.

Tests the correctness properties for summary artifact indexing,
re-indexing, and incremental updates.

**Feature: multi-granularity-indexing**
**Validates: Requirements 3.3, 5.4**
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Set, Tuple

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.core.chunker import Chunker, create_chunker
from aci.core.file_scanner import FileScanner
from aci.core.summary_artifact import ArtifactType
from aci.core.summary_generator import SummaryGenerator
from aci.core.tokenizer import TiktokenTokenizer
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.services.indexing_service import IndexingService


# Strategies for generating test data
@st.composite
def python_function_code(draw):
    """Generate valid Python function code."""
    func_name = draw(st.from_regex(r"[a-z][a-z0-9_]{2,10}", fullmatch=True))
    body_lines = draw(st.integers(min_value=1, max_value=3))
    body = "\n".join([f"    x = {i}" for i in range(body_lines)])
    return f"def {func_name}():\n{body}\n    return x\n"


@st.composite
def python_class_code(draw):
    """Generate valid Python class code with methods."""
    class_name = draw(st.from_regex(r"[A-Z][a-zA-Z0-9]{2,10}", fullmatch=True))
    method_name = draw(st.from_regex(r"[a-z][a-z0-9_]{2,8}", fullmatch=True))
    return f"""class {class_name}:
    def {method_name}(self):
        return 42
"""


@st.composite
def python_file_with_functions(draw):
    """Generate Python file content with functions (generates summaries)."""
    num_functions = draw(st.integers(min_value=1, max_value=2))
    functions = [draw(python_function_code()) for _ in range(num_functions)]
    return "\n\n".join(functions)


@st.composite
def python_file_with_class(draw):
    """Generate Python file content with a class (generates summaries)."""
    class_code = draw(python_class_code())
    return class_code


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


def create_indexing_components_with_summaries(temp_dir: Path, db_name: str = "metadata.db"):
    """Create indexing service components with summary generation enabled."""
    vector_store = InMemoryVectorStore()
    embedding_client = LocalEmbeddingClient()
    metadata_store = IndexMetadataStore(temp_dir / db_name)
    file_scanner = FileScanner(extensions={".py"})
    
    # Create chunker with summary generator
    tokenizer = TiktokenTokenizer()
    summary_generator = SummaryGenerator(tokenizer)
    chunker = Chunker(
        tokenizer=tokenizer,
        summary_generator=summary_generator,
    )

    service = IndexingService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        chunker=chunker,
        max_workers=1,  # Sequential for deterministic summary generation
    )
    return service, vector_store, metadata_store


def get_artifacts_by_file(
    vector_store: InMemoryVectorStore,
) -> dict[str, list[Tuple[str, str]]]:
    """
    Get all artifacts grouped by file path.
    
    Returns:
        Dict mapping file_path -> list of (artifact_id, artifact_type)
    """
    result: dict[str, list[Tuple[str, str]]] = {}
    collection_data = vector_store._collections.get(vector_store._collection_name, {})
    
    for artifact_id, (_, payload) in collection_data.items():
        file_path = payload.get("file_path", "")
        artifact_type = payload.get("artifact_type", "chunk")
        if file_path not in result:
            result[file_path] = []
        result[file_path].append((artifact_id, artifact_type))
    
    return result


def get_summary_artifacts_for_file(
    vector_store: InMemoryVectorStore,
    file_path: str,
) -> list[Tuple[str, str, str]]:
    """
    Get all summary artifacts for a specific file.
    
    Returns:
        List of (artifact_id, artifact_type, content) tuples
    """
    result = []
    collection_data = vector_store._collections.get(vector_store._collection_name, {})
    
    for artifact_id, (_, payload) in collection_data.items():
        if payload.get("file_path") != file_path:
            continue
        artifact_type = payload.get("artifact_type", "chunk")
        if artifact_type != ArtifactType.CHUNK.value:
            content = payload.get("content", "")
            result.append((artifact_id, artifact_type, content))
    
    return result


def count_artifacts_by_type(
    vector_store: InMemoryVectorStore,
) -> dict[str, int]:
    """Count artifacts by type in the vector store."""
    counts: dict[str, int] = {}
    collection_data = vector_store._collections.get(vector_store._collection_name, {})
    
    for _, (_, payload) in collection_data.items():
        artifact_type = payload.get("artifact_type", "chunk")
        counts[artifact_type] = counts.get(artifact_type, 0) + 1
    
    return counts


class TestReindexingReplacesSummaries:
    """
    **Feature: multi-granularity-indexing, Property 8: Re-indexing replaces summaries**
    **Validates: Requirements 3.3**

    *For any* file that is modified and re-indexed, all previously stored 
    summary artifacts for that file SHALL be deleted and replaced with 
    newly generated summaries.
    """

    @given(
        original_content=python_file_with_functions(),
        modified_content=python_file_with_functions(),
    )
    @settings(
        max_examples=100,
        deadline=60000,
    )
    def test_reindexing_replaces_summaries(self, original_content, modified_content):
        """
        Modified files should have old summaries removed and new summaries added.
        
        Property: For any file that is modified and re-indexed, all previously
        stored summary artifacts for that file SHALL be deleted and replaced
        with newly generated summaries.
        """
        # Ensure content is actually different
        assume(original_content.strip() != modified_content.strip())

        temp_dir = Path(tempfile.mkdtemp())
        metadata_store = None
        try:
            service, vector_store, metadata_store = create_indexing_components_with_summaries(
                temp_dir
            )

            # Create initial file
            test_file = create_test_file(temp_dir, "test_module.py", original_content)
            file_path_str = str(test_file)

            # Initial indexing
            result1 = run_async(service.index_directory(temp_dir))
            assert result1.total_files >= 1

            # Get original summaries for the file
            original_summaries = get_summary_artifacts_for_file(vector_store, file_path_str)
            original_summary_ids = {s[0] for s in original_summaries}

            # Modify the file
            test_file.write_text(modified_content, encoding="utf-8")

            # Incremental update (re-indexing)
            result2 = run_async(service.update_incremental(temp_dir))

            # Verify modified file was detected
            assert result2.modified_files == 1

            # Get new summaries for the file
            new_summaries = get_summary_artifacts_for_file(vector_store, file_path_str)
            new_summary_ids = {s[0] for s in new_summaries}

            # Property verification: old summaries should be replaced
            # None of the original summary IDs should remain
            assert original_summary_ids.isdisjoint(new_summary_ids), (
                f"Old summary IDs should be deleted. "
                f"Original: {original_summary_ids}, New: {new_summary_ids}"
            )

        finally:
            if metadata_store:
                metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestIncrementalUpdateOnlyAffectsModifiedFiles:
    """
    **Feature: multi-granularity-indexing, Property 12: Incremental update only affects modified files**
    **Validates: Requirements 5.4**

    *For any* incremental update where only a subset of files are modified,
    only the summaries for modified files SHALL be regenerated; summaries
    for unmodified files SHALL remain unchanged.
    """

    @given(
        file1_content=python_file_with_functions(),
        file2_content=python_file_with_functions(),
        file1_modified_content=python_file_with_functions(),
    )
    @settings(
        max_examples=100,
        deadline=60000,
    )
    def test_incremental_update_preserves_unmodified_summaries(
        self, file1_content, file2_content, file1_modified_content
    ):
        """
        Incremental update should only regenerate summaries for modified files.
        
        Property: For any incremental update where only a subset of files are
        modified, only the summaries for modified files SHALL be regenerated;
        summaries for unmodified files SHALL remain unchanged.
        """
        # Ensure file1 is actually modified
        assume(file1_content.strip() != file1_modified_content.strip())
        # Ensure files are different from each other
        assume(file1_content.strip() != file2_content.strip())

        temp_dir = Path(tempfile.mkdtemp())
        metadata_store = None
        try:
            service, vector_store, metadata_store = create_indexing_components_with_summaries(
                temp_dir
            )

            # Create two files
            file1 = create_test_file(temp_dir, "module1.py", file1_content)
            file2 = create_test_file(temp_dir, "module2.py", file2_content)
            file1_path = str(file1)
            file2_path = str(file2)

            # Initial indexing
            result1 = run_async(service.index_directory(temp_dir))
            assert result1.total_files == 2

            # Get original summaries for both files
            file1_original_summaries = get_summary_artifacts_for_file(vector_store, file1_path)
            file2_original_summaries = get_summary_artifacts_for_file(vector_store, file2_path)
            
            file1_original_ids = {s[0] for s in file1_original_summaries}
            file2_original_ids = {s[0] for s in file2_original_summaries}
            file2_original_contents = {s[2] for s in file2_original_summaries}

            # Modify only file1
            file1.write_text(file1_modified_content, encoding="utf-8")

            # Incremental update
            result2 = run_async(service.update_incremental(temp_dir))

            # Verify only file1 was detected as modified
            assert result2.modified_files == 1
            assert result2.new_files == 0
            assert result2.deleted_files == 0

            # Get new summaries
            file1_new_summaries = get_summary_artifacts_for_file(vector_store, file1_path)
            file2_new_summaries = get_summary_artifacts_for_file(vector_store, file2_path)
            
            file1_new_ids = {s[0] for s in file1_new_summaries}
            file2_new_ids = {s[0] for s in file2_new_summaries}
            file2_new_contents = {s[2] for s in file2_new_summaries}

            # Property verification:
            # 1. File1 summaries should be replaced (different IDs)
            assert file1_original_ids.isdisjoint(file1_new_ids), (
                f"Modified file summaries should be replaced. "
                f"Original: {file1_original_ids}, New: {file1_new_ids}"
            )

            # 2. File2 summaries should remain unchanged (same IDs and content)
            assert file2_original_ids == file2_new_ids, (
                f"Unmodified file summary IDs should remain the same. "
                f"Original: {file2_original_ids}, New: {file2_new_ids}"
            )
            assert file2_original_contents == file2_new_contents, (
                f"Unmodified file summary contents should remain the same."
            )

        finally:
            if metadata_store:
                metadata_store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

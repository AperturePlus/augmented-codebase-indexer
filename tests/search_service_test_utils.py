"""Shared helpers for search service property tests."""

import asyncio
from pathlib import Path

from hypothesis import strategies as st

from aci.core.file_scanner import FileScanner
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.services.indexing_service import IndexingService
from aci.services.search_service import SearchService


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


def create_indexed_search_env(temp_dir: Path, file_contents: dict):
    """
    Create an indexed environment for search testing.

    Args:
        temp_dir: Temporary directory for files
        file_contents: Dict mapping filename to content

    Returns:
        Tuple of (search_service, vector_store, metadata_store)
    """
    vector_store = InMemoryVectorStore()
    embedding_client = LocalEmbeddingClient()
    metadata_store = IndexMetadataStore(temp_dir / "metadata.db")
    file_scanner = FileScanner(extensions={".py"})

    # Create indexing service
    indexing_service = IndexingService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        max_workers=1,
    )

    # Create test files
    for filename, content in file_contents.items():
        create_test_file(temp_dir, filename, content)

    # Index the directory
    run_async(indexing_service.index_directory(temp_dir))

    # Create search service
    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=None,
        default_limit=10,
    )

    return search_service, vector_store, metadata_store

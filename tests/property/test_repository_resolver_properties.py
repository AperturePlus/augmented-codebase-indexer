"""
Property-based tests for repository resolution.

Tests for Properties 8, 9, and 10 from the service-initialization-refactor spec.
"""

import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.services.repository_resolver import resolve_repository

# Strategy for generating valid directory names
valid_dir_name_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-"),
    min_size=1,
    max_size=20,
).filter(lambda s: s and not s.startswith("-"))


# Strategy for generating collection names
collection_name_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
    min_size=5,
    max_size=30,
).filter(lambda s: s and s[0].isalpha())


@st.composite
def indexed_repository_strategy(draw):
    """
    Generate a temporary directory that is registered as indexed.

    Returns a tuple of (temp_dir_path, metadata_store, collection_name).
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # Create in-memory metadata store
    metadata_store = IndexMetadataStore(":memory:")
    metadata_store.initialize()

    # Generate a collection name
    collection_name = draw(collection_name_strategy)

    # Register the repository
    metadata_store.register_repository(str(temp_path.resolve()), collection_name)

    return (temp_path, metadata_store, collection_name)


@st.composite
def indexed_repository_without_collection_strategy(draw):
    """
    Generate a temporary directory that is indexed but has no collection name.

    This simulates a legacy index that needs collection name generation.
    Returns a tuple of (temp_dir_path, metadata_store).
    """
    # Draw a dummy value to satisfy hypothesis composite requirement
    _ = draw(st.just(None))

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # Create in-memory metadata store
    metadata_store = IndexMetadataStore(":memory:")
    metadata_store.initialize()

    # Register the repository WITHOUT a collection name (legacy behavior)
    metadata_store.set_index_info(
        index_id=str(temp_path.resolve()),
        root_path=str(temp_path.resolve()),
        collection_name=None,  # No collection name - legacy index
    )

    return (temp_path, metadata_store)


@st.composite
def non_indexed_directory_strategy(draw):
    """
    Generate a temporary directory that exists but is NOT indexed.

    Returns a tuple of (temp_dir_path, metadata_store).
    """
    # Draw a dummy value to satisfy hypothesis composite requirement
    _ = draw(st.just(None))

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # Create in-memory metadata store (empty - no indexes registered)
    metadata_store = IndexMetadataStore(":memory:")
    metadata_store.initialize()

    return (temp_path, metadata_store)


@given(data=indexed_repository_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_resolve_repository_returns_valid_for_indexed_path(data):
    """
    **Feature: service-initialization-refactor, Property 8: Repository Resolution Consistency**
    **Validates: Requirements 4.1, 4.2, 4.3**

    For any valid indexed repository path, the resolve_repository() helper
    SHALL return a valid resolution with a non-empty collection_name.
    """
    temp_path, metadata_store, expected_collection = data

    try:
        result = resolve_repository(temp_path, metadata_store)

        # Resolution should be valid
        assert result.valid, f"Expected valid resolution, got error: {result.error_message}"

        # Collection name should be non-empty
        assert result.collection_name, "Collection name should not be empty"
        assert result.collection_name == expected_collection, \
            f"Expected collection '{expected_collection}', got '{result.collection_name}'"

        # Indexed root should be set
        assert result.indexed_root, "Indexed root should be set"

        # Error message should be None for valid resolution
        assert result.error_message is None, \
            f"Error message should be None for valid resolution, got: {result.error_message}"
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)


@given(data=non_indexed_directory_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_resolve_repository_returns_error_for_non_indexed_path(data):
    """
    **Feature: service-initialization-refactor, Property 9: Non-Indexed Path Error**
    **Validates: Requirements 4.4**

    For any path that exists but is not indexed, the resolve_repository() helper
    SHALL return an invalid resolution with an error message indicating indexing is required.
    """
    temp_path, metadata_store = data

    try:
        result = resolve_repository(temp_path, metadata_store)

        # Resolution should be invalid
        assert not result.valid, "Expected invalid resolution for non-indexed path"

        # Collection name should be None
        assert result.collection_name is None, \
            f"Collection name should be None, got: {result.collection_name}"

        # Error message should indicate indexing is required
        assert result.error_message is not None, "Error message should be set"
        assert "not been indexed" in result.error_message.lower() or \
               "indexing" in result.error_message.lower(), \
            f"Error message should mention indexing: {result.error_message}"
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)


@given(data=indexed_repository_without_collection_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_resolve_repository_generates_collection_name_for_legacy_index(data):
    """
    **Feature: service-initialization-refactor, Property 10: Legacy Collection Name Generation**
    **Validates: Requirements 4.5**

    For any indexed repository with missing collection_name in metadata,
    the resolve_repository() helper SHALL generate a collection name and register it.
    """
    temp_path, metadata_store = data

    try:
        # Verify the index exists but has no collection name
        path_abs = str(temp_path.resolve())
        index_info_before = metadata_store.get_index_info(path_abs)
        assert index_info_before is not None, "Index should exist"
        assert index_info_before.get("collection_name") is None, \
            "Collection name should be None before resolution"

        # Resolve the repository
        result = resolve_repository(temp_path, metadata_store)

        # Resolution should be valid
        assert result.valid, f"Expected valid resolution, got error: {result.error_message}"

        # Collection name should be generated
        assert result.collection_name, "Collection name should be generated"
        assert len(result.collection_name) > 0, "Collection name should not be empty"

        # Collection name should be registered in metadata store
        index_info_after = metadata_store.get_index_info(path_abs)
        assert index_info_after is not None, "Index should still exist"
        assert index_info_after.get("collection_name") == result.collection_name, \
            "Collection name should be registered in metadata store"
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)


def test_resolve_repository_returns_error_for_nonexistent_path():
    """
    **Feature: service-initialization-refactor, Property 8: Repository Resolution Consistency**
    **Validates: Requirements 4.1**

    For any path that does not exist, resolve_repository() SHALL return
    an invalid resolution with an appropriate error message.
    """
    metadata_store = IndexMetadataStore(":memory:")
    metadata_store.initialize()

    # Use a path that definitely doesn't exist
    nonexistent_path = Path("/nonexistent/path/that/does/not/exist/12345")

    result = resolve_repository(nonexistent_path, metadata_store)

    assert not result.valid, "Expected invalid resolution for nonexistent path"
    assert result.error_message is not None, "Error message should be set"
    assert "does not exist" in result.error_message.lower(), \
        f"Error message should mention path doesn't exist: {result.error_message}"


def test_resolve_repository_returns_error_for_file_path():
    """
    **Feature: service-initialization-refactor, Property 8: Repository Resolution Consistency**
    **Validates: Requirements 4.1**

    For any path that is a file (not a directory), resolve_repository() SHALL return
    an invalid resolution with an appropriate error message.
    """
    import tempfile

    metadata_store = IndexMetadataStore(":memory:")
    metadata_store.initialize()

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        file_path = Path(f.name)

    try:
        result = resolve_repository(file_path, metadata_store)

        assert not result.valid, "Expected invalid resolution for file path"
        assert result.error_message is not None, "Error message should be set"
        assert "not a directory" in result.error_message.lower(), \
            f"Error message should mention not a directory: {result.error_message}"
    finally:
        file_path.unlink(missing_ok=True)


def test_resolve_repository_accepts_string_path():
    """
    **Feature: service-initialization-refactor, Property 8: Repository Resolution Consistency**
    **Validates: Requirements 4.1, 4.2, 4.3**

    resolve_repository() SHALL accept both Path objects and string paths.
    """
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp()

    try:
        metadata_store = IndexMetadataStore(":memory:")
        metadata_store.initialize()

        # Register with string path
        metadata_store.register_repository(temp_dir, "test_collection")

        # Resolve with string path
        result = resolve_repository(temp_dir, metadata_store)

        assert result.valid, f"Expected valid resolution, got error: {result.error_message}"
        assert result.collection_name == "test_collection"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_repository_finds_parent_index():
    """
    **Feature: service-initialization-refactor, Property 8: Repository Resolution Consistency**
    **Validates: Requirements 4.2, 4.3**

    When searching a subdirectory of an indexed repository, resolve_repository()
    SHALL find the parent index and return its collection name.
    """
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp()

    try:
        # Create a subdirectory
        subdir = Path(temp_dir) / "subdir" / "nested"
        subdir.mkdir(parents=True)

        metadata_store = IndexMetadataStore(":memory:")
        metadata_store.initialize()

        # Register the parent directory
        metadata_store.register_repository(temp_dir, "parent_collection")

        # Resolve the subdirectory
        result = resolve_repository(subdir, metadata_store)

        assert result.valid, f"Expected valid resolution, got error: {result.error_message}"
        assert result.collection_name == "parent_collection", \
            f"Expected parent collection, got: {result.collection_name}"
        assert result.indexed_root == temp_dir, \
            f"Expected indexed root to be parent, got: {result.indexed_root}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

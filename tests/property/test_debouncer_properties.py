"""
Property-based tests for Debouncer batch merging.

**Feature: file-watcher-service, Property 4: Debouncer Batch Merging**
**Validates: Requirements 3.1, 3.4, 3.5**
"""

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.core.file_events import DebouncedBatch, FileEvent, FileEventType

# Strategies for generating file events - use sampled_from for efficiency
COMMON_DIRS = ["src", "lib", "tests", "pkg", "utils", "core", "app"]
COMMON_NAMES = ["main", "test", "utils", "helper", "config", "index", "app"]
COMMON_EXTS = ["py", "js", "ts", "go", "java", "c", "cpp"]


@st.composite
def safe_path_strategy(draw):
    """Generate a safe file path efficiently."""
    num_dirs = draw(st.integers(min_value=1, max_value=3))
    dirs = [draw(st.sampled_from(COMMON_DIRS)) for _ in range(num_dirs)]
    name = draw(st.sampled_from(COMMON_NAMES))
    ext = draw(st.sampled_from(COMMON_EXTS))
    return "/" + "/".join(dirs) + "/" + name + "." + ext


safe_path = safe_path_strategy()


@st.composite
def file_event_strategy(draw):
    """Generate valid FileEvent instances."""
    event_type = draw(st.sampled_from(list(FileEventType)))
    file_path = Path(draw(safe_path))

    old_path = None
    if event_type == FileEventType.MOVED:
        old_path = Path(draw(safe_path))

    return FileEvent(
        event_type=event_type,
        file_path=file_path,
        old_path=old_path,
        timestamp=draw(st.floats(min_value=0, max_value=1e10)),
    )


@st.composite
def event_sequence_strategy(draw):
    """Generate a sequence of file events."""
    return draw(st.lists(file_event_strategy(), min_size=1, max_size=50))


@given(events=event_sequence_strategy())
@settings(max_examples=100)
def test_debouncer_batch_merging_multiple_modifications(events: list[FileEvent]):
    """
    **Feature: file-watcher-service, Property 4: Debouncer Batch Merging**
    **Validates: Requirements 3.1, 3.4**

    For any sequence of file events within the debounce window,
    multiple modifications to the same file SHALL result in a single entry.
    """
    batch = DebouncedBatch()

    for event in events:
        batch.merge(event)

    # Verify no duplicates across sets
    all_paths = list(batch.created) + list(batch.modified) + list(batch.deleted)
    unique_paths = set(all_paths)

    # Each path should appear at most once across all sets
    assert len(all_paths) == len(unique_paths), (
        f"Duplicate paths found: created={batch.created}, "
        f"modified={batch.modified}, deleted={batch.deleted}"
    )


@given(path=safe_path)
@settings(max_examples=100)
def test_debouncer_create_then_delete_cancels(path: str):
    """
    **Feature: file-watcher-service, Property 4: Debouncer Batch Merging**
    **Validates: Requirements 3.5**

    When a file is created and then deleted within the debounce window,
    the Debouncer SHALL cancel both events.
    """
    batch = DebouncedBatch()
    file_path = Path(path)

    # Create event
    batch.merge(FileEvent(event_type=FileEventType.CREATED, file_path=file_path))

    # Delete event
    batch.merge(FileEvent(event_type=FileEventType.DELETED, file_path=file_path))

    # Both should be cancelled
    assert file_path not in batch.created
    assert file_path not in batch.deleted
    assert file_path not in batch.modified


@given(path=safe_path)
@settings(max_examples=100)
def test_debouncer_create_then_modify_stays_created(path: str):
    """
    **Feature: file-watcher-service, Property 4: Debouncer Batch Merging**
    **Validates: Requirements 3.4**

    When a file is created and then modified within the debounce window,
    the Debouncer SHALL only record the create (final state is new file).
    """
    batch = DebouncedBatch()
    file_path = Path(path)

    # Create event
    batch.merge(FileEvent(event_type=FileEventType.CREATED, file_path=file_path))

    # Modify event
    batch.merge(FileEvent(event_type=FileEventType.MODIFIED, file_path=file_path))

    # Should only be in created set
    assert file_path in batch.created
    assert file_path not in batch.modified
    assert file_path not in batch.deleted


@given(path=safe_path, num_modifications=st.integers(min_value=2, max_value=10))
@settings(max_examples=100)
def test_debouncer_multiple_modifications_single_entry(path: str, num_modifications: int):
    """
    **Feature: file-watcher-service, Property 4: Debouncer Batch Merging**
    **Validates: Requirements 3.4**

    When a file is modified multiple times within the debounce window,
    the Debouncer SHALL only record the final state (single entry).
    """
    batch = DebouncedBatch()
    file_path = Path(path)

    # Multiple modify events
    for _ in range(num_modifications):
        batch.merge(FileEvent(event_type=FileEventType.MODIFIED, file_path=file_path))

    # Should only appear once in modified set
    assert file_path in batch.modified
    assert file_path not in batch.created
    assert file_path not in batch.deleted


@given(path=safe_path)
@settings(max_examples=100)
def test_debouncer_delete_then_create_becomes_modified(path: str):
    """
    **Feature: file-watcher-service, Property 4: Debouncer Batch Merging**
    **Validates: Requirements 3.4**

    When a file is deleted and then recreated within the debounce window,
    the Debouncer SHALL record it as modified (file was replaced).
    """
    batch = DebouncedBatch()
    file_path = Path(path)

    # Delete event
    batch.merge(FileEvent(event_type=FileEventType.DELETED, file_path=file_path))

    # Create event (recreate)
    batch.merge(FileEvent(event_type=FileEventType.CREATED, file_path=file_path))

    # Should be in modified set (file was replaced)
    assert file_path in batch.modified
    assert file_path not in batch.created
    assert file_path not in batch.deleted


@given(old_path=safe_path, new_path=safe_path)
@settings(max_examples=100)
def test_debouncer_move_event_handling(old_path: str, new_path: str):
    """
    **Feature: file-watcher-service, Property 4: Debouncer Batch Merging**
    **Validates: Requirements 3.1**

    When a file is moved, the Debouncer SHALL record the old path as deleted
    and the new path as created.
    """
    batch = DebouncedBatch()
    old = Path(old_path)
    new = Path(new_path)

    # Move event
    batch.merge(
        FileEvent(event_type=FileEventType.MOVED, file_path=new, old_path=old)
    )

    # Old path should be deleted, new path should be created
    if old != new:
        assert old in batch.deleted
        assert new in batch.created


def test_debouncer_batch_is_empty():
    """Test that is_empty() correctly identifies empty batches."""
    batch = DebouncedBatch()
    assert batch.is_empty()

    batch.merge(FileEvent(event_type=FileEventType.CREATED, file_path=Path("/test.py")))
    assert not batch.is_empty()


def test_debouncer_batch_clear():
    """Test that clear() removes all events."""
    batch = DebouncedBatch()
    batch.merge(FileEvent(event_type=FileEventType.CREATED, file_path=Path("/a.py")))
    batch.merge(FileEvent(event_type=FileEventType.MODIFIED, file_path=Path("/b.py")))
    batch.merge(FileEvent(event_type=FileEventType.DELETED, file_path=Path("/c.py")))

    assert not batch.is_empty()
    batch.clear()
    assert batch.is_empty()


def test_debouncer_batch_copy():
    """Test that copy() creates an independent copy."""
    batch = DebouncedBatch()
    batch.merge(FileEvent(event_type=FileEventType.CREATED, file_path=Path("/test.py")))

    copy = batch.copy()
    assert copy.created == batch.created

    # Modifying copy should not affect original
    copy.created.add(Path("/other.py"))
    assert Path("/other.py") not in batch.created

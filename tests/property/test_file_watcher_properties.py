"""
Property-based tests for FileWatcher.

**Feature: file-watcher-service, Property 2: File Event Emission**
**Feature: file-watcher-service, Property 3: Ignore Pattern Filtering**
**Validates: Requirements 2.1-2.6**
"""

import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.core.file_events import FileEvent, FileEventType
from aci.infrastructure.file_watcher import FileWatcher, _WatchdogEventHandler

# Strategies for generating test data - use sampled_from for efficiency
COMMON_EXTENSIONS = [".py", ".js", ".ts", ".go", ".java", ".c", ".cpp", ".rs", ".rb"]
COMMON_FILENAMES = ["main", "test", "utils", "helper", "config", "index", "app", "lib"]
COMMON_DIRNAMES = ["src", "lib", "utils", "tests", "pkg", "internal"]

file_extension = st.sampled_from(COMMON_EXTENSIONS)
safe_filename = st.sampled_from(COMMON_FILENAMES)
dir_name = st.sampled_from(COMMON_DIRNAMES)


@st.composite
def extension_set_strategy(draw):
    """Generate a set of file extensions."""
    # Use sampled_from for efficiency
    num_extensions = draw(st.integers(min_value=1, max_value=5))
    extensions = set()
    for _ in range(num_extensions):
        extensions.add(draw(file_extension))
    return extensions


@st.composite
def ignore_pattern_strategy(draw):
    """Generate ignore patterns."""
    pattern_types = draw(
        st.lists(
            st.sampled_from(["dir", "ext", "file"]),
            min_size=0,
            max_size=3,
            unique=True,
        )
    )

    patterns = []
    for ptype in pattern_types:
        if ptype == "dir":
            patterns.append(draw(st.sampled_from(["ignored", "skip", "temp", "cache"])))
        elif ptype == "ext":
            patterns.append(f"*{draw(st.sampled_from(['.log', '.tmp', '.bak']))}")
        else:
            patterns.append(draw(st.sampled_from(["ignore_me.py", "skip.js"])))

    return patterns


class MockCallback:
    """Mock callback for collecting emitted events."""

    def __init__(self):
        self.events: list[FileEvent] = []

    def __call__(self, event: FileEvent) -> None:
        self.events.append(event)

    def clear(self):
        self.events.clear()


# ============================================================================
# Property 2: File Event Emission Tests
# ============================================================================


@given(
    extensions=extension_set_strategy(),
    filename=safe_filename,
)
@settings(max_examples=100)
def test_event_handler_emits_for_valid_extensions(extensions: set, filename: str):
    """
    **Feature: file-watcher-service, Property 2: File Event Emission**
    **Validates: Requirements 2.1, 2.2, 2.3**

    For any file with a supported extension, the FileWatcher SHALL emit
    the corresponding event type with the correct file path.
    """
    # Pick one extension from the set
    ext = list(extensions)[0]
    full_filename = f"{filename}{ext}"

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        callback = MockCallback()

        handler = _WatchdogEventHandler(
            callback=callback,
            extensions=extensions,
            ignore_patterns=[],
            root_path=root_path,
        )

        # Test CREATED event
        file_path = root_path / full_filename
        handler._emit_event(FileEventType.CREATED, file_path)

        assert len(callback.events) == 1
        assert callback.events[0].event_type == FileEventType.CREATED
        assert callback.events[0].file_path == file_path


@given(
    extensions=extension_set_strategy(),
    filename=safe_filename,
)
@settings(max_examples=100)
def test_event_handler_filters_unsupported_extensions(extensions: set, filename: str):
    """
    **Feature: file-watcher-service, Property 3: Ignore Pattern Filtering**
    **Validates: Requirements 2.6**

    For any file with an unsupported extension, the FileWatcher SHALL NOT
    emit any events for that file.
    """
    # Create an extension NOT in the set
    unsupported_ext = ".xyz"
    assume(unsupported_ext not in extensions)

    full_filename = f"{filename}{unsupported_ext}"

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        callback = MockCallback()

        handler = _WatchdogEventHandler(
            callback=callback,
            extensions=extensions,
            ignore_patterns=[],
            root_path=root_path,
        )

        file_path = root_path / full_filename

        # _should_process should return False for unsupported extension
        assert not handler._should_process(file_path, is_directory=False)


@given(filename=safe_filename)
@settings(max_examples=100)
def test_event_handler_skips_directories(filename: str):
    """
    **Feature: file-watcher-service, Property 2: File Event Emission**
    **Validates: Requirements 2.1-2.4**

    The FileWatcher SHALL only emit events for files, not directories.
    """
    extensions = {".py", ".js"}

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        callback = MockCallback()

        handler = _WatchdogEventHandler(
            callback=callback,
            extensions=extensions,
            ignore_patterns=[],
            root_path=root_path,
        )

        dir_path = root_path / filename

        # _should_process should return False for directories
        assert not handler._should_process(dir_path, is_directory=True)


# ============================================================================
# Property 3: Ignore Pattern Filtering Tests
# ============================================================================


@given(
    ignore_patterns=ignore_pattern_strategy(),
    filename=safe_filename,
)
@settings(max_examples=100)
def test_event_handler_respects_ignore_patterns(
    ignore_patterns: list, filename: str
):
    """
    **Feature: file-watcher-service, Property 3: Ignore Pattern Filtering**
    **Validates: Requirements 2.5**

    For any file that matches an ignore pattern, the FileWatcher SHALL NOT
    emit any events for that file.
    """
    extensions = {".py", ".js", ".log", ".tmp", ".bak"}

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        callback = MockCallback()

        handler = _WatchdogEventHandler(
            callback=callback,
            extensions=extensions,
            ignore_patterns=ignore_patterns,
            root_path=root_path,
        )

        # Test files in ignored directories
        for pattern in ignore_patterns:
            if not pattern.startswith("*"):
                # Directory pattern
                ignored_file = root_path / pattern / f"{filename}.py"
                assert handler._should_ignore(ignored_file), (
                    f"File {ignored_file} should be ignored by pattern {pattern}"
                )


@given(filename=safe_filename)
@settings(max_examples=100)
def test_event_handler_ignores_files_matching_extension_pattern(filename: str):
    """
    **Feature: file-watcher-service, Property 3: Ignore Pattern Filtering**
    **Validates: Requirements 2.5**

    Files matching extension-based ignore patterns (*.log) SHALL be ignored.
    """
    extensions = {".py", ".js", ".log"}
    ignore_patterns = ["*.log"]

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        callback = MockCallback()

        handler = _WatchdogEventHandler(
            callback=callback,
            extensions=extensions,
            ignore_patterns=ignore_patterns,
            root_path=root_path,
        )

        # .log files should be ignored
        log_file = root_path / f"{filename}.log"
        assert handler._should_ignore(log_file)

        # .py files should NOT be ignored
        py_file = root_path / f"{filename}.py"
        assert not handler._should_ignore(py_file)


@given(subdir=dir_name, filename=safe_filename)
@settings(max_examples=100)
def test_event_handler_ignores_files_in_ignored_directories(
    subdir: str, filename: str
):
    """
    **Feature: file-watcher-service, Property 3: Ignore Pattern Filtering**
    **Validates: Requirements 2.5**

    Files in directories matching ignore patterns SHALL be ignored.
    """
    extensions = {".py", ".js"}
    ignore_patterns = ["node_modules", "__pycache__", subdir]

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        callback = MockCallback()

        handler = _WatchdogEventHandler(
            callback=callback,
            extensions=extensions,
            ignore_patterns=ignore_patterns,
            root_path=root_path,
        )

        # Files in ignored directory should be ignored
        ignored_file = root_path / subdir / f"{filename}.py"
        assert handler._should_ignore(ignored_file), (
            f"File {ignored_file} should be ignored (in ignored dir {subdir})"
        )


# ============================================================================
# Event Type Tests
# ============================================================================


def test_event_handler_emits_correct_event_types():
    """
    **Feature: file-watcher-service, Property 2: File Event Emission**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

    The FileWatcher SHALL emit the correct event type for each file operation.
    """
    extensions = {".py"}

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        callback = MockCallback()

        handler = _WatchdogEventHandler(
            callback=callback,
            extensions=extensions,
            ignore_patterns=[],
            root_path=root_path,
        )

        file_path = root_path / "test.py"

        # Test each event type
        for event_type in [
            FileEventType.CREATED,
            FileEventType.MODIFIED,
            FileEventType.DELETED,
        ]:
            callback.clear()
            handler._emit_event(event_type, file_path)

            assert len(callback.events) == 1
            assert callback.events[0].event_type == event_type
            assert callback.events[0].file_path == file_path


def test_event_handler_move_event_includes_old_path():
    """
    **Feature: file-watcher-service, Property 2: File Event Emission**
    **Validates: Requirements 2.4**

    For MOVED events, the FileWatcher SHALL include both old and new paths.
    """
    extensions = {".py"}

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        callback = MockCallback()

        handler = _WatchdogEventHandler(
            callback=callback,
            extensions=extensions,
            ignore_patterns=[],
            root_path=root_path,
        )

        old_path = root_path / "old.py"
        new_path = root_path / "new.py"

        handler._emit_event(FileEventType.MOVED, new_path, old_path=old_path)

        assert len(callback.events) == 1
        assert callback.events[0].event_type == FileEventType.MOVED
        assert callback.events[0].file_path == new_path
        assert callback.events[0].old_path == old_path


# ============================================================================
# FileWatcher Lifecycle Tests
# ============================================================================


def test_file_watcher_start_validates_path():
    """
    **Feature: file-watcher-service, Property 2: File Event Emission**
    **Validates: Requirements 2.1**

    FileWatcher.start() SHALL validate that the path exists and is a directory.
    """
    watcher = FileWatcher(extensions={".py"})
    callback = MockCallback()

    # Test non-existent path
    try:
        watcher.start(Path("/nonexistent/path"), callback)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "does not exist" in str(e)

    # Test file path (not directory)
    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        try:
            watcher.start(Path(f.name), callback)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "not a directory" in str(e)


def test_file_watcher_start_stop_lifecycle():
    """
    **Feature: file-watcher-service, Property 2: File Event Emission**
    **Validates: Requirements 2.1**

    FileWatcher SHALL properly manage start/stop lifecycle.
    """
    watcher = FileWatcher(extensions={".py"})
    callback = MockCallback()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initially not running
        assert not watcher.is_running()

        # Start watching
        watcher.start(Path(tmpdir), callback)
        assert watcher.is_running()

        # Stop watching
        watcher.stop()
        assert not watcher.is_running()


def test_file_watcher_cannot_start_twice():
    """
    **Feature: file-watcher-service, Property 2: File Event Emission**
    **Validates: Requirements 2.1**

    FileWatcher SHALL raise an error if started while already running.
    """
    watcher = FileWatcher(extensions={".py"})
    callback = MockCallback()

    with tempfile.TemporaryDirectory() as tmpdir:
        watcher.start(Path(tmpdir), callback)

        try:
            watcher.start(Path(tmpdir), callback)
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError as e:
            assert "already running" in str(e)
        finally:
            watcher.stop()


# ============================================================================
# Extension Filtering Tests
# ============================================================================


@given(extensions=extension_set_strategy())
@settings(max_examples=100)
def test_has_valid_extension_property(extensions: set):
    """
    **Feature: file-watcher-service, Property 3: Ignore Pattern Filtering**
    **Validates: Requirements 2.6**

    _has_valid_extension SHALL return True only for files with extensions
    in the configured set.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        callback = MockCallback()

        handler = _WatchdogEventHandler(
            callback=callback,
            extensions=extensions,
            ignore_patterns=[],
            root_path=root_path,
        )

        # Test each extension in the set
        for ext in extensions:
            file_path = root_path / f"test{ext}"
            assert handler._has_valid_extension(file_path), (
                f"Extension {ext} should be valid"
            )

        # Test an extension NOT in the set
        invalid_ext = ".invalid_extension_xyz"
        assume(invalid_ext not in extensions)
        invalid_file = root_path / f"test{invalid_ext}"
        assert not handler._has_valid_extension(invalid_file)

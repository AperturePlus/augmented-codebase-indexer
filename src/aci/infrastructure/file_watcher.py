"""
File watcher infrastructure component.

Provides file system monitoring using watchdog library with support for:
- File creation, modification, deletion, and move events
- Extension filtering for supported file types
- Ignore pattern filtering (gitignore-style)
- Async callback integration
"""

import fnmatch
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from aci.core.file_events import FileEvent, FileEventType

logger = logging.getLogger(__name__)


class FileWatcherInterface(Protocol):
    """Protocol for file watcher implementations."""

    def start(self, path: Path, callback: Callable[[FileEvent], None]) -> None:
        """
        Start watching the specified directory.

        Args:
            path: Directory path to watch
            callback: Function to call when file events occur
        """
        ...

    def stop(self) -> None:
        """Stop watching and release resources."""
        ...

    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        ...


class FileWatcher(FileWatcherInterface):
    """
    File system watcher implementation using watchdog.

    Monitors a directory for file changes and emits FileEvent objects
    through a callback. Supports filtering by file extension and
    ignore patterns.
    """

    def __init__(
        self,
        extensions: set[str],
        ignore_patterns: list[str] | None = None,
        follow_symlinks: bool = False,
    ):
        """
        Initialize the file watcher.

        Args:
            extensions: Set of file extensions to watch (e.g., {'.py', '.js'})
            ignore_patterns: List of gitignore-style patterns to ignore
            follow_symlinks: Whether to follow symbolic links (default: False)
        """
        self._extensions = {ext.lower() for ext in extensions}
        self._ignore_patterns = ignore_patterns or []
        self._follow_symlinks = follow_symlinks
        self._observer: Observer | None = None
        self._callback: Callable[[FileEvent], None] | None = None
        self._watch_path: Path | None = None
        self._lock = threading.Lock()

    def start(self, path: Path, callback: Callable[[FileEvent], None]) -> None:
        """
        Start watching the specified directory.

        Args:
            path: Directory path to watch (must exist and be a directory)
            callback: Function to call when file events occur

        Raises:
            ValueError: If path doesn't exist or isn't a directory
            RuntimeError: If watcher is already running
        """
        with self._lock:
            if self._observer is not None and self._observer.is_alive():
                raise RuntimeError("File watcher is already running")

            path = Path(path).resolve()
            if not path.exists():
                raise ValueError(f"Path does not exist: {path}")
            if not path.is_dir():
                raise ValueError(f"Path is not a directory: {path}")

            self._watch_path = path
            self._callback = callback

            # Create event handler
            handler = _WatchdogEventHandler(
                callback=self._handle_event,
                extensions=self._extensions,
                ignore_patterns=self._ignore_patterns,
                root_path=path,
            )

            # Create and start observer
            self._observer = Observer()
            self._observer.schedule(handler, str(path), recursive=True)
            self._observer.start()

            logger.info(f"Started watching: {path}")

    def stop(self) -> None:
        """Stop watching and release resources."""
        with self._lock:
            if self._observer is not None:
                self._observer.stop()
                self._observer.join(timeout=5.0)
                self._observer = None
                logger.info(f"Stopped watching: {self._watch_path}")
            self._callback = None
            self._watch_path = None

    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        with self._lock:
            return self._observer is not None and self._observer.is_alive()

    def _handle_event(self, event: FileEvent) -> None:
        """Internal handler that forwards events to the callback."""
        if self._callback is not None:
            try:
                self._callback(event)
            except Exception as e:
                logger.error(f"Error in file event callback: {e}")


class _WatchdogEventHandler(FileSystemEventHandler):
    """
    Internal watchdog event handler.

    Converts watchdog events to FileEvent objects and applies filtering.
    """

    def __init__(
        self,
        callback: Callable[[FileEvent], None],
        extensions: set[str],
        ignore_patterns: list[str],
        root_path: Path,
    ):
        """
        Initialize the event handler.

        Args:
            callback: Function to call with FileEvent objects
            extensions: Set of file extensions to include
            ignore_patterns: List of patterns to ignore
            root_path: Root path being watched (for relative path calculation)
        """
        super().__init__()
        self._callback = callback
        self._extensions = extensions
        self._ignore_patterns = ignore_patterns
        self._root_path = root_path

    def _should_ignore(self, path: Path) -> bool:
        """
        Check if a path should be ignored based on patterns.

        Args:
            path: Path to check

        Returns:
            True if the path should be ignored
        """
        # Get relative path for pattern matching
        try:
            rel_path = path.relative_to(self._root_path)
        except ValueError:
            rel_path = path

        path_str = str(rel_path).replace("\\", "/")
        name = path.name

        for pattern in self._ignore_patterns:
            # Match against filename
            if fnmatch.fnmatch(name, pattern):
                return True
            # Match against full relative path
            if fnmatch.fnmatch(path_str, pattern):
                return True
            # Match against any path component
            for part in rel_path.parts:
                if fnmatch.fnmatch(part, pattern):
                    return True

        return False

    def _has_valid_extension(self, path: Path) -> bool:
        """
        Check if a file has a valid extension.

        Args:
            path: Path to check

        Returns:
            True if the file has a supported extension
        """
        return path.suffix.lower() in self._extensions

    def _should_process(self, path: Path, is_directory: bool = False) -> bool:
        """
        Determine if an event for this path should be processed.

        Args:
            path: Path to check
            is_directory: Whether the path is a directory

        Returns:
            True if the event should be processed
        """
        # Skip directories - we only care about file events
        if is_directory:
            return False

        # Check ignore patterns
        if self._should_ignore(path):
            logger.debug(f"Ignoring event for: {path}")
            return False

        # Check extension
        if not self._has_valid_extension(path):
            logger.debug(f"Ignoring unsupported extension: {path}")
            return False

        return True

    def _emit_event(
        self,
        event_type: FileEventType,
        file_path: Path,
        old_path: Path | None = None,
    ) -> None:
        """
        Create and emit a FileEvent.

        Args:
            event_type: Type of the event
            file_path: Path to the affected file
            old_path: Previous path for move events
        """
        event = FileEvent(
            event_type=event_type,
            file_path=file_path,
            old_path=old_path,
        )
        logger.debug(f"Emitting event: {event_type.value} - {file_path}")
        self._callback(event)

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file/directory creation events."""
        if isinstance(event, DirCreatedEvent):
            return  # Skip directory events

        path = Path(event.src_path)
        if self._should_process(path, is_directory=False):
            self._emit_event(FileEventType.CREATED, path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file/directory modification events."""
        if isinstance(event, DirModifiedEvent):
            return  # Skip directory events

        path = Path(event.src_path)
        if self._should_process(path, is_directory=False):
            self._emit_event(FileEventType.MODIFIED, path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file/directory deletion events."""
        if isinstance(event, DirDeletedEvent):
            return  # Skip directory events

        path = Path(event.src_path)
        # For deletion, we can't check if it's a directory anymore
        # so we check extension and ignore patterns only
        if self._should_ignore(path):
            logger.debug(f"Ignoring delete event for: {path}")
            return
        if not self._has_valid_extension(path):
            logger.debug(f"Ignoring delete for unsupported extension: {path}")
            return

        self._emit_event(FileEventType.DELETED, path)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file/directory move events."""
        if isinstance(event, DirMovedEvent):
            return  # Skip directory events

        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)

        # Check if source had valid extension (for deletion part)
        src_valid = self._has_valid_extension(src_path) and not self._should_ignore(
            src_path
        )
        # Check if destination has valid extension (for creation part)
        dest_valid = self._should_process(dest_path, is_directory=False)

        if src_valid or dest_valid:
            # Emit as a MOVED event with both paths
            self._emit_event(
                FileEventType.MOVED,
                dest_path,
                old_path=src_path if src_valid else None,
            )

"""
File event models for the file watching service.

Provides data structures for representing file system events and
batched event collections for debounced processing.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FileEventType(Enum):
    """Types of file system events."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileEvent:
    """
    Represents a single file system event.

    Attributes:
        event_type: Type of the file event (CREATED, MODIFIED, DELETED, MOVED)
        file_path: Path to the affected file
        old_path: Previous path for MOVED events, None otherwise
        timestamp: Unix timestamp when the event occurred
    """

    event_type: FileEventType
    file_path: Path
    old_path: Path | None = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        if isinstance(self.old_path, str):
            self.old_path = Path(self.old_path)


@dataclass
class DebouncedBatch:
    """
    A batch of debounced file events ready for processing.

    Events are merged intelligently:
    - Multiple modifications to the same file result in a single entry
    - A create followed by delete cancels both out
    - A create followed by modify results in just a create

    Attributes:
        created: Set of paths for newly created files
        modified: Set of paths for modified files
        deleted: Set of paths for deleted files
    """

    created: set[Path] = field(default_factory=set)
    modified: set[Path] = field(default_factory=set)
    deleted: set[Path] = field(default_factory=set)

    def is_empty(self) -> bool:
        """Check if the batch contains no events."""
        return not (self.created or self.modified or self.deleted)

    def merge(self, event: FileEvent) -> None:
        """
        Merge a single event into this batch with intelligent deduplication.

        Merge rules:
        - CREATED: Add to created set (unless already in modified/deleted)
        - MODIFIED: Add to modified set (unless in created set)
        - DELETED: Remove from created/modified, add to deleted
        - MOVED: Treat as delete of old_path + create of new path

        Args:
            event: The file event to merge into this batch
        """
        path = event.file_path

        if event.event_type == FileEventType.CREATED:
            # If file was previously deleted, it's now modified (recreated)
            if path in self.deleted:
                self.deleted.discard(path)
                self.modified.add(path)
            # If not already tracked as created or modified, add to created
            elif path not in self.created and path not in self.modified:
                self.created.add(path)

        elif event.event_type == FileEventType.MODIFIED:
            # If file was created in this batch, keep it as created
            # If file was deleted in this batch, it's being modified after delete
            # which means it was recreated - move to modified
            if path in self.deleted:
                self.deleted.discard(path)
                self.modified.add(path)
            elif path not in self.created:
                self.modified.add(path)

        elif event.event_type == FileEventType.DELETED:
            # If file was created in this batch, cancel both out
            if path in self.created:
                self.created.discard(path)
            else:
                # Remove from modified if present, add to deleted
                self.modified.discard(path)
                self.deleted.add(path)

        elif event.event_type == FileEventType.MOVED:
            # Handle old path as deletion
            if event.old_path:
                old_path = event.old_path
                if old_path in self.created:
                    self.created.discard(old_path)
                else:
                    self.modified.discard(old_path)
                    self.deleted.add(old_path)

            # Handle new path as creation
            if path in self.deleted:
                self.deleted.discard(path)
                self.modified.add(path)
            elif path not in self.created and path not in self.modified:
                self.created.add(path)

    def total_count(self) -> int:
        """Return total number of events in the batch."""
        return len(self.created) + len(self.modified) + len(self.deleted)

    def clear(self) -> None:
        """Clear all events from the batch."""
        self.created.clear()
        self.modified.clear()
        self.deleted.clear()

    def copy(self) -> "DebouncedBatch":
        """Create a copy of this batch."""
        return DebouncedBatch(
            created=set(self.created),
            modified=set(self.modified),
            deleted=set(self.deleted),
        )

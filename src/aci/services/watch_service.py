"""
Watch Service for automatic file monitoring and incremental indexing.

Coordinates file system watching with debounced incremental index updates.
Provides lifecycle management, statistics tracking, and error recovery.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from aci.core.debouncer import DebouncedBatch, Debouncer
from aci.core.file_events import FileEvent
from aci.core.path_utils import is_system_directory
from aci.core.watch_config import WatchConfig
from aci.infrastructure.file_watcher import FileWatcherInterface
from aci.services.indexing_service import IndexingService

if TYPE_CHECKING:
    from aci.services.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class WatchStats:
    """
    Statistics for the watch service.

    Tracks events received, updates triggered, timing information,
    and error counts for monitoring and debugging.
    """

    started_at: datetime = field(default_factory=datetime.now)
    events_received: int = 0
    updates_triggered: int = 0
    last_update_at: datetime | None = None
    last_update_duration_ms: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict:
        """Serialize stats to dictionary for JSON reporting."""
        return {
            "started_at": self.started_at.isoformat(),
            "events_received": self.events_received,
            "updates_triggered": self.updates_triggered,
            "last_update_at": (
                self.last_update_at.isoformat() if self.last_update_at else None
            ),
            "last_update_duration_ms": self.last_update_duration_ms,
            "errors": self.errors,
        }


class WatchServiceError(Exception):
    """Base exception for watch service errors."""

    pass


class PathValidationError(WatchServiceError):
    """Raised when path validation fails."""

    pass


class WatchService:
    """
    File watching service that monitors directories and triggers incremental updates.

    Coordinates between FileWatcher (infrastructure) and IndexingService (services)
    with debouncing to batch rapid file changes into single update operations.

    Attributes:
        config: Watch configuration including path, debounce delay, and patterns
    """

    def __init__(
        self,
        indexing_service: IndexingService,
        file_watcher: FileWatcherInterface,
        config: WatchConfig,
        metrics_collector: Optional["MetricsCollector"] = None,
    ):
        """
        Initialize the watch service.

        Args:
            indexing_service: Service for performing incremental index updates
            file_watcher: File system watcher implementation
            config: Watch configuration
            metrics_collector: Optional metrics collector for observability
        """
        self._indexing_service = indexing_service
        self._file_watcher = file_watcher
        self._config = config
        self._metrics_collector = metrics_collector
        self._debouncer: Debouncer | None = None
        self._stats = WatchStats()
        self._running = False
        self._update_lock = asyncio.Lock()
        self._update_in_progress = False
        self._pending_during_update: DebouncedBatch | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

    @property
    def config(self) -> WatchConfig:
        """Get the watch configuration."""
        return self._config

    async def start(self) -> None:
        """
        Start the watch service.

        Validates the path, checks if initial indexing is needed,
        and begins file system monitoring.

        Raises:
            PathValidationError: If the path doesn't exist or isn't a directory
            WatchServiceError: If the service is already running
        """
        if self._running:
            raise WatchServiceError("Watch service is already running")

        # Validate path
        watch_path = self._config.watch_path.resolve()
        self._validate_path(watch_path)

        logger.info(
            f"Starting watch service for: {watch_path}",
            extra={
                "watch_path": str(watch_path),
                "debounce_ms": self._config.debounce_ms,
            },
        )

        # Store event loop reference for thread-safe callback
        self._event_loop = asyncio.get_running_loop()
        self._shutdown_event = asyncio.Event()

        # Check if path is indexed, perform initial index if needed
        await self._ensure_indexed(watch_path)

        # Create debouncer with callback
        self._debouncer = Debouncer(
            delay_ms=self._config.debounce_ms,
            on_batch_ready=self._on_batch_ready,
        )

        # Start file watcher
        self._file_watcher.start(watch_path, self._on_file_event_sync)
        self._running = True
        self._stats = WatchStats()

        logger.info(
            "Watch service started",
            extra={
                "watch_path": str(watch_path),
                "config": self._config.to_dict(),
            },
        )

    async def stop(self) -> None:
        """
        Stop the watch service gracefully.

        Completes any in-progress update, flushes pending events,
        and releases file system watchers.
        """
        if not self._running:
            logger.debug("Watch service is not running, nothing to stop")
            return

        logger.info("Stopping watch service...")

        # Signal shutdown
        if self._shutdown_event:
            self._shutdown_event.set()

        # Wait for any in-progress update to complete
        async with self._update_lock:
            pass  # Just wait for lock to be released

        # Flush any pending events
        if self._debouncer and self._debouncer.has_pending():
            logger.debug("Flushing pending events before shutdown")
            await self._debouncer.flush()

        # Stop file watcher
        self._file_watcher.stop()
        self._running = False
        self._debouncer = None
        self._event_loop = None

        logger.info(
            "Watch service stopped",
            extra={"stats": self._stats.to_dict()},
        )

    def is_running(self) -> bool:
        """Check if the watch service is currently running."""
        return self._running

    def get_stats(self) -> WatchStats:
        """Get current watch statistics."""
        return self._stats

    def get_pending_count(self) -> int:
        """Get the number of pending events waiting to be processed."""
        if self._debouncer:
            return self._debouncer.get_pending_count()
        return 0

    def _validate_path(self, path: Path) -> None:
        """
        Validate that the path exists and is a directory.

        Args:
            path: Path to validate

        Raises:
            PathValidationError: If validation fails
        """
        if not path.exists():
            raise PathValidationError(f"Path does not exist: {path}")

        if not path.is_dir():
            raise PathValidationError(f"Path is not a directory: {path}")

        if is_system_directory(path):
            raise PathValidationError(f"Cannot watch system directory: {path}")

    async def _ensure_indexed(self, path: Path) -> None:
        """
        Ensure the path has been indexed, performing initial index if needed.

        Args:
            path: Path to check/index
        """
        # Check if already indexed by inspecting repository-scoped metadata.
        # If no files are indexed for this root, perform initial indexing.
        try:
            abs_root = str(path.resolve())
            existing_hashes = self._indexing_service._metadata_store.get_file_hashes_under_root(abs_root)
            if len(existing_hashes) == 0:
                logger.info(
                    "No existing index found, performing initial indexing",
                    extra={"path": str(path)},
                )
                await self._indexing_service.index_directory(path)
        except Exception as e:
            logger.warning(
                f"Could not check index status, assuming indexed: {e}",
                extra={"path": str(path)},
            )

    def _on_file_event_sync(self, event: FileEvent) -> None:
        """
        Synchronous callback for file events from watchdog thread.

        Schedules the async handler on the event loop.

        Args:
            event: The file event from the watcher
        """
        if self._event_loop is None or not self._running:
            return

        # Schedule async handler on the event loop
        asyncio.run_coroutine_threadsafe(
            self._on_file_event(event),
            self._event_loop,
        )

    async def _on_file_event(self, event: FileEvent) -> None:
        """
        Handle a file event from the watcher.

        Logs the event at DEBUG level and adds it to the debouncer.
        Requirements 8.1: Log event type and file path at DEBUG level.

        Args:
            event: The file event to handle
        """
        if not self._running:
            return

        self._stats.events_received += 1

        # Requirements 8.1: Always log file events at DEBUG level
        logger.debug(
            "File change detected: %s - %s",
            event.event_type.value,
            event.file_path,
            extra={
                "event_type": event.event_type.value,
                "file_path": str(event.file_path),
                "timestamp": event.timestamp,
                "watch_path": str(self._config.watch_path),
            },
        )

        # Record metric for events received
        if self._metrics_collector:
            self._metrics_collector.record_watch_event(event.event_type.value)

        if self._debouncer:
            await self._debouncer.add_event(event)

    async def _on_batch_ready(self, batch: DebouncedBatch) -> None:
        """
        Handle a debounced batch of events.

        Triggers an incremental update for the accumulated changes.
        If an update is already in progress, queues the batch for later.

        Args:
            batch: The batch of debounced events
        """
        if not self._running or batch.is_empty():
            return

        # Check if update is already in progress
        if self._update_in_progress:
            logger.debug(
                "Update in progress, queueing batch for next cycle",
                extra={"batch_size": batch.total_count()},
            )
            # Merge into pending batch
            if self._pending_during_update is None:
                self._pending_during_update = batch.copy()
            else:
                # Merge events from new batch into pending
                for path in batch.created:
                    self._pending_during_update.created.add(path)
                for path in batch.modified:
                    self._pending_during_update.modified.add(path)
                for path in batch.deleted:
                    self._pending_during_update.deleted.add(path)
            return

        await self._process_batch(batch)

        # Process any batches that accumulated during the update
        while self._pending_during_update and not self._pending_during_update.is_empty():
            pending = self._pending_during_update
            self._pending_during_update = None
            await self._process_batch(pending)

    async def _process_batch(self, batch: DebouncedBatch) -> None:
        """
        Process a batch of file changes by triggering incremental update.

        Requirements 8.2: Log pending changes at INFO level when update starts.
        Requirements 8.3: Log duration and statistics at INFO level when complete.
        Requirements 8.4: Log errors with full context at ERROR level.

        Args:
            batch: The batch to process
        """
        async with self._update_lock:
            self._update_in_progress = True
            total_changes = batch.total_count()
            start_time = time.time()

            try:
                # Requirements 8.2: Log pending changes at INFO level
                logger.info(
                    "Starting incremental update with %d pending changes",
                    total_changes,
                    extra={
                        "created_count": len(batch.created),
                        "modified_count": len(batch.modified),
                        "deleted_count": len(batch.deleted),
                        "total_changes": total_changes,
                        "watch_path": str(self._config.watch_path),
                        "created_files": [str(p) for p in list(batch.created)[:10]],
                        "modified_files": [str(p) for p in list(batch.modified)[:10]],
                        "deleted_files": [str(p) for p in list(batch.deleted)[:10]],
                    },
                )

                # Trigger incremental update
                result = await self._indexing_service.update_incremental(
                    self._config.watch_path
                )

                duration_ms = (time.time() - start_time) * 1000
                self._stats.updates_triggered += 1
                self._stats.last_update_at = datetime.now()
                self._stats.last_update_duration_ms = duration_ms

                # Requirements 8.3: Log duration and statistics at INFO level
                logger.info(
                    "Incremental update completed in %.2fms",
                    duration_ms,
                    extra={
                        "duration_ms": duration_ms,
                        "new_files": result.new_files,
                        "modified_files": result.modified_files,
                        "deleted_files": result.deleted_files,
                        "total_chunks": result.total_chunks,
                        "total_files": result.total_files,
                        "watch_path": str(self._config.watch_path),
                        "updates_triggered_total": self._stats.updates_triggered,
                    },
                )

                # Record metrics for update completion
                if self._metrics_collector:
                    self._metrics_collector.record_watch_update(
                        duration_ms=duration_ms,
                        files_processed=result.total_files,
                        chunks_processed=result.total_chunks,
                    )

            except Exception as e:
                self._stats.errors += 1
                duration_ms = (time.time() - start_time) * 1000

                # Requirements 8.4: Log error with full context at ERROR level
                logger.error(
                    "Error during incremental update: %s",
                    str(e),
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "duration_ms": duration_ms,
                        "watch_path": str(self._config.watch_path),
                        "pending_changes": total_changes,
                        "created_count": len(batch.created),
                        "modified_count": len(batch.modified),
                        "deleted_count": len(batch.deleted),
                        "errors_total": self._stats.errors,
                    },
                    exc_info=True,
                )

                # Record error metric
                if self._metrics_collector:
                    self._metrics_collector.record_watch_error()

                # Continue watching despite error (error recovery)

            finally:
                self._update_in_progress = False

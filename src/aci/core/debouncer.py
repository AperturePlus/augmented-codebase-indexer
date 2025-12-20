"""
Debouncer component for batching file system events.

Provides async debouncing logic to combine rapid file changes into
single update batches, reducing unnecessary index updates.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from aci.core.file_events import DebouncedBatch, FileEvent

logger = logging.getLogger(__name__)


class Debouncer:
    """
    Async debouncer that batches file events within a configurable time window.

    Events are collected and merged into a batch. When no new events arrive
    for the debounce delay period, the batch is emitted via the callback.

    Attributes:
        delay_ms: Debounce delay in milliseconds
        on_batch_ready: Async callback invoked when a batch is ready
    """

    def __init__(
        self,
        delay_ms: int = 2000,
        on_batch_ready: Callable[[DebouncedBatch], Awaitable[None]] | None = None,
    ):
        """
        Initialize the debouncer.

        Args:
            delay_ms: Debounce delay in milliseconds (default: 2000)
            on_batch_ready: Async callback to invoke when batch is ready
        """
        self._delay_ms = delay_ms
        self._on_batch_ready = on_batch_ready
        self._pending = DebouncedBatch()
        self._timer_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    @property
    def delay_ms(self) -> int:
        """Get the debounce delay in milliseconds."""
        return self._delay_ms

    async def add_event(self, event: FileEvent) -> None:
        """
        Add an event to the debounce queue.

        Resets the debounce timer each time an event is added.

        Args:
            event: The file event to add
        """
        async with self._lock:
            # Merge event into pending batch
            self._pending.merge(event)

            # Cancel existing timer if any
            if self._timer_task is not None and not self._timer_task.done():
                self._timer_task.cancel()
                try:
                    await self._timer_task
                except asyncio.CancelledError:
                    pass

            # Start new timer
            self._timer_task = asyncio.create_task(self._timer_callback())

    async def _timer_callback(self) -> None:
        """Internal timer callback that fires after debounce delay."""
        try:
            await asyncio.sleep(self._delay_ms / 1000.0)
            await self._emit_batch()
        except asyncio.CancelledError:
            # Timer was cancelled, this is expected
            pass

    async def _emit_batch(self) -> None:
        """Emit the current batch via callback and reset."""
        async with self._lock:
            if self._pending.is_empty():
                return

            # Copy and clear pending batch
            batch = self._pending.copy()
            self._pending.clear()

        # Invoke callback outside of lock
        if self._on_batch_ready is not None:
            try:
                await self._on_batch_ready(batch)
            except Exception as e:
                logger.error(f"Error in batch callback: {e}")

    async def flush(self) -> DebouncedBatch:
        """
        Immediately flush all pending events and return the batch.

        Cancels any pending timer and returns the current batch.

        Returns:
            The batch of pending events (may be empty)
        """
        async with self._lock:
            # Cancel timer if running
            if self._timer_task is not None and not self._timer_task.done():
                self._timer_task.cancel()
                try:
                    await self._timer_task
                except asyncio.CancelledError:
                    pass
                self._timer_task = None

            # Copy and clear pending batch
            batch = self._pending.copy()
            self._pending.clear()

        # Invoke callback if batch is not empty
        if not batch.is_empty() and self._on_batch_ready is not None:
            try:
                await self._on_batch_ready(batch)
            except Exception as e:
                logger.error(f"Error in batch callback during flush: {e}")

        return batch

    def get_pending_count(self) -> int:
        """
        Get the number of pending events.

        Returns:
            Total count of events in the pending batch
        """
        return self._pending.total_count()

    def has_pending(self) -> bool:
        """
        Check if there are pending events.

        Returns:
            True if there are pending events, False otherwise
        """
        return not self._pending.is_empty()

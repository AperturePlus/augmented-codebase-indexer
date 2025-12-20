"""
Event loop management for the ACI REPL.

Provides a persistent event loop manager that maintains a single event loop
throughout the REPL session lifecycle, with proper cleanup when switching
codebases and error recovery for closed loop scenarios.
"""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class EventLoopManager:
    """
    Manages a persistent event loop for REPL async operations.

    Instead of creating a new event loop for each async command via
    asyncio.run(), this manager maintains a single event loop throughout
    the REPL session. This prevents "Event loop is closed" errors that
    occur when executing multiple async commands in succession.

    The event loop can be reset when switching codebases, and will
    automatically recover from closed loop errors by creating a new loop.

    Attributes:
        _loop: The managed asyncio event loop.
    """

    def __init__(self) -> None:
        """Initialize with a new event loop."""
        self._loop: asyncio.AbstractEventLoop = self._create_new_loop()

    def _create_new_loop(self) -> asyncio.AbstractEventLoop:
        """
        Create a new event loop.

        Returns:
            A new asyncio event loop.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    def run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """
        Execute a coroutine on the managed event loop.

        If the current loop is closed, automatically creates a new one
        and retries the operation once.

        Args:
            coro: The coroutine to execute.

        Returns:
            The result of the coroutine execution.

        Raises:
            RuntimeError: If the operation fails after retry.
        """
        try:
            if self._loop.is_closed():
                self._loop = self._create_new_loop()
            return self._loop.run_until_complete(coro)
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Recovery: create new loop and retry once
                self._loop = self._create_new_loop()
                return self._loop.run_until_complete(coro)
            raise

    def reset(self) -> None:
        """
        Close current loop and create a new one.

        Used when switching codebases to ensure a clean state.
        Gracefully closes the current loop before creating a new one.
        """
        if not self._loop.is_closed():
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()

                # Run until all tasks are cancelled
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )

                self._loop.close()
            except Exception:
                # If closing fails, just proceed to create new loop
                pass

        self._loop = self._create_new_loop()

    def close(self) -> None:
        """
        Close the event loop for REPL shutdown.

        Should be called when the REPL session ends to release resources.
        """
        if not self._loop.is_closed():
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()

                # Run until all tasks are cancelled
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )

                self._loop.close()
            except Exception:
                # Best effort cleanup
                pass

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """
        Get the current event loop.

        Returns:
            The managed asyncio event loop.
        """
        return self._loop

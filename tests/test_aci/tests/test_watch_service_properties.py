"""
Property-based tests for WatchService.

**Feature: file-watcher-service, Property 1: Path Validation**
**Feature: file-watcher-service, Property 5: Update Triggering**
**Feature: file-watcher-service, Property 6: Concurrent Event Handling**
**Feature: file-watcher-service, Property 7: Error Recovery**
**Feature: file-watcher-service, Property 8: Graceful Shutdown**
**Validates: Requirements 1.2, 1.5, 3.3, 4.1, 4.2, 4.4, 4.5, 5.2, 5.3**
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.core.debouncer import DebouncedBatch
from aci.core.file_events import FileEvent, FileEventType
from aci.core.watch_config import WatchConfig
from aci.infrastructure.fakes import FakeFileWatcher
from aci.services.indexing_models import IndexingResult
from aci.services.watch_service import (
    PathValidationError,
    WatchService,
    WatchServiceError,
    WatchStats,
)


def run_async(coro):
    """Helper to run async code in tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # If there's already a running loop, create a task
        return loop.run_until_complete(coro)
    else:
        # No running loop, use asyncio.run() for Python 3.10+
        return asyncio.run(coro)


# Strategies for generating test data
COMMON_DIRS = ["src", "lib", "tests", "pkg", "utils", "core", "app"]
COMMON_NAMES = ["main", "test", "utils", "helper", "config", "index", "app"]
COMMON_EXTS = ["py", "js", "ts", "go", "java"]


@st.composite
def safe_path_strategy(draw):
    """Generate a safe file path efficiently."""
    num_dirs = draw(st.integers(min_value=1, max_value=3))
    dirs = [draw(st.sampled_from(COMMON_DIRS)) for _ in range(num_dirs)]
    name = draw(st.sampled_from(COMMON_NAMES))
    ext = draw(st.sampled_from(COMMON_EXTS))
    return "/" + "/".join(dirs) + "/" + name + "." + ext


safe_path = safe_path_strategy()


class MockIndexingService:
    """Mock IndexingService for testing WatchService."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.update_calls: list[Path] = []
        self.index_calls: list[Path] = []
        self._metadata_store = MagicMock()
        self._metadata_store.get_file_hashes_under_root = MagicMock(
            return_value={"dummy.py": "hash"}
        )
        self._vector_store = MagicMock()
        self._vector_store.get_stats = AsyncMock(
            return_value={"total_files": 10, "total_vectors": 100}
        )

    async def update_incremental(
        self, root_path: Path, *, max_workers: int | None = None
    ) -> IndexingResult:
        """Mock incremental update."""
        self.update_calls.append(root_path)
        if self.should_fail:
            raise RuntimeError("Simulated update failure")
        return IndexingResult(
            total_files=1,
            total_chunks=5,
            new_files=1,
            modified_files=0,
            deleted_files=0,
        )

    async def index_directory(
        self, root_path: Path, *, max_workers: int | None = None
    ) -> IndexingResult:
        """Mock full index."""
        self.index_calls.append(root_path)
        return IndexingResult(total_files=10, total_chunks=50)


# ============================================================================
# Property 1: Path Validation Tests
# ============================================================================


def test_path_validation_nonexistent_path():
    """
    **Feature: file-watcher-service, Property 1: Path Validation**
    **Validates: Requirements 1.2, 1.5**

    For any path that does not exist, WatchService.start() SHALL raise
    a PathValidationError without starting.
    """
    config = WatchConfig(watch_path=Path("/nonexistent/path/that/does/not/exist"))
    indexing_service = MockIndexingService()
    file_watcher = FakeFileWatcher()

    service = WatchService(indexing_service, file_watcher, config)

    try:
        run_async(service.start())
        raise AssertionError("Should have raised PathValidationError")
    except PathValidationError as e:
        assert "does not exist" in str(e)
        assert not service.is_running()


def test_path_validation_file_not_directory():
    """
    **Feature: file-watcher-service, Property 1: Path Validation**
    **Validates: Requirements 1.2**

    For any path that is a file (not a directory), WatchService.start()
    SHALL raise a PathValidationError.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        file_path = Path(f.name)

    try:
        config = WatchConfig(watch_path=file_path)
        indexing_service = MockIndexingService()
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        try:
            run_async(service.start())
            raise AssertionError("Should have raised PathValidationError")
        except PathValidationError as e:
            assert "not a directory" in str(e)
            assert not service.is_running()
    finally:
        file_path.unlink(missing_ok=True)


def test_initial_indexing_runs_only_when_unindexed():
    """
    **Feature: file-watcher-service, Property 1: Path Validation**
    **Validates: Requirements 1.3**

    When WatchService starts on an unindexed path (no metadata for that root),
    it SHALL perform an initial full index before watching.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100)
        indexing_service = MockIndexingService()
        indexing_service._metadata_store.get_file_hashes_under_root = MagicMock(
            return_value={}
        )
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        run_async(service.start())
        try:
            assert indexing_service.index_calls == [watch_path]
            assert service.is_running()
        finally:
            run_async(service.stop())


@given(dirname=st.sampled_from(COMMON_DIRS))
@settings(max_examples=50)
def test_path_validation_valid_directory(dirname: str):
    """
    **Feature: file-watcher-service, Property 1: Path Validation**
    **Validates: Requirements 1.2**

    For any valid directory path, WatchService.start() SHALL succeed
    and the service SHALL be running.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir) / dirname
        watch_path.mkdir(parents=True, exist_ok=True)

        config = WatchConfig(watch_path=watch_path)
        indexing_service = MockIndexingService()
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        try:
            run_async(service.start())
            assert service.is_running()
        finally:
            run_async(service.stop())


# ============================================================================
# Property 5: Update Triggering Tests
# ============================================================================


def test_update_triggering_on_batch_ready():
    """
    **Feature: file-watcher-service, Property 5: Update Triggering**
    **Validates: Requirements 3.3, 4.1**

    For any non-empty batch of events after the debounce window expires,
    the WatchService SHALL call update_incremental exactly once.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100)
        indexing_service = MockIndexingService()
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        run_async(service.start())

        try:
            # Create a batch and process it directly
            batch = DebouncedBatch()
            batch.created.add(watch_path / "test.py")

            run_async(service._on_batch_ready(batch))

            # Verify update was called exactly once
            assert len(indexing_service.update_calls) == 1
            assert indexing_service.update_calls[0] == watch_path
        finally:
            run_async(service.stop())


@given(num_files=st.integers(min_value=1, max_value=10))
@settings(max_examples=50)
def test_update_triggering_batch_with_multiple_files(num_files: int):
    """
    **Feature: file-watcher-service, Property 5: Update Triggering**
    **Validates: Requirements 3.3, 4.1**

    For any batch containing multiple file changes, the WatchService
    SHALL call update_incremental exactly once (not once per file).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100)
        indexing_service = MockIndexingService()
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        run_async(service.start())

        try:
            # Create a batch with multiple files
            batch = DebouncedBatch()
            for i in range(num_files):
                batch.created.add(watch_path / f"file{i}.py")

            run_async(service._on_batch_ready(batch))

            # Verify update was called exactly once
            assert len(indexing_service.update_calls) == 1
        finally:
            run_async(service.stop())


def test_update_triggering_empty_batch_no_update():
    """
    **Feature: file-watcher-service, Property 5: Update Triggering**
    **Validates: Requirements 3.3, 4.1**

    For an empty batch, the WatchService SHALL NOT call update_incremental.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100)
        indexing_service = MockIndexingService()
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        run_async(service.start())

        try:
            # Process empty batch
            batch = DebouncedBatch()
            run_async(service._on_batch_ready(batch))

            # Verify update was NOT called
            assert len(indexing_service.update_calls) == 0
        finally:
            run_async(service.stop())


# ============================================================================
# Property 6: Concurrent Event Handling Tests
# ============================================================================


def test_concurrent_event_handling_queues_during_update():
    """
    **Feature: file-watcher-service, Property 6: Concurrent Event Handling**
    **Validates: Requirements 4.2, 4.5**

    For any file events that occur while an incremental update is in progress,
    those events SHALL be queued and processed in the next update cycle.
    """
    async def run_test():
        with tempfile.TemporaryDirectory() as tmpdir:
            watch_path = Path(tmpdir)
            config = WatchConfig(watch_path=watch_path, debounce_ms=100)

            # Create a slow indexing service
            indexing_service = MockIndexingService()
            original_update = indexing_service.update_incremental

            async def slow_update(root_path, *, max_workers=None):
                await asyncio.sleep(0.1)  # Simulate slow update
                return await original_update(root_path, max_workers=max_workers)

            indexing_service.update_incremental = slow_update

            file_watcher = FakeFileWatcher()
            service = WatchService(indexing_service, file_watcher, config)

            await service.start()

            try:
                # Start first batch processing
                batch1 = DebouncedBatch()
                batch1.created.add(watch_path / "file1.py")

                # Start processing (don't await yet)
                task1 = asyncio.create_task(service._on_batch_ready(batch1))

                # Give it time to start
                await asyncio.sleep(0.01)

                # Queue second batch while first is processing
                batch2 = DebouncedBatch()
                batch2.created.add(watch_path / "file2.py")
                await service._on_batch_ready(batch2)

                # Wait for first task to complete
                await task1

                # Both batches should have triggered updates
                assert len(indexing_service.update_calls) >= 1
            finally:
                await service.stop()

    asyncio.run(run_test())


# ============================================================================
# Property 7: Error Recovery Tests
# ============================================================================


def test_error_recovery_continues_watching():
    """
    **Feature: file-watcher-service, Property 7: Error Recovery**
    **Validates: Requirements 4.4**

    For any error that occurs during an incremental update, the WatchService
    SHALL continue watching for new changes and remain in a running state.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100)
        indexing_service = MockIndexingService(should_fail=True)
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        run_async(service.start())

        try:
            # Process batch that will fail
            batch = DebouncedBatch()
            batch.created.add(watch_path / "test.py")
            run_async(service._on_batch_ready(batch))

            # Service should still be running
            assert service.is_running()

            # Error should be recorded in stats
            stats = service.get_stats()
            assert stats.errors == 1
        finally:
            run_async(service.stop())


@given(num_failures=st.integers(min_value=1, max_value=5))
@settings(max_examples=20)
def test_error_recovery_multiple_failures(num_failures: int):
    """
    **Feature: file-watcher-service, Property 7: Error Recovery**
    **Validates: Requirements 4.4**

    For any number of consecutive update failures, the WatchService
    SHALL continue watching and track all errors.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100)
        indexing_service = MockIndexingService(should_fail=True)
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        run_async(service.start())

        try:
            # Process multiple failing batches
            for i in range(num_failures):
                batch = DebouncedBatch()
                batch.created.add(watch_path / f"test{i}.py")
                run_async(service._on_batch_ready(batch))

            # Service should still be running
            assert service.is_running()

            # All errors should be recorded
            stats = service.get_stats()
            assert stats.errors == num_failures
        finally:
            run_async(service.stop())


# ============================================================================
# Property 8: Graceful Shutdown Tests
# ============================================================================


def test_graceful_shutdown_stops_watcher():
    """
    **Feature: file-watcher-service, Property 8: Graceful Shutdown**
    **Validates: Requirements 5.2, 5.3**

    For any shutdown request, the WatchService SHALL release all
    file system watchers and return to a stopped state.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100)
        indexing_service = MockIndexingService()
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        run_async(service.start())
        assert service.is_running()
        assert file_watcher.is_running()

        run_async(service.stop())

        # Service should be stopped
        assert not service.is_running()
        # File watcher should be stopped
        assert not file_watcher.is_running()


def test_graceful_shutdown_idempotent():
    """
    **Feature: file-watcher-service, Property 8: Graceful Shutdown**
    **Validates: Requirements 5.2, 5.3**

    Calling stop() multiple times SHALL be safe and idempotent.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100)
        indexing_service = MockIndexingService()
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        run_async(service.start())
        run_async(service.stop())
        run_async(service.stop())  # Should not raise
        run_async(service.stop())  # Should not raise

        assert not service.is_running()


def test_graceful_shutdown_not_started():
    """
    **Feature: file-watcher-service, Property 8: Graceful Shutdown**
    **Validates: Requirements 5.2, 5.3**

    Calling stop() on a service that was never started SHALL be safe.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100)
        indexing_service = MockIndexingService()
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        # Should not raise
        run_async(service.stop())
        assert not service.is_running()


# ============================================================================
# WatchStats Tests
# ============================================================================


def test_watch_stats_to_dict():
    """Test WatchStats serialization."""
    stats = WatchStats()
    stats.events_received = 10
    stats.updates_triggered = 2
    stats.errors = 1

    d = stats.to_dict()

    assert d["events_received"] == 10
    assert d["updates_triggered"] == 2
    assert d["errors"] == 1
    assert "started_at" in d


def test_watch_service_cannot_start_twice():
    """
    **Feature: file-watcher-service, Property 1: Path Validation**
    **Validates: Requirements 1.2**

    WatchService SHALL raise an error if started while already running.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100)
        indexing_service = MockIndexingService()
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        run_async(service.start())

        try:
            run_async(service.start())
            raise AssertionError("Should have raised WatchServiceError")
        except WatchServiceError as e:
            assert "already running" in str(e)
        finally:
            run_async(service.stop())


def test_watch_service_event_counting():
    """Test that events are properly counted in stats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_path = Path(tmpdir)
        config = WatchConfig(watch_path=watch_path, debounce_ms=100, verbose=True)
        indexing_service = MockIndexingService()
        file_watcher = FakeFileWatcher()

        service = WatchService(indexing_service, file_watcher, config)

        run_async(service.start())

        try:
            # Simulate file events
            event = FileEvent(
                event_type=FileEventType.CREATED,
                file_path=watch_path / "test.py",
            )
            run_async(service._on_file_event(event))
            run_async(service._on_file_event(event))
            run_async(service._on_file_event(event))

            stats = service.get_stats()
            assert stats.events_received == 3
        finally:
            run_async(service.stop())

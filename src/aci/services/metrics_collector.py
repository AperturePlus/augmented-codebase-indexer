"""
Metrics Collector service for indexing observability.

Provides in-memory metrics collection for embedding latency, Qdrant operations,
and indexing statistics. Supports percentile calculations (p50, p95).
"""

from dataclasses import dataclass, field
from datetime import datetime


def calculate_percentile(values: list[float], percentile: float) -> float:
    """
    Calculate the given percentile of a list of values.

    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)

    Returns:
        The percentile value, or 0.0 if the list is empty
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    # Use linear interpolation for percentile calculation
    index = (percentile / 100.0) * (n - 1)
    lower_idx = int(index)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = index - lower_idx

    return sorted_values[lower_idx] + fraction * (
        sorted_values[upper_idx] - sorted_values[lower_idx]
    )


@dataclass
class IndexingMetrics:
    """
    Container for indexing operation metrics.

    Stores recent latency measurements and cumulative totals for
    embedding operations, Qdrant calls, and overall indexing statistics.
    """

    embedding_latency_ms: list[float] = field(default_factory=list)
    qdrant_call_duration_ms: list[float] = field(default_factory=list)
    chunks_indexed_total: int = 0
    files_indexed_total: int = 0
    last_update_duration_seconds: float = 0.0
    last_update_timestamp: datetime | None = None


@dataclass
class WatchMetrics:
    """
    Container for watch service metrics.

    Tracks file events received, updates triggered, and error counts
    for monitoring the file watcher service.
    """

    events_received: int = 0
    events_by_type: dict[str, int] = field(default_factory=dict)
    updates_triggered: int = 0
    update_duration_ms: list[float] = field(default_factory=list)
    files_processed_total: int = 0
    chunks_processed_total: int = 0
    errors: int = 0
    last_event_timestamp: datetime | None = None
    last_update_timestamp: datetime | None = None



class MetricsCollector:
    """
    Lightweight in-memory metrics collector for indexing and watch operations.

    Tracks embedding latencies, Qdrant operation durations, indexing
    statistics, and watch service metrics. Provides aggregated metrics
    with percentile calculations.
    """

    # Maximum number of latency samples to retain
    MAX_LATENCY_SAMPLES = 1000

    def __init__(self) -> None:
        """Initialize the metrics collector with empty metrics."""
        self._metrics = IndexingMetrics()
        self._watch_metrics = WatchMetrics()

    def record_embedding_latency(self, latency_ms: float) -> None:
        """
        Record an embedding API call latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._metrics.embedding_latency_ms.append(latency_ms)
        # Trim to max samples to prevent unbounded memory growth
        if len(self._metrics.embedding_latency_ms) > self.MAX_LATENCY_SAMPLES:
            self._metrics.embedding_latency_ms = self._metrics.embedding_latency_ms[
                -self.MAX_LATENCY_SAMPLES :
            ]

    def record_qdrant_duration(self, duration_ms: float, chunk_count: int) -> None:
        """
        Record a Qdrant upsert operation duration.

        Args:
            duration_ms: Operation duration in milliseconds
            chunk_count: Number of chunks in the operation
        """
        self._metrics.qdrant_call_duration_ms.append(duration_ms)
        self._metrics.chunks_indexed_total += chunk_count
        # Trim to max samples to prevent unbounded memory growth
        if len(self._metrics.qdrant_call_duration_ms) > self.MAX_LATENCY_SAMPLES:
            self._metrics.qdrant_call_duration_ms = (
                self._metrics.qdrant_call_duration_ms[-self.MAX_LATENCY_SAMPLES :]
            )

    def record_indexing_complete(
        self, chunks: int, files: int, duration_s: float
    ) -> None:
        """
        Record completion of an indexing run.

        Args:
            chunks: Total chunks indexed in this run
            files: Total files indexed in this run
            duration_s: Total duration in seconds
        """
        self._metrics.files_indexed_total += files
        self._metrics.last_update_duration_seconds = duration_s
        self._metrics.last_update_timestamp = datetime.now()

    def get_metrics(self) -> dict:
        """
        Get aggregated metrics as a dictionary.

        Returns:
            Dictionary containing:
            - embedding_latency_ms: {avg, p50, p95, count}
            - qdrant_call_duration_ms: {avg, p50, p95, count}
            - chunks_indexed_total: int
            - files_indexed_total: int
            - update_duration_seconds: float
            - last_update_timestamp: ISO format string or None
            - watch: Watch service metrics (if any events recorded)
        """
        embedding_latencies = self._metrics.embedding_latency_ms
        qdrant_durations = self._metrics.qdrant_call_duration_ms

        result = {
            "embedding_latency_ms": {
                "avg": (
                    sum(embedding_latencies) / len(embedding_latencies)
                    if embedding_latencies
                    else 0.0
                ),
                "p50": calculate_percentile(embedding_latencies, 50),
                "p95": calculate_percentile(embedding_latencies, 95),
                "count": len(embedding_latencies),
            },
            "qdrant_call_duration_ms": {
                "avg": (
                    sum(qdrant_durations) / len(qdrant_durations)
                    if qdrant_durations
                    else 0.0
                ),
                "p50": calculate_percentile(qdrant_durations, 50),
                "p95": calculate_percentile(qdrant_durations, 95),
                "count": len(qdrant_durations),
            },
            "chunks_indexed_total": self._metrics.chunks_indexed_total,
            "files_indexed_total": self._metrics.files_indexed_total,
            "update_duration_seconds": self._metrics.last_update_duration_seconds,
            "last_update_timestamp": (
                self._metrics.last_update_timestamp.isoformat()
                if self._metrics.last_update_timestamp
                else None
            ),
        }

        # Include watch metrics if any events have been recorded
        if self._watch_metrics.events_received > 0:
            result["watch"] = self.get_watch_metrics()

        return result

    def record_watch_event(self, event_type: str) -> None:
        """
        Record a file watch event.

        Args:
            event_type: Type of event (created, modified, deleted, moved)
        """
        self._watch_metrics.events_received += 1
        self._watch_metrics.events_by_type[event_type] = (
            self._watch_metrics.events_by_type.get(event_type, 0) + 1
        )
        self._watch_metrics.last_event_timestamp = datetime.now()

    def record_watch_update(
        self, duration_ms: float, files_processed: int, chunks_processed: int
    ) -> None:
        """
        Record a watch-triggered incremental update.

        Args:
            duration_ms: Update duration in milliseconds
            files_processed: Number of files processed in the update
            chunks_processed: Number of chunks processed in the update
        """
        self._watch_metrics.updates_triggered += 1
        self._watch_metrics.update_duration_ms.append(duration_ms)
        self._watch_metrics.files_processed_total += files_processed
        self._watch_metrics.chunks_processed_total += chunks_processed
        self._watch_metrics.last_update_timestamp = datetime.now()

        # Trim to max samples to prevent unbounded memory growth
        if len(self._watch_metrics.update_duration_ms) > self.MAX_LATENCY_SAMPLES:
            self._watch_metrics.update_duration_ms = (
                self._watch_metrics.update_duration_ms[-self.MAX_LATENCY_SAMPLES:]
            )

    def record_watch_error(self) -> None:
        """Record a watch service error."""
        self._watch_metrics.errors += 1

    def get_watch_metrics(self) -> dict:
        """
        Get aggregated watch metrics as a dictionary.

        Returns:
            Dictionary containing:
            - events_received: Total events received
            - events_by_type: Breakdown by event type
            - updates_triggered: Total updates triggered
            - update_duration_ms: {avg, p50, p95, count}
            - files_processed_total: Total files processed
            - chunks_processed_total: Total chunks processed
            - errors: Total error count
            - last_event_timestamp: ISO format string or None
            - last_update_timestamp: ISO format string or None
        """
        update_durations = self._watch_metrics.update_duration_ms

        return {
            "events_received": self._watch_metrics.events_received,
            "events_by_type": dict(self._watch_metrics.events_by_type),
            "updates_triggered": self._watch_metrics.updates_triggered,
            "update_duration_ms": {
                "avg": (
                    sum(update_durations) / len(update_durations)
                    if update_durations
                    else 0.0
                ),
                "p50": calculate_percentile(update_durations, 50),
                "p95": calculate_percentile(update_durations, 95),
                "count": len(update_durations),
            },
            "files_processed_total": self._watch_metrics.files_processed_total,
            "chunks_processed_total": self._watch_metrics.chunks_processed_total,
            "errors": self._watch_metrics.errors,
            "last_event_timestamp": (
                self._watch_metrics.last_event_timestamp.isoformat()
                if self._watch_metrics.last_event_timestamp
                else None
            ),
            "last_update_timestamp": (
                self._watch_metrics.last_update_timestamp.isoformat()
                if self._watch_metrics.last_update_timestamp
                else None
            ),
        }

    def reset(self) -> None:
        """Reset all metrics to initial state. Useful for testing."""
        self._metrics = IndexingMetrics()
        self._watch_metrics = WatchMetrics()

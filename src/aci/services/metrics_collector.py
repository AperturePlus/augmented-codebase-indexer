"""
Metrics Collector service for indexing observability.

Provides in-memory metrics collection for embedding latency, Qdrant operations,
and indexing statistics. Supports percentile calculations (p50, p95).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


def calculate_percentile(values: List[float], percentile: float) -> float:
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

    embedding_latency_ms: List[float] = field(default_factory=list)
    qdrant_call_duration_ms: List[float] = field(default_factory=list)
    chunks_indexed_total: int = 0
    files_indexed_total: int = 0
    last_update_duration_seconds: float = 0.0
    last_update_timestamp: Optional[datetime] = None



class MetricsCollector:
    """
    Lightweight in-memory metrics collector for indexing operations.

    Tracks embedding latencies, Qdrant operation durations, and indexing
    statistics. Provides aggregated metrics with percentile calculations.
    """

    # Maximum number of latency samples to retain
    MAX_LATENCY_SAMPLES = 1000

    def __init__(self) -> None:
        """Initialize the metrics collector with empty metrics."""
        self._metrics = IndexingMetrics()

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

    def get_metrics(self) -> Dict:
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
        """
        embedding_latencies = self._metrics.embedding_latency_ms
        qdrant_durations = self._metrics.qdrant_call_duration_ms

        return {
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

    def reset(self) -> None:
        """Reset all metrics to initial state. Useful for testing."""
        self._metrics = IndexingMetrics()

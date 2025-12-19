"""
Property-based tests for Metrics collection in Indexing Observability.

Tests for metrics endpoint field presence (Property 6).
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.services.metrics_collector import MetricsCollector


# Strategies for metrics tests
latency_strategy = st.floats(min_value=0.0, max_value=10000.0, allow_nan=False)
chunk_count_strategy = st.integers(min_value=0, max_value=1000)
file_count_strategy = st.integers(min_value=0, max_value=100)
duration_strategy = st.floats(min_value=0.0, max_value=3600.0, allow_nan=False)


@given(
    embedding_latencies=st.lists(latency_strategy, min_size=0, max_size=50),
    qdrant_durations=st.lists(latency_strategy, min_size=0, max_size=50),
    qdrant_chunk_counts=st.lists(chunk_count_strategy, min_size=0, max_size=50),
    indexing_chunks=chunk_count_strategy,
    indexing_files=file_count_strategy,
    indexing_duration=duration_strategy,
)
@settings(max_examples=100, deadline=None)
def test_metrics_endpoint_field_presence(
    embedding_latencies: list[float],
    qdrant_durations: list[float],
    qdrant_chunk_counts: list[int],
    indexing_chunks: int,
    indexing_files: int,
    indexing_duration: float,
):
    """
    **Feature: indexing-observability, Property 6: Metrics endpoint field presence**
    **Validates: Requirements 2.5**

    *For any* call to the `/metrics` endpoint (via MetricsCollector.get_metrics()),
    the response should contain `embedding_latency_ms`, `qdrant_call_duration_ms`,
    `chunks_indexed_total`, and `update_duration_seconds` fields.
    """
    collector = MetricsCollector()

    for latency in embedding_latencies:
        collector.record_embedding_latency(latency)

    min_len = min(len(qdrant_durations), len(qdrant_chunk_counts))
    for i in range(min_len):
        collector.record_qdrant_duration(qdrant_durations[i], qdrant_chunk_counts[i])

    collector.record_indexing_complete(indexing_chunks, indexing_files, indexing_duration)

    metrics = collector.get_metrics()

    required_fields = [
        "embedding_latency_ms",
        "qdrant_call_duration_ms",
        "chunks_indexed_total",
        "update_duration_seconds",
    ]

    for field in required_fields:
        assert field in metrics, f"Required field '{field}' missing from metrics response"

    embedding_metrics = metrics["embedding_latency_ms"]
    assert "avg" in embedding_metrics, "embedding_latency_ms should have 'avg' field"
    assert "p50" in embedding_metrics, "embedding_latency_ms should have 'p50' field"
    assert "p95" in embedding_metrics, "embedding_latency_ms should have 'p95' field"
    assert "count" in embedding_metrics, "embedding_latency_ms should have 'count' field"

    qdrant_metrics = metrics["qdrant_call_duration_ms"]
    assert "avg" in qdrant_metrics, "qdrant_call_duration_ms should have 'avg' field"
    assert "p50" in qdrant_metrics, "qdrant_call_duration_ms should have 'p50' field"
    assert "p95" in qdrant_metrics, "qdrant_call_duration_ms should have 'p95' field"
    assert "count" in qdrant_metrics, "qdrant_call_duration_ms should have 'count' field"

    assert metrics["chunks_indexed_total"] >= 0, "chunks_indexed_total should be non-negative"
    assert metrics["update_duration_seconds"] >= 0, "update_duration_seconds should be non-negative"

    assert embedding_metrics["count"] == len(embedding_latencies), (
        f"embedding count {embedding_metrics['count']} should match recorded {len(embedding_latencies)}"
    )
    assert qdrant_metrics["count"] == min_len, (
        f"qdrant count {qdrant_metrics['count']} should match recorded {min_len}"
    )

"""
Fake implementations for testing.

Provides in-memory implementations of infrastructure interfaces
for use in unit and integration tests without external dependencies.
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from aci.infrastructure.embedding import EmbeddingClientInterface
from aci.infrastructure.vector_store import SearchResult, VectorStoreInterface

if TYPE_CHECKING:
    from aci.core.file_events import FileEvent


class InMemoryVectorStore(VectorStoreInterface):
    """
    In-memory vector store for testing.

    Implements VectorStoreInterface without requiring Qdrant.
    Uses cosine similarity for search.
    Supports multiple collections for isolation testing.
    """

    def __init__(self, vector_size: int = 1536, collection_name: str = "default"):
        """
        Initialize in-memory store.

        Args:
            vector_size: Expected dimension of vectors
            collection_name: Default collection name
        """
        self._vector_size = vector_size
        self._collection_name = collection_name
        # Multi-collection storage: collection_name -> {chunk_id -> data}
        self._collections: dict[str, dict[str, tuple[list[float], dict]]] = {}
        # Legacy single-collection references for backward compatibility
        self._vectors: dict[str, list[float]] = {}
        self._payloads: dict[str, dict] = {}

    async def upsert(
        self,
        chunk_id: str,
        vector: list[float],
        payload: dict,
        collection_name: str | None = None,
    ) -> None:
        """Insert or update a vector with its payload."""
        target_collection = collection_name or self._collection_name
        # Ensure artifact_type is set, default to "chunk" for backward compatibility
        if "artifact_type" not in payload:
            payload = {**payload, "artifact_type": "chunk"}
        self._vectors[chunk_id] = vector
        self._payloads[chunk_id] = payload
        # Also store in collection-based storage
        if target_collection not in self._collections:
            self._collections[target_collection] = {}
        self._collections[target_collection][chunk_id] = (vector, payload)

    async def upsert_batch(
        self,
        points: list[tuple[str, list[float], dict]],
        collection_name: str | None = None,
    ) -> None:
        """Batch insert or update vectors."""
        for chunk_id, vector, payload in points:
            await self.upsert(chunk_id, vector, payload, collection_name=collection_name)

    async def delete_by_file(self, file_path: str, collection_name: str | None = None) -> int:
        """Delete all vectors for a file, return count deleted."""
        target_collection = collection_name or self._collection_name
        collection_data = self._collections.get(target_collection, {})
        to_delete = [
            cid for cid, (_, payload) in collection_data.items()
            if payload.get("file_path") == file_path
        ]
        for cid in to_delete:
            del collection_data[cid]
            # Also clean up legacy storage
            if cid in self._vectors:
                del self._vectors[cid]
            if cid in self._payloads:
                del self._payloads[cid]
        return len(to_delete)

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        file_filter: str | None = None,
        collection_name: str | None = None,
        artifact_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar vectors using cosine similarity.

        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            file_filter: Optional glob pattern for file paths
            collection_name: Optional collection to search. If provided, searches
                that collection without modifying instance state. If None, uses
                the instance's default collection.
            artifact_types: Optional list of artifact types to filter by
                (e.g., ["chunk", "function_summary", "class_summary", "file_summary"]).
                If None, returns all artifact types.

        Returns:
            List of SearchResult sorted by score descending
        """
        results = []

        # Use provided collection or fall back to instance default
        target_collection = collection_name or self._collection_name

        # Get data from the target collection
        collection_data = self._collections.get(target_collection, {})

        for chunk_id, (vector, payload) in collection_data.items():
            file_path = payload.get("file_path", "")

            # Apply file filter if specified
            if file_filter and not fnmatch.fnmatch(file_path, file_filter):
                continue

            # Apply artifact type filter if specified
            if artifact_types:
                artifact_type = payload.get("artifact_type", "chunk")
                if artifact_type not in artifact_types:
                    continue

            # Calculate cosine similarity
            score = self._cosine_similarity(query_vector, vector)

            # Ensure artifact_type is included in metadata, default to "chunk"
            artifact_type = payload.get("artifact_type", "chunk")
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    start_line=payload.get("start_line", 0),
                    end_line=payload.get("end_line", 0),
                    content=payload.get("content", ""),
                    score=score,
                    metadata={
                        **{
                            k: v
                            for k, v in payload.items()
                            if k not in ("file_path", "start_line", "end_line", "content")
                        },
                        "artifact_type": artifact_type,
                    },
                )
            )

        # Sort by score descending and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def get_stats(self, collection_name: str | None = None) -> dict:
        """
        Get storage statistics.

        Args:
            collection_name: Optional collection to get stats for. If provided,
                returns stats for that collection without modifying instance state.
                If None, uses the instance's default collection.

        Returns:
            Dictionary with storage statistics
        """
        # Use provided collection or fall back to instance default
        target_collection = collection_name or self._collection_name
        collection_data = self._collections.get(target_collection, {})

        unique_files = {
            payload.get("file_path", "")
            for _, payload in collection_data.values()
        }
        return {
            "total_vectors": len(collection_data),
            "total_files": len(unique_files),
            "collection_name": target_collection,
            "vector_size": self._vector_size,
        }

    async def get_by_id(self, chunk_id: str) -> SearchResult | None:
        """Get a specific chunk by ID."""
        collection_data = self._collections.get(self._collection_name, {})
        if chunk_id not in collection_data:
            return None

        vector, payload = collection_data[chunk_id]
        # Ensure artifact_type is included in metadata, default to "chunk"
        artifact_type = payload.get("artifact_type", "chunk")
        return SearchResult(
            chunk_id=chunk_id,
            file_path=payload.get("file_path", ""),
            start_line=payload.get("start_line", 0),
            end_line=payload.get("end_line", 0),
            content=payload.get("content", ""),
            score=1.0,
            metadata={
                **{
                    k: v
                    for k, v in payload.items()
                    if k not in ("file_path", "start_line", "end_line", "content")
                },
                "artifact_type": artifact_type,
            },
        )

    async def reset(self) -> None:
        """Clear all stored data."""
        self.clear()

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection entirely.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if the collection was deleted, False if it did not exist
        """
        if collection_name in self._collections:
            del self._collections[collection_name]
            return True
        return False

    async def get_all_file_paths(self, collection_name: str | None = None) -> list[str]:
        """
        Get all unique file paths in the store.

        Args:
            collection_name: Optional collection to query. If provided, returns
                file paths from that collection without modifying instance state.
                If None, uses the instance's default collection.

        Returns:
            List of unique file paths
        """
        # Use provided collection or fall back to instance default
        target_collection = collection_name or self._collection_name
        collection_data = self._collections.get(target_collection, {})
        unique_files = {
            payload.get("file_path", "")
            for _, payload in collection_data.values()
        }
        return [f for f in unique_files if f]  # Filter out empty strings

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def clear(self) -> None:
        """Clear all stored data."""
        self._vectors.clear()
        self._payloads.clear()
        self._collections.clear()

    def set_collection(self, collection_name: str) -> None:
        """
        Switch to a different collection.

        Args:
            collection_name: Name of the collection to use.
        """
        self._collection_name = collection_name

    def get_collection_name(self) -> str:
        """Get the current collection name."""
        return self._collection_name


class LocalEmbeddingClient(EmbeddingClientInterface):
    """
    Local embedding client for testing.

    Returns deterministic pseudo-random vectors based on text hash.
    No API calls required.
    """

    def __init__(self, dimension: int = 1536):
        """
        Initialize local embedding client.

        Args:
            dimension: Dimension of embedding vectors to generate
        """
        self._dimension = dimension

    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate deterministic embeddings for texts.

        Uses SHA-256 hash of text to generate reproducible vectors.
        Same text always produces same embedding.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        return [self._text_to_vector(text) for text in texts]

    def _text_to_vector(self, text: str) -> list[float]:
        """
        Convert text to a deterministic vector.

        Uses hash-based approach to generate reproducible vectors
        that have some semantic properties (similar texts have
        somewhat similar vectors due to shared substrings).
        """
        # Get hash of text
        text_hash = hashlib.sha256(text.encode("utf-8")).digest()

        # Generate vector from hash bytes
        # Extend hash if needed to fill dimension
        vector = []
        hash_bytes = text_hash

        while len(vector) < self._dimension:
            for byte in hash_bytes:
                if len(vector) >= self._dimension:
                    break
                # Convert byte to float in range [-1, 1]
                value = (byte / 127.5) - 1.0
                vector.append(value)

            # Generate more bytes if needed
            if len(vector) < self._dimension:
                hash_bytes = hashlib.sha256(hash_bytes).digest()

        # Normalize to unit vector
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def embed_sync(self, text: str) -> list[float]:
        """
        Synchronous embedding for single text.

        Convenience method for testing.
        """
        return self._text_to_vector(text)


class FakeFileWatcher:
    """
    Fake file watcher for testing.

    Allows manual triggering of file events without actual file system monitoring.
    Implements the same interface as FileWatcher for use in tests.
    """

    def __init__(
        self,
        extensions: set[str] | None = None,
        ignore_patterns: list[str] | None = None,
        follow_symlinks: bool = False,
    ):
        """
        Initialize the fake file watcher.

        Args:
            extensions: Set of file extensions to watch (ignored in fake)
            ignore_patterns: List of patterns to ignore (ignored in fake)
            follow_symlinks: Whether to follow symlinks (ignored in fake)
        """


        self._extensions = extensions or {".py", ".js"}
        self._ignore_patterns = ignore_patterns or []
        self._follow_symlinks = follow_symlinks
        self._callback: Callable[[FileEvent], None] | None = None
        self._watch_path: Path | None = None
        self._running = False
        self._events: list[FileEvent] = []

    def start(self, path: Path, callback: Callable[[FileEvent], None]) -> None:
        """
        Start the fake watcher.

        Args:
            path: Directory path to watch
            callback: Function to call when events are triggered
        """
        from pathlib import Path

        if self._running:
            raise RuntimeError("File watcher is already running")

        self._watch_path = Path(path).resolve()
        self._callback = callback
        self._running = True

    def stop(self) -> None:
        """Stop the fake watcher."""
        self._running = False
        self._callback = None
        self._watch_path = None

    def is_running(self) -> bool:
        """Check if the fake watcher is running."""
        return self._running

    def trigger_event(self, event: FileEvent) -> None:
        """
        Manually trigger a file event.

        This is the main testing interface - allows tests to simulate
        file system events without actual file operations.

        Args:
            event: The FileEvent to trigger
        """
        if not self._running:
            raise RuntimeError("File watcher is not running")

        self._events.append(event)
        if self._callback is not None:
            self._callback(event)

    def get_triggered_events(self) -> list[FileEvent]:
        """Get all events that have been triggered."""
        return list(self._events)

    def clear_events(self) -> None:
        """Clear the list of triggered events."""
        self._events.clear()

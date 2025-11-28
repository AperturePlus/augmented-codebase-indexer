"""
Fake implementations for testing.

Provides in-memory implementations of infrastructure interfaces
for use in unit and integration tests without external dependencies.
"""

import fnmatch
import hashlib
import math
from typing import Dict, List, Optional

from aci.infrastructure.embedding_client import EmbeddingClientInterface
from aci.infrastructure.vector_store import SearchResult, VectorStoreInterface


class InMemoryVectorStore(VectorStoreInterface):
    """
    In-memory vector store for testing.

    Implements VectorStoreInterface without requiring Qdrant.
    Uses cosine similarity for search.
    """

    def __init__(self, vector_size: int = 1536):
        """
        Initialize in-memory store.

        Args:
            vector_size: Expected dimension of vectors
        """
        self._vector_size = vector_size
        self._vectors: Dict[str, List[float]] = {}
        self._payloads: Dict[str, dict] = {}

    async def upsert(self, chunk_id: str, vector: List[float], payload: dict) -> None:
        """Insert or update a vector with its payload."""
        self._vectors[chunk_id] = vector
        self._payloads[chunk_id] = payload

    async def upsert_batch(self, points: List[tuple[str, List[float], dict]]) -> None:
        """Batch insert or update vectors."""
        for chunk_id, vector, payload in points:
            self._vectors[chunk_id] = vector
            self._payloads[chunk_id] = payload

    async def delete_by_file(self, file_path: str) -> int:
        """Delete all vectors for a file, return count deleted."""
        to_delete = [
            cid for cid, payload in self._payloads.items() if payload.get("file_path") == file_path
        ]
        for cid in to_delete:
            del self._vectors[cid]
            del self._payloads[cid]
        return len(to_delete)

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        file_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors using cosine similarity.

        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            file_filter: Optional glob pattern for file paths

        Returns:
            List of SearchResult sorted by score descending
        """
        results = []

        for chunk_id, vector in self._vectors.items():
            payload = self._payloads.get(chunk_id, {})
            file_path = payload.get("file_path", "")

            # Apply file filter if specified
            if file_filter and not fnmatch.fnmatch(file_path, file_filter):
                continue

            # Calculate cosine similarity
            score = self._cosine_similarity(query_vector, vector)

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    start_line=payload.get("start_line", 0),
                    end_line=payload.get("end_line", 0),
                    content=payload.get("content", ""),
                    score=score,
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ("file_path", "start_line", "end_line", "content")
                    },
                )
            )

        # Sort by score descending and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def get_stats(self) -> dict:
        """Get storage statistics."""
        unique_files = set(p.get("file_path", "") for p in self._payloads.values())
        return {
            "total_vectors": len(self._vectors),
            "total_files": len(unique_files),
            "collection_name": "in_memory",
            "vector_size": self._vector_size,
        }

    async def get_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """Get a specific chunk by ID."""
        if chunk_id not in self._vectors:
            return None

        payload = self._payloads.get(chunk_id, {})
        return SearchResult(
            chunk_id=chunk_id,
            file_path=payload.get("file_path", ""),
            start_line=payload.get("start_line", 0),
            end_line=payload.get("end_line", 0),
            content=payload.get("content", ""),
            score=1.0,
            metadata={
                k: v
                for k, v in payload.items()
                if k not in ("file_path", "start_line", "end_line", "content")
            },
        )

    async def reset(self) -> None:
        """Clear all stored data."""
        self.clear()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def clear(self) -> None:
        """Clear all stored data."""
        self._vectors.clear()
        self._payloads.clear()


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

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
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

    def _text_to_vector(self, text: str) -> List[float]:
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

    def embed_sync(self, text: str) -> List[float]:
        """
        Synchronous embedding for single text.

        Convenience method for testing.
        """
        return self._text_to_vector(text)

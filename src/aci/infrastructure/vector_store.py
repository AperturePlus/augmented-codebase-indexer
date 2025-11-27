"""
Vector Store for Project ACI.

Provides Qdrant-based vector storage and retrieval for code chunks.
"""

import fnmatch
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    pass


@dataclass
class SearchResult:
    """Search result from vector store."""
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float
    metadata: Dict


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    async def upsert(self, chunk_id: str, vector: List[float], payload: dict) -> None:
        """Insert or update a vector with its payload."""
        pass

    @abstractmethod
    async def delete_by_file(self, file_path: str) -> int:
        """Delete all vectors for a file, return count deleted."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        file_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            file_filter: Optional glob pattern for file paths
            
        Returns:
            List of SearchResult sorted by score descending
        """
        pass

    @abstractmethod
    async def get_stats(self) -> dict:
        """Get storage statistics."""
        pass

    @abstractmethod
    async def get_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """Get a specific chunk by ID."""
        pass


class QdrantVectorStore(VectorStoreInterface):
    """
    Qdrant-based vector store implementation.
    
    Stores code chunk embeddings with metadata for semantic search.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "aci_codebase",
        vector_size: int = 1536,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection
            vector_size: Dimension of embedding vectors
            api_key: Optional API key for authentication
        """
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._api_key = api_key
        self._client: Optional[AsyncQdrantClient] = None
        self._initialized = False

    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create the Qdrant client."""
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=self._host,
                port=self._port,
                api_key=self._api_key,
            )
        return self._client

    async def initialize(self) -> None:
        """Initialize the collection if it doesn't exist."""
        if self._initialized:
            return

        client = await self._get_client()

        try:
            # Check if collection exists
            collections = await client.get_collections()
            exists = any(
                c.name == self._collection_name for c in collections.collections
            )

            if not exists:
                await client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self._vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {self._collection_name}")

                # Create payload index for file_path filtering
                await client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name="file_path",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index for file_path")

            self._initialized = True

        except Exception as e:
            raise VectorStoreError(f"Failed to initialize collection: {e}") from e

    async def upsert(self, chunk_id: str, vector: List[float], payload: dict) -> None:
        """Insert or update a vector with its payload."""
        await self.initialize()
        client = await self._get_client()

        try:
            await client.upsert(
                collection_name=self._collection_name,
                points=[
                    models.PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert vector: {e}") from e

    async def upsert_batch(
        self, points: List[tuple[str, List[float], dict]]
    ) -> None:
        """
        Batch insert or update vectors.
        
        Args:
            points: List of (chunk_id, vector, payload) tuples
        """
        await self.initialize()
        client = await self._get_client()

        try:
            qdrant_points = [
                models.PointStruct(id=chunk_id, vector=vector, payload=payload)
                for chunk_id, vector, payload in points
            ]
            await client.upsert(
                collection_name=self._collection_name,
                points=qdrant_points,
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to batch upsert vectors: {e}") from e

    async def delete_by_file(self, file_path: str) -> int:
        """Delete all vectors for a file, return count deleted."""
        await self.initialize()
        client = await self._get_client()

        try:
            # First count how many will be deleted
            count_result = await client.count(
                collection_name=self._collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path),
                        )
                    ]
                ),
            )
            count = count_result.count

            if count > 0:
                # Delete by filter
                await client.delete(
                    collection_name=self._collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="file_path",
                                    match=models.MatchValue(value=file_path),
                                )
                            ]
                        )
                    ),
                )

            return count

        except Exception as e:
            raise VectorStoreError(f"Failed to delete vectors: {e}") from e

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        file_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            file_filter: Optional glob pattern for file paths
            
        Returns:
            List of SearchResult sorted by score descending
        """
        await self.initialize()
        client = await self._get_client()

        try:
            # Build filter if file_filter is provided
            search_filter = None
            if file_filter:
                # For glob patterns, we need to fetch more and filter client-side
                # Qdrant doesn't support glob matching natively
                pass

            results = await client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=limit * 5 if file_filter else limit,  # Fetch more if filtering
                query_filter=search_filter,
                with_payload=True,
            )

            search_results = []
            for point in results:
                payload = point.payload or {}
                file_path = payload.get("file_path", "")

                # Apply glob filter if specified
                if file_filter and not fnmatch.fnmatch(file_path, file_filter):
                    continue

                search_results.append(
                    SearchResult(
                        chunk_id=str(point.id),
                        file_path=file_path,
                        start_line=payload.get("start_line", 0),
                        end_line=payload.get("end_line", 0),
                        content=payload.get("content", ""),
                        score=point.score,
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k not in ("file_path", "start_line", "end_line", "content")
                        },
                    )
                )

                if len(search_results) >= limit:
                    break

            return search_results

        except Exception as e:
            raise VectorStoreError(f"Failed to search vectors: {e}") from e

    async def get_stats(self) -> dict:
        """Get storage statistics."""
        await self.initialize()
        client = await self._get_client()

        try:
            collection_info = await client.get_collection(self._collection_name)

            # Get unique file count by scrolling through points
            # This is expensive for large collections, consider caching
            unique_files = set()
            offset = None
            while True:
                records, offset = await client.scroll(
                    collection_name=self._collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["file_path"],
                )
                for record in records:
                    if record.payload:
                        unique_files.add(record.payload.get("file_path", ""))
                if offset is None:
                    break

            return {
                "total_vectors": collection_info.points_count,
                "total_files": len(unique_files),
                "collection_name": self._collection_name,
                "vector_size": self._vector_size,
            }

        except Exception as e:
            raise VectorStoreError(f"Failed to get stats: {e}") from e

    async def get_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """Get a specific chunk by ID."""
        await self.initialize()
        client = await self._get_client()

        try:
            results = await client.retrieve(
                collection_name=self._collection_name,
                ids=[chunk_id],
                with_payload=True,
            )

            if not results:
                return None

            point = results[0]
            payload = point.payload or {}

            return SearchResult(
                chunk_id=str(point.id),
                file_path=payload.get("file_path", ""),
                start_line=payload.get("start_line", 0),
                end_line=payload.get("end_line", 0),
                content=payload.get("content", ""),
                score=1.0,  # No score for direct retrieval
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in ("file_path", "start_line", "end_line", "content")
                },
            )

        except UnexpectedResponse:
            return None
        except Exception as e:
            raise VectorStoreError(f"Failed to get by ID: {e}") from e

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False


def create_vector_store(
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "aci_codebase",
    vector_size: int = 1536,
    api_key: Optional[str] = None,
) -> VectorStoreInterface:
    """
    Factory function to create a vector store.
    
    Args:
        host: Qdrant server host
        port: Qdrant server port
        collection_name: Name of the collection
        vector_size: Dimension of embedding vectors
        api_key: Optional API key for authentication
        
    Returns:
        Configured VectorStoreInterface instance
    """
    return QdrantVectorStore(
        host=host,
        port=port,
        collection_name=collection_name,
        vector_size=vector_size,
        api_key=api_key,
    )

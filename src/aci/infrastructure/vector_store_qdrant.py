"""
Qdrant-based vector store implementation.

Stores code chunk embeddings with metadata for semantic search.
"""

import asyncio
import fnmatch
import logging
from typing import List, Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from aci.infrastructure.vector_store_base import (
    SearchResult,
    VectorStoreError,
    VectorStoreInterface,
    is_glob_pattern,
)

logger = logging.getLogger(__name__)


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

    def set_collection(self, collection_name: str) -> None:
        """
        Switch to a different collection.
        
        This allows using the same vector store instance for different
        repositories, each with their own isolated collection.
        
        Args:
            collection_name: Name of the collection to use.
        """
        if collection_name != self._collection_name:
            self._collection_name = collection_name
            self._initialized = False  # Force re-initialization for new collection

    def get_collection_name(self) -> str:
        """Get the current collection name."""
        return self._collection_name

    async def initialize(self) -> None:
        """Initialize the collection if it doesn't exist."""
        if self._initialized:
            return

        client = await self._get_client()

        try:
            # Check if collection exists
            collections = await client.get_collections()
            exists = any(c.name == self._collection_name for c in collections.collections)

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
            else:
                # Validate existing collection vector size matches config
                collection_info = await client.get_collection(self._collection_name)
                existing_params = collection_info.config.params if collection_info.config else None
                existing_param_size = (
                    existing_params.vectors.size
                    if existing_params and existing_params.vectors
                    else None
                )
                if existing_param_size and existing_param_size != self._vector_size:
                    raise VectorStoreError(
                        f"Existing collection '{self._collection_name}' has vector size {existing_param_size}, "
                        f"but config requested {self._vector_size}. "
                        f"Either delete/recreate the collection or set ACI_VECTOR_STORE_VECTOR_SIZE="
                        f"{existing_param_size} (and match your embedding dimension)."
                    )

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
            raise VectorStoreError(
                f"Failed to upsert vector (host={self._host}, port={self._port}, "
                f"collection={self._collection_name}, vector_size={self._vector_size}): {e}"
            ) from e

    async def upsert_batch(self, points: List[tuple[str, List[float], dict]]) -> None:
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
            raise VectorStoreError(
                f"Failed to batch upsert vectors (host={self._host}, port={self._port}, "
                f"collection={self._collection_name}, vector_size={self._vector_size}, "
                f"count={len(points)}): {e}"
            ) from e

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
            raise VectorStoreError(
                f"Failed to delete vectors for file '{file_path}' "
                f"(host={self._host}, port={self._port}, collection={self._collection_name}): {e}"
            ) from e


    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        file_filter: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            file_filter: Optional glob pattern or exact file path
            collection_name: Optional collection to search. If provided, searches
                that collection without modifying instance state. If None, uses
                the instance's default collection.

        Returns:
            List of SearchResult sorted by score descending
        """
        await self.initialize()
        client = await self._get_client()
        
        # Use provided collection or fall back to instance default
        target_collection = collection_name or self._collection_name

        try:
            # Determine if file_filter is an exact path or glob pattern
            # Exact paths have no glob wildcards: *, ?, [
            is_exact_path = file_filter and not is_glob_pattern(file_filter)

            # Build filter and determine fetch limit
            search_filter = None
            if is_exact_path:
                # Server-side exact match filter - more efficient and complete
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_filter),
                        )
                    ]
                )
                fetch_limit = limit  # Exact match, no need to over-fetch
            elif file_filter:
                # Glob pattern - need client-side filtering with larger fetch
                fetch_limit = limit * 10  # Increased from 5x to 10x for better recall
            else:
                fetch_limit = limit

            # Prefer query_points (available in modern clients)
            try:
                results = await client.query_points(
                    collection_name=target_collection,
                    query=query_vector,
                    limit=fetch_limit,
                    query_filter=search_filter,
                    with_payload=True,
                )
            except AttributeError:
                logger.warning(
                    "AsyncQdrantClient.query_points unavailable, falling back to sync client"
                )
                results = await self._query_with_sync_client(
                    query_vector=query_vector,
                    limit=fetch_limit,
                    search_filter=search_filter,
                    target_collection=target_collection,
                )

            search_results = []
            # query_points returns a models.QueryResponse; for sync fallback, we will return list of ScoredPoint
            points = results if isinstance(results, list) else getattr(results, "points", results)

            # Determine if client-side filtering is needed (glob patterns only)
            needs_client_filter = file_filter and is_glob_pattern(file_filter)

            for point in points:
                payload = point.payload or {}
                file_path = payload.get("file_path", "")

                # Apply glob filter client-side if needed
                # Exact path matches are handled server-side
                if needs_client_filter and not fnmatch.fnmatch(file_path, file_filter):
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
            raise VectorStoreError(
                f"Failed to search vectors (host={self._host}, port={self._port}, "
                f"collection={target_collection}, limit={limit}, filter={file_filter}): {e}"
            ) from e

    async def _search_with_sync_client(
        self,
        query_vector: List[float],
        limit: int,
        search_filter: Optional[models.Filter],
        target_collection: Optional[str] = None,
    ):
        """Fallback search using sync QdrantClient executed in a thread."""
        from qdrant_client import QdrantClient

        collection_to_use = target_collection or self._collection_name

        def _do_query():
            client = QdrantClient(
                host=self._host,
                port=self._port,
                api_key=self._api_key,
            )
            return client.query_points(
                collection_name=collection_to_use,
                query=query_vector,
                limit=limit,
                query_filter=search_filter,
                with_payload=True,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_query)

    async def _query_with_sync_client(
        self,
        query_vector: List[float],
        limit: int,
        search_filter: Optional[models.Filter],
        target_collection: Optional[str] = None,
    ):
        """Backward compatible alias; kept for clarity."""
        from qdrant_client import QdrantClient

        collection_to_use = target_collection or self._collection_name

        def _do_search():
            client = QdrantClient(
                host=self._host,
                port=self._port,
                api_key=self._api_key,
            )
            return client.query_points(
                collection_name=collection_to_use,
                query=query_vector,
                limit=limit,
                query_filter=search_filter,
                with_payload=True,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_search)

    async def get_stats(self, collection_name: Optional[str] = None) -> dict:
        """
        Get storage statistics.

        Args:
            collection_name: Optional collection to get stats for. If provided,
                returns stats for that collection without modifying instance state.
                If None, uses the instance's default collection.

        Returns:
            Dictionary with storage statistics
        """
        await self.initialize()
        client = await self._get_client()
        
        # Use provided collection or fall back to instance default
        target_collection = collection_name or self._collection_name

        try:
            collection_info = await client.get_collection(target_collection)

            # Get unique file count by scrolling through points
            # This is expensive for large collections, consider caching
            unique_files = set()
            offset = None
            while True:
                records, offset = await client.scroll(
                    collection_name=target_collection,
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
                "collection_name": target_collection,
                "vector_size": self._vector_size,
            }

        except Exception as e:
            raise VectorStoreError(
                f"Failed to get stats (host={self._host}, port={self._port}, "
                f"collection={target_collection}): {e}"
            ) from e

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
            raise VectorStoreError(
                f"Failed to get by ID (host={self._host}, port={self._port}, "
                f"collection={self._collection_name}, id={chunk_id}): {e}"
            ) from e

    async def get_all_file_paths(self, collection_name: Optional[str] = None) -> List[str]:
        """
        Get all unique file paths in the store.

        Args:
            collection_name: Optional collection to query. If provided, returns
                file paths from that collection without modifying instance state.
                If None, uses the instance's default collection.

        Returns:
            List of unique file paths
        """
        await self.initialize()
        client = await self._get_client()
        
        # Use provided collection or fall back to instance default
        target_collection = collection_name or self._collection_name

        try:
            unique_files = set()
            offset = None
            while True:
                records, offset = await client.scroll(
                    collection_name=target_collection,
                    limit=1000,
                    offset=offset,
                    with_payload=["file_path"],
                )
                for record in records:
                    if record.payload:
                        file_path = record.payload.get("file_path", "")
                        if file_path:
                            unique_files.add(file_path)
                if offset is None:
                    break

            return list(unique_files)

        except Exception as e:
            raise VectorStoreError(
                f"Failed to get file paths (host={self._host}, port={self._port}, "
                f"collection={target_collection}): {e}"
            ) from e

    async def reset(self) -> None:
        """Drop and recreate the collection."""
        client = await self._get_client()
        try:
            await client.delete_collection(self._collection_name)
        except Exception:
            # Ignore if doesn't exist
            pass
        self._initialized = False
        await self.initialize()

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection entirely.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if the collection was deleted, False if it did not exist
        """
        client = await self._get_client()
        try:
            # Check if collection exists first
            collections = await client.get_collections()
            exists = any(c.name == collection_name for c in collections.collections)
            
            if not exists:
                return False
            
            await client.delete_collection(collection_name)
            
            # If we deleted the current collection, reset initialized flag
            if collection_name == self._collection_name:
                self._initialized = False
            
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection '{collection_name}': {e}")
            return False

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False

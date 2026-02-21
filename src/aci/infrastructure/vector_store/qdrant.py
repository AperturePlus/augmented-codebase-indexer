"""
Qdrant-based vector store implementation.

Stores code chunk embeddings with metadata for semantic search.
"""

import asyncio
import fnmatch
import logging

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from .base import SearchResult, VectorStoreError, VectorStoreInterface, is_glob_pattern

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
        api_key: str | None = None,
        url: str | None = None,
        timeout_seconds: float = 60.0,
        write_retry_attempts: int = 3,
        write_retry_backoff_seconds: float = 0.5,
    ):
        url = (url or "").strip()
        if not url and host.startswith(("http://", "https://")):
            url = host.strip()
        self._host = host
        self._port = port
        self._url = url
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._api_key = api_key or None
        self._timeout_seconds = timeout_seconds
        self._write_retry_attempts = max(1, write_retry_attempts)
        self._write_retry_backoff_seconds = max(0.0, write_retry_backoff_seconds)
        self._client: AsyncQdrantClient | None = None
        self._initialized_collections: set[str] = set()
        self._init_locks: dict[str, asyncio.Lock] = {}

    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create the Qdrant client."""
        if self._client is None:
            client_kwargs: dict = {
                "api_key": self._api_key,
                "timeout": self._timeout_seconds,
            }
            if self._url:
                client_kwargs["url"] = self._url
            else:
                client_kwargs["host"] = self._host
                client_kwargs["port"] = self._port
            self._client = AsyncQdrantClient(**client_kwargs)
        return self._client

    def _format_exception_chain(self, exc: Exception) -> str:
        """Return a readable error string including nested causes."""
        parts: list[str] = []
        current: Exception | None = exc
        depth = 0
        while current is not None and depth < 5:
            message = str(current).strip()
            if not message:
                message = repr(current)
            parts.append(f"{type(current).__name__}: {message}")
            current = current.__cause__
            depth += 1
        return " <- ".join(parts)

    def _is_retryable_exception(self, exc: Exception) -> bool:
        """Return True for transient timeout-style failures."""
        current: Exception | None = exc
        depth = 0
        while current is not None and depth < 5:
            name = type(current).__name__.lower()
            message = str(current).lower()
            if "timeout" in name or "timed out" in message or "timeout" in message:
                return True
            current = current.__cause__
            depth += 1
        return False

    def set_collection(self, collection_name: str) -> None:
        """Switch to a different collection."""
        self._collection_name = collection_name

    def get_collection_name(self) -> str:
        """Get the current collection name."""
        return self._collection_name

    async def initialize(self, collection_name: str | None = None) -> None:
        """Initialize the target collection if it doesn't exist."""
        target_collection = collection_name or self._collection_name
        if target_collection in self._initialized_collections:
            return

        init_lock = self._init_locks.get(target_collection)
        if init_lock is None:
            init_lock = asyncio.Lock()
            self._init_locks[target_collection] = init_lock

        async with init_lock:
            if target_collection in self._initialized_collections:
                return

            client = await self._get_client()

            try:
                collections = await client.get_collections()
                exists = any(c.name == target_collection for c in collections.collections)

                if not exists:
                    await client.create_collection(
                        collection_name=target_collection,
                        vectors_config=models.VectorParams(
                            size=self._vector_size,
                            distance=models.Distance.COSINE,
                        ),
                    )
                    logger.info("Created collection: %s", target_collection)

                    await client.create_payload_index(
                        collection_name=target_collection,
                        field_name="file_path",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    await client.create_payload_index(
                        collection_name=target_collection,
                        field_name="artifact_type",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    logger.info("Created payload indexes for collection: %s", target_collection)
                else:
                    collection_info = await client.get_collection(target_collection)
                    existing_params = collection_info.config.params if collection_info.config else None
                    existing_param_size = (
                        existing_params.vectors.size
                        if existing_params and existing_params.vectors
                        else None
                    )
                    if existing_param_size and existing_param_size != self._vector_size:
                        raise VectorStoreError(
                            f"Existing collection '{target_collection}' has vector size {existing_param_size}, "
                            f"but config requested {self._vector_size}."
                        )

                self._initialized_collections.add(target_collection)

            except Exception as e:
                raise VectorStoreError(
                    f"Failed to initialize collection '{target_collection}': {e}"
                ) from e

    async def upsert(
        self,
        chunk_id: str,
        vector: list[float],
        payload: dict,
        collection_name: str | None = None,
    ) -> None:
        """Insert or update a vector with its payload."""
        target_collection = collection_name or self._collection_name
        await self.initialize(target_collection)
        client = await self._get_client()

        if "artifact_type" not in payload:
            payload = {**payload, "artifact_type": "chunk"}

        last_error: Exception | None = None
        for attempt in range(1, self._write_retry_attempts + 1):
            try:
                await client.upsert(
                    collection_name=target_collection,
                    points=[
                        models.PointStruct(id=chunk_id, vector=vector, payload=payload)
                    ],
                )
                return
            except Exception as e:
                last_error = e
                if attempt < self._write_retry_attempts and self._is_retryable_exception(e):
                    delay = self._write_retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Qdrant upsert timeout (attempt %s/%s) for chunk '%s' in collection '%s'; retrying in %.2fs",
                        attempt,
                        self._write_retry_attempts,
                        chunk_id,
                        target_collection,
                        delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                break

        details = self._format_exception_chain(last_error or Exception("unknown error"))
        raise VectorStoreError(
            f"Failed to upsert vector in collection '{target_collection}' after "
            f"{self._write_retry_attempts} attempt(s): {details}"
        ) from last_error

    async def upsert_batch(
        self,
        points: list[tuple[str, list[float], dict]],
        collection_name: str | None = None,
    ) -> None:
        """Batch insert or update vectors."""
        target_collection = collection_name or self._collection_name
        await self.initialize(target_collection)
        client = await self._get_client()

        qdrant_points = [
            models.PointStruct(
                id=chunk_id,
                vector=vector,
                payload=(
                    payload if "artifact_type" in payload
                    else {**payload, "artifact_type": "chunk"}
                ),
            )
            for chunk_id, vector, payload in points
        ]

        last_error: Exception | None = None
        for attempt in range(1, self._write_retry_attempts + 1):
            try:
                await client.upsert(
                    collection_name=target_collection,
                    points=qdrant_points,
                )
                return
            except Exception as e:
                last_error = e
                if attempt < self._write_retry_attempts and self._is_retryable_exception(e):
                    delay = self._write_retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Qdrant batch upsert timeout (attempt %s/%s) for %s point(s) in collection '%s'; retrying in %.2fs",
                        attempt,
                        self._write_retry_attempts,
                        len(points),
                        target_collection,
                        delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                break

        details = self._format_exception_chain(last_error or Exception("unknown error"))
        raise VectorStoreError(
            f"Failed to batch upsert {len(points)} vector(s) in collection "
            f"'{target_collection}' after {self._write_retry_attempts} attempt(s): {details}"
        ) from last_error

    async def delete_by_file(self, file_path: str, collection_name: str | None = None) -> int:
        """Delete all vectors for a file, return count deleted."""
        target_collection = collection_name or self._collection_name
        await self.initialize(target_collection)
        client = await self._get_client()

        try:
            count_result = await client.count(
                collection_name=target_collection,
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
                await client.delete(
                    collection_name=target_collection,
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
            raise VectorStoreError(f"Failed to delete vectors for file '{file_path}': {e}") from e

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        file_filter: str | None = None,
        collection_name: str | None = None,
        artifact_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        target_collection = collection_name or self._collection_name
        await self.initialize(target_collection)
        client = await self._get_client()

        try:
            is_exact_path = file_filter and not is_glob_pattern(file_filter)

            filter_conditions = []

            if is_exact_path:
                filter_conditions.append(
                    models.FieldCondition(
                        key="file_path",
                        match=models.MatchValue(value=file_filter),
                    )
                )

            if artifact_types:
                filter_conditions.append(
                    models.FieldCondition(
                        key="artifact_type",
                        match=models.MatchAny(any=artifact_types),
                    )
                )

            search_filter = models.Filter(must=filter_conditions) if filter_conditions else None

            if is_exact_path:
                fetch_limit = limit
            elif file_filter:
                fetch_limit = limit * 10
            else:
                fetch_limit = limit

            try:
                results = await client.query_points(
                    collection_name=target_collection,
                    query=query_vector,
                    limit=fetch_limit,
                    query_filter=search_filter,
                    with_payload=True,
                )
            except AttributeError:
                logger.warning("AsyncQdrantClient.query_points unavailable, falling back to sync client")
                results = await self._query_with_sync_client(
                    query_vector=query_vector,
                    limit=fetch_limit,
                    search_filter=search_filter,
                    target_collection=target_collection,
                )

            search_results = []
            points = results if isinstance(results, list) else getattr(results, "points", results)
            needs_client_filter = file_filter and is_glob_pattern(file_filter)

            for point in points:
                payload = point.payload or {}
                file_path = payload.get("file_path", "")

                if needs_client_filter and not fnmatch.fnmatch(file_path, file_filter):
                    continue

                artifact_type = payload.get("artifact_type", "chunk")
                search_results.append(
                    SearchResult(
                        chunk_id=str(point.id),
                        file_path=file_path,
                        start_line=payload.get("start_line", 0),
                        end_line=payload.get("end_line", 0),
                        content=payload.get("content", ""),
                        score=point.score,
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

                if len(search_results) >= limit:
                    break

            return search_results

        except Exception as e:
            raise VectorStoreError(f"Failed to search vectors: {e}") from e

    async def _query_with_sync_client(
        self,
        query_vector: list[float],
        limit: int,
        search_filter: models.Filter | None,
        target_collection: str | None = None,
    ):
        """Fallback search using sync QdrantClient executed in a thread."""
        from qdrant_client import QdrantClient

        collection_to_use = target_collection or self._collection_name

        def _do_search():
            client_kwargs: dict = {"api_key": self._api_key}
            if self._url:
                client_kwargs["url"] = self._url
            else:
                client_kwargs["host"] = self._host
                client_kwargs["port"] = self._port
            client = QdrantClient(**client_kwargs)
            return client.query_points(
                collection_name=collection_to_use,
                query=query_vector,
                limit=limit,
                query_filter=search_filter,
                with_payload=True,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_search)

    async def get_stats(self, collection_name: str | None = None) -> dict:
        """Get storage statistics."""
        target_collection = collection_name or self._collection_name
        await self.initialize(target_collection)
        client = await self._get_client()

        try:
            collection_info = await client.get_collection(target_collection)

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
            raise VectorStoreError(f"Failed to get stats: {e}") from e

    async def get_by_id(self, chunk_id: str) -> SearchResult | None:
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

            artifact_type = payload.get("artifact_type", "chunk")
            return SearchResult(
                chunk_id=str(point.id),
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

        except UnexpectedResponse:
            return None
        except Exception as e:
            raise VectorStoreError(f"Failed to get by ID: {e}") from e

    async def get_all_file_paths(self, collection_name: str | None = None) -> list[str]:
        """Get all unique file paths in the store."""
        target_collection = collection_name or self._collection_name
        await self.initialize(target_collection)
        client = await self._get_client()

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
            raise VectorStoreError(f"Failed to get file paths: {e}") from e

    async def reset(self) -> None:
        """Drop and recreate the collection."""
        client = await self._get_client()
        try:
            await client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._initialized_collections.discard(self._collection_name)
        await self.initialize(self._collection_name)

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection entirely."""
        client = await self._get_client()
        try:
            collections = await client.get_collections()
            exists = any(c.name == collection_name for c in collections.collections)

            if not exists:
                return False

            await client.delete_collection(collection_name)
            self._initialized_collections.discard(collection_name)

            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection '{collection_name}': {e}")
            return False

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized_collections.clear()
            self._init_locks.clear()

"""
Vector Store module for Project ACI.

Provides Qdrant-based vector storage and retrieval for code chunks.
"""

from typing import Optional

from .base import SearchResult, VectorStoreError, VectorStoreInterface, is_glob_pattern
from .qdrant import QdrantVectorStore

__all__ = [
    "SearchResult",
    "VectorStoreError",
    "VectorStoreInterface",
    "QdrantVectorStore",
    "create_vector_store",
    "is_glob_pattern",
]


def create_vector_store(
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "aci_codebase",
    vector_size: int = 1536,
    api_key: str | None = None,
    url: str | None = None,
) -> VectorStoreInterface:
    """
    Factory function to create a vector store.

    Args:
        url: Optional Qdrant URL (takes precedence over host/port)
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
        url=url,
    )

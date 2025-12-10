"""
Vector Store for Project ACI.

Provides Qdrant-based vector storage and retrieval for code chunks.

This module re-exports all public types from the vector store submodules
for backward compatibility.
"""

from typing import Optional

from aci.infrastructure.vector_store_base import (
    SearchResult,
    VectorStoreError,
    VectorStoreInterface,
    is_glob_pattern,
)
from aci.infrastructure.vector_store_qdrant import QdrantVectorStore

# Re-export for backward compatibility
__all__ = [
    "SearchResult",
    "VectorStoreError",
    "VectorStoreInterface",
    "QdrantVectorStore",
    "create_vector_store",
    "is_glob_pattern",
]

# Keep the old private name for any internal references
_is_glob_pattern = is_glob_pattern


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

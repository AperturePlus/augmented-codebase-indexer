"""
Infrastructure Layer - Embedding client and vector store implementations.
"""

from aci.infrastructure.embedding_client import (
    EmbeddingClientError,
    EmbeddingClientInterface,
    NonRetryableError,
    OpenAIEmbeddingClient,
    RetryableError,
    RetryConfig,
    create_embedding_client,
)
from aci.infrastructure.vector_store import (
    QdrantVectorStore,
    SearchResult,
    VectorStoreError,
    VectorStoreInterface,
    create_vector_store,
)
from aci.infrastructure.metadata_store import (
    IndexedFileInfo,
    IndexMetadataStore,
    MetadataStoreError,
    create_metadata_store,
)
from aci.infrastructure.fakes import (
    InMemoryVectorStore,
    LocalEmbeddingClient,
)

__all__ = [
    # Embedding client
    "EmbeddingClientInterface",
    "OpenAIEmbeddingClient",
    "EmbeddingClientError",
    "RetryableError",
    "NonRetryableError",
    "RetryConfig",
    "create_embedding_client",
    # Vector store
    "VectorStoreInterface",
    "QdrantVectorStore",
    "SearchResult",
    "VectorStoreError",
    "create_vector_store",
    # Metadata store
    "IndexMetadataStore",
    "IndexedFileInfo",
    "MetadataStoreError",
    "create_metadata_store",
    # Fakes for testing
    "InMemoryVectorStore",
    "LocalEmbeddingClient",
]

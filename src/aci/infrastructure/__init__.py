"""
Infrastructure Layer - Embedding client and vector store implementations.
"""

from aci.infrastructure.embedding import (
    BatchSizeError,
    EmbeddingClientError,
    EmbeddingClientInterface,
    NonRetryableError,
    OpenAIEmbeddingClient,
    RetryableError,
    RetryConfig,
    create_embedding_client,
)
from aci.infrastructure.fakes import (
    FakeFileWatcher,
    InMemoryVectorStore,
    LocalEmbeddingClient,
)
from aci.infrastructure.file_watcher import (
    FileWatcher,
    FileWatcherInterface,
)
from aci.infrastructure.grep_searcher import (
    GrepSearcher,
    GrepSearcherError,
    GrepSearcherInterface,
)
from aci.infrastructure.metadata_store import (
    IndexedFileInfo,
    IndexMetadataStore,
    MetadataStoreError,
    PendingBatch,
    create_metadata_store,
)
from aci.infrastructure.vector_store import (
    QdrantVectorStore,
    SearchResult,
    VectorStoreError,
    VectorStoreInterface,
    create_vector_store,
    is_glob_pattern,
)

__all__ = [
    # Embedding client
    "EmbeddingClientInterface",
    "OpenAIEmbeddingClient",
    "EmbeddingClientError",
    "RetryableError",
    "NonRetryableError",
    "BatchSizeError",
    "RetryConfig",
    "create_embedding_client",
    # Vector store
    "VectorStoreInterface",
    "QdrantVectorStore",
    "SearchResult",
    "VectorStoreError",
    "create_vector_store",
    "is_glob_pattern",
    # Grep searcher
    "GrepSearcherInterface",
    "GrepSearcher",
    "GrepSearcherError",
    # Metadata store
    "IndexMetadataStore",
    "IndexedFileInfo",
    "PendingBatch",
    "MetadataStoreError",
    "create_metadata_store",
    # File watcher
    "FileWatcherInterface",
    "FileWatcher",
    # Fakes for testing
    "InMemoryVectorStore",
    "LocalEmbeddingClient",
    "FakeFileWatcher",
]

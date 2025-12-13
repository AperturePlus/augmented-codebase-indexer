"""
Centralized services container module for ACI.

Provides a shared container for all services used across CLI, HTTP, and MCP
entry points. This module eliminates the dependency on CLI code from non-CLI
components.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aci.core.chunker import Chunker, create_chunker
from aci.core.config import ACIConfig, load_config
from aci.core.file_scanner import FileScanner
from aci.core.qdrant_launcher import ensure_qdrant_running
from aci.core.summary_generator import SummaryGenerator
from aci.infrastructure import (
    EmbeddingClientInterface,
    IndexMetadataStore,
    VectorStoreInterface,
    create_embedding_client,
    create_metadata_store,
    create_vector_store,
)
from aci.services.reranker import (
    OpenAICompatibleReranker,
    SimpleReranker,
)
from aci.services.search_types import RerankerInterface


@dataclass
class ServicesContainer:
    """
    Container holding all shared service instances.

    This container is initialized once and reused across commands
    within an interactive session to avoid repeated initialization overhead.

    Attributes:
        config: Application configuration
        embedding_client: Client for generating embeddings
        vector_store: Vector database for similarity search
        metadata_store: SQLite store for file metadata
        file_scanner: Scanner for discovering code files
        chunker: Code chunker for splitting files
        reranker: Optional reranker for improving search results
    """

    config: ACIConfig
    embedding_client: EmbeddingClientInterface
    vector_store: VectorStoreInterface
    metadata_store: IndexMetadataStore
    file_scanner: FileScanner
    chunker: Chunker
    reranker: Optional[RerankerInterface] = None


def create_services(
    config_path: Optional[Path] = None,
    metadata_db_path: Optional[Path] = None,
) -> ServicesContainer:
    """
    Create and initialize all services.

    This factory function initializes all required services from configuration,
    ensuring Qdrant is running and all components are properly configured.

    Args:
        config_path: Optional path to configuration file. If None, uses
                    environment variables and defaults.
        metadata_db_path: Optional path for metadata database. If None,
                         defaults to '.aci/index.db'.

    Returns:
        ServicesContainer with all initialized services.

    Raises:
        ValueError: If required configuration is missing (e.g., API key).
        ConnectionError: If unable to connect to required services.
    """
    # Load configuration
    config = load_config(config_path)

    # Ensure Qdrant is running
    ensure_qdrant_running(port=config.vector_store.port)

    # Create embedding client
    embedding_client = create_embedding_client(
        api_url=config.embedding.api_url,
        api_key=config.embedding.api_key,
        model=config.embedding.model,
        batch_size=config.embedding.batch_size,
        max_retries=config.embedding.max_retries,
        timeout=config.embedding.timeout,
        dimension=config.embedding.dimension,
    )

    # Create vector store
    vector_store = create_vector_store(
        host=config.vector_store.host,
        port=config.vector_store.port,
        collection_name=config.vector_store.collection_name,
        vector_size=config.vector_store.vector_size,
    )

    # Create metadata store
    db_path = metadata_db_path or Path(".aci/index.db")
    metadata_store = create_metadata_store(db_path)

    # Create file scanner with config-driven settings
    file_scanner = FileScanner(
        extensions=set(config.indexing.file_extensions),
        ignore_patterns=config.indexing.ignore_patterns,
    )

    # Create summary generator for multi-granularity indexing
    summary_generator = SummaryGenerator()

    # Create chunker with config-driven settings
    chunker = create_chunker(
        max_tokens=config.indexing.max_chunk_tokens,
        overlap_lines=config.indexing.chunk_overlap_lines,
        summary_generator=summary_generator,
    )

    # Create reranker if enabled
    reranker: Optional[RerankerInterface] = None
    if config.search.use_rerank:
        if config.search.rerank_api_url:
            reranker = OpenAICompatibleReranker(
                api_url=config.search.rerank_api_url,
                api_key=config.search.rerank_api_key,
                model=config.search.rerank_model,
                timeout=config.search.rerank_timeout,
                endpoint=config.search.rerank_endpoint,
            )
        else:
            reranker = SimpleReranker()

    return ServicesContainer(
        config=config,
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        chunker=chunker,
        reranker=reranker,
    )

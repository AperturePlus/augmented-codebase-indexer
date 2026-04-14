"""
Centralized services container module for ACI.

Provides a shared container for all services used across CLI, HTTP, and MCP
entry points. This module eliminates the dependency on CLI code from non-CLI
components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aci.core.chunker import Chunker, create_chunker
from aci.core.config import ACIConfig, load_config
from aci.core.file_scanner import FileScanner
from aci.core.qdrant_launcher import ensure_qdrant_running
from aci.core.summary_generator import SummaryGenerator
from aci.core.tokenizer import get_default_tokenizer
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

if TYPE_CHECKING:
    from aci.core.graph_store import GraphStoreInterface
    from aci.core.parsers.reference_extractor import ReferenceExtractorInterface
    from aci.services.context_assembler import ContextAssembler
    from aci.services.graph_builder import GraphBuilder
    from aci.services.llm_enricher import LLMEnricher
    from aci.services.pagerank_scorer import PageRankScorer
    from aci.services.query_router import QueryRouter
    from aci.services.rrf_fuser import RRFFuser
    from aci.services.topology_analyzer import TopologyAnalyzer

logger = logging.getLogger(__name__)


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
        graph_store: Optional graph store for code-relationship graphs
        graph_builder: Optional graph builder for constructing graphs
        topology_analyzer: Optional topology analyzer for graph traversals
        pagerank_scorer: Optional PageRank scorer for graph centrality
        context_assembler: Optional context assembler for structured context
        query_router: Optional unified query router
        llm_enricher: Optional LLM enricher for semantic summaries
        rrf_fuser: Optional RRF fuser for rank fusion
    """

    config: ACIConfig
    embedding_client: EmbeddingClientInterface
    vector_store: VectorStoreInterface
    metadata_store: IndexMetadataStore
    file_scanner: FileScanner
    chunker: Chunker
    reranker: RerankerInterface | None = None
    graph_store: GraphStoreInterface | None = None
    graph_builder: GraphBuilder | None = None
    topology_analyzer: TopologyAnalyzer | None = None
    pagerank_scorer: PageRankScorer | None = None
    context_assembler: ContextAssembler | None = None
    query_router: QueryRouter | None = None
    llm_enricher: LLMEnricher | None = None
    rrf_fuser: RRFFuser | None = None


def _create_reference_extractors() -> dict[str, ReferenceExtractorInterface]:
    """Create a registry of language-specific reference extractors.

    Returns:
        Mapping from language name to its :class:`ReferenceExtractorInterface`.
    """
    from aci.core.parsers.cpp_reference_extractor import CppReferenceExtractor
    from aci.core.parsers.go_reference_extractor import GoReferenceExtractor
    from aci.core.parsers.java_reference_extractor import JavaReferenceExtractor
    from aci.core.parsers.javascript_reference_extractor import JavaScriptReferenceExtractor
    from aci.core.parsers.python_reference_extractor import PythonReferenceExtractor

    extractors: dict[str, ReferenceExtractorInterface] = {
        "python": PythonReferenceExtractor(),
        "javascript": JavaScriptReferenceExtractor(),
        "typescript": JavaScriptReferenceExtractor(),
        "go": GoReferenceExtractor(),
        "java": JavaReferenceExtractor(),
        "c": CppReferenceExtractor(),
        "cpp": CppReferenceExtractor(),
    }
    return extractors


def create_services(
    config_path: Path | None = None,
    metadata_db_path: Path | None = None,
) -> ServicesContainer:
    """
    Create and initialize all services.

    This factory function initializes all required services from configuration,
    ensuring Qdrant is running and all components are properly configured.

    Conditionally creates graph components when ``config.graph.enabled`` and
    LLM enricher when ``config.llm.enabled``.

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
    ensure_qdrant_running(
        host=config.vector_store.host,
        port=config.vector_store.port,
        url=config.vector_store.url,
    )

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
        api_key=config.vector_store.api_key or None,
        url=config.vector_store.url or None,
    )

    # Create metadata store
    db_path = metadata_db_path or Path(".aci/index.db")
    metadata_store = create_metadata_store(db_path)

    # Create file scanner with config-driven settings
    file_scanner = FileScanner(
        extensions=set(config.indexing.file_extensions),
        ignore_patterns=config.indexing.ignore_patterns,
    )

    # Create tokenizer and summary generator for multi-granularity indexing
    tokenizer = get_default_tokenizer(config.indexing.tokenizer)
    summary_generator = SummaryGenerator(tokenizer=tokenizer)

    # Create chunker with config-driven settings
    chunker = create_chunker(
        tokenizer=tokenizer,
        max_tokens=config.indexing.max_chunk_tokens,
        overlap_lines=config.indexing.chunk_overlap_lines,
        summary_generator=summary_generator,
    )

    # Create reranker if enabled
    reranker: RerankerInterface | None = None
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

    # ------------------------------------------------------------------
    # Graph components (conditional on config.graph.enabled)
    # ------------------------------------------------------------------
    graph_store: GraphStoreInterface | None = None
    graph_builder: GraphBuilder | None = None
    topology_analyzer: TopologyAnalyzer | None = None
    pagerank_scorer: PageRankScorer | None = None

    if config.graph.enabled:
        from aci.core.ast_parser import TreeSitterParser
        from aci.infrastructure.graph_store import SQLiteGraphStore
        from aci.services.graph_builder import GraphBuilder as _GraphBuilder
        from aci.services.pagerank_scorer import PageRankScorer as _PageRankScorer
        from aci.services.topology_analyzer import TopologyAnalyzer as _TopologyAnalyzer

        graph_store = SQLiteGraphStore(db_path=config.graph.storage_path)
        graph_store.initialize()
        logger.info("Graph store initialized at %s", config.graph.storage_path)

        ast_parser = TreeSitterParser()
        reference_extractors = _create_reference_extractors()

        graph_builder = _GraphBuilder(
            graph_store=graph_store,
            ast_parser=ast_parser,
            reference_extractors=reference_extractors,
        )
        topology_analyzer = _TopologyAnalyzer(graph_store=graph_store)
        pagerank_scorer = _PageRankScorer(graph_store=graph_store)

    # ------------------------------------------------------------------
    # LLM enricher (conditional on config.llm.enabled)
    # ------------------------------------------------------------------
    llm_enricher: LLMEnricher | None = None

    if config.llm.enabled:
        from aci.services.llm_enricher import LLMEnricher as _LLMEnricher

        llm_enricher = _LLMEnricher(
            config=config.llm,
            summary_generator=summary_generator,
        )

    # ------------------------------------------------------------------
    # RRF fuser, context assembler, query router (always created)
    # ------------------------------------------------------------------
    from aci.services.context_assembler import ContextAssembler as _ContextAssembler
    from aci.services.rrf_fuser import RRFFuser as _RRFFuser

    rrf_fuser = _RRFFuser()

    context_assembler = _ContextAssembler(
        graph_store=graph_store,
        topology_analyzer=topology_analyzer,
        tokenizer=tokenizer,
        llm_enricher=llm_enricher,
    )

    # QueryRouter requires a SearchService, which is created by callers
    # (e.g. create_mcp_context). We store the components and let callers
    # build the router once they have a SearchService instance.
    # For now, query_router is left as None in the container.

    return ServicesContainer(
        config=config,
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        chunker=chunker,
        reranker=reranker,
        graph_store=graph_store,
        graph_builder=graph_builder,
        topology_analyzer=topology_analyzer,
        pagerank_scorer=pagerank_scorer,
        context_assembler=context_assembler,
        query_router=None,
        llm_enricher=llm_enricher,
        rrf_fuser=rrf_fuser,
    )

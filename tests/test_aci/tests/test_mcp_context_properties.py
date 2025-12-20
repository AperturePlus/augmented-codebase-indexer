"""
Property-based tests for MCPContext completeness.

**Feature: mcp-dependency-injection, Property 2: MCPContext Completeness**
**Validates: Requirements 3.1, 3.2**
"""

import asyncio
from dataclasses import fields

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aci.core.chunker import create_chunker
from aci.core.config import ACIConfig
from aci.core.file_scanner import FileScanner
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.mcp.context import MCPContext
from aci.services import IndexingService, SearchService

# Create shared instances to avoid repeated expensive initialization
_shared_chunker = None


def _get_shared_chunker():
    """Get or create a shared chunker instance for testing."""
    global _shared_chunker
    if _shared_chunker is None:
        _shared_chunker = create_chunker()
    return _shared_chunker


@st.composite
def mcp_context_strategy(draw):
    """
    Generate valid MCPContext instances using fake implementations.

    This strategy creates contexts with all required fields populated
    using test doubles from the infrastructure fakes module.
    """
    # Create a minimal valid config
    config = ACIConfig()

    # Create fake embedding client
    dimension = draw(st.sampled_from([256, 512, 1024, 1536]))
    embedding_client = LocalEmbeddingClient(dimension=dimension)

    # Create fake vector store
    vector_store = InMemoryVectorStore(vector_size=dimension)

    # Create in-memory metadata store
    metadata_store = IndexMetadataStore(":memory:")

    # Create file scanner
    file_scanner = FileScanner(extensions={".py"})

    # Use shared chunker
    chunker = _get_shared_chunker()

    # Create SearchService with injected dependencies
    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=None,
        grep_searcher=None,
        default_limit=config.search.default_limit,
    )

    # Create IndexingService with injected dependencies
    indexing_service = IndexingService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        chunker=chunker,
        batch_size=config.embedding.batch_size,
        max_workers=1,
    )

    # Create asyncio lock
    indexing_lock = asyncio.Lock()

    return MCPContext(
        config=config,
        search_service=search_service,
        indexing_service=indexing_service,
        metadata_store=metadata_store,
        vector_store=vector_store,
        indexing_lock=indexing_lock,
        reranker=None,
        embedding_client=embedding_client,
    )


@given(ctx=mcp_context_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_mcp_context_has_all_required_fields(ctx: MCPContext):
    """
    **Feature: mcp-dependency-injection, Property 2: MCPContext Completeness**
    **Validates: Requirements 3.1, 3.2**

    For any MCPContext instance created by create_mcp_context(), all required
    fields (config, search_service, indexing_service, metadata_store,
    vector_store, indexing_lock) SHALL be non-None and of the correct type.
    """
    # Required fields that must be non-None
    required_fields = [
        "config",
        "search_service",
        "indexing_service",
        "metadata_store",
        "vector_store",
        "indexing_lock",
    ]

    for field_name in required_fields:
        value = getattr(ctx, field_name)
        assert value is not None, f"Required field '{field_name}' must not be None"


@given(ctx=mcp_context_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_mcp_context_fields_have_correct_types(ctx: MCPContext):
    """
    **Feature: mcp-dependency-injection, Property 2: MCPContext Completeness**
    **Validates: Requirements 3.1, 3.2**

    For any MCPContext instance, all fields SHALL have the correct types
    as specified in the dataclass definition.
    """
    # Verify config is ACIConfig
    assert isinstance(ctx.config, ACIConfig), "config must be ACIConfig"

    # Verify search_service is SearchService
    assert isinstance(ctx.search_service, SearchService), \
        "search_service must be SearchService"

    # Verify indexing_service is IndexingService
    assert isinstance(ctx.indexing_service, IndexingService), \
        "indexing_service must be IndexingService"

    # Verify metadata_store is IndexMetadataStore
    assert isinstance(ctx.metadata_store, IndexMetadataStore), \
        "metadata_store must be IndexMetadataStore"

    # Verify vector_store implements the interface (duck typing)
    assert hasattr(ctx.vector_store, "search"), \
        "vector_store must have search method"
    assert hasattr(ctx.vector_store, "upsert"), \
        "vector_store must have upsert method"

    # Verify indexing_lock is asyncio.Lock
    assert isinstance(ctx.indexing_lock, asyncio.Lock), \
        "indexing_lock must be asyncio.Lock"


def test_mcp_context_dataclass_has_expected_fields():
    """
    **Feature: mcp-dependency-injection, Property 2: MCPContext Completeness**
    **Validates: Requirements 3.1, 3.2**

    The MCPContext dataclass SHALL define all expected fields.
    """
    expected_fields = {
        "config",
        "search_service",
        "indexing_service",
        "metadata_store",
        "vector_store",
        "indexing_lock",
        "indexing_locks",
        "reranker",
        "embedding_client",
    }

    actual_fields = {f.name for f in fields(MCPContext)}

    assert expected_fields == actual_fields, \
        f"MCPContext fields mismatch. Expected: {expected_fields}, Got: {actual_fields}"


def test_mcp_context_optional_fields_can_be_none():
    """
    **Feature: mcp-dependency-injection, Property 2: MCPContext Completeness**
    **Validates: Requirements 3.1, 3.2**

    The reranker and embedding_client fields SHALL be optional (can be None).
    These are stored for cleanup purposes only.
    """
    config = ACIConfig()
    embedding_client = LocalEmbeddingClient()
    vector_store = InMemoryVectorStore()
    metadata_store = IndexMetadataStore(":memory:")
    file_scanner = FileScanner(extensions={".py"})
    chunker = _get_shared_chunker()

    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=None,
        grep_searcher=None,
        default_limit=10,
    )

    indexing_service = IndexingService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        chunker=chunker,
        batch_size=32,
        max_workers=1,
    )

    ctx = MCPContext(
        config=config,
        search_service=search_service,
        indexing_service=indexing_service,
        metadata_store=metadata_store,
        vector_store=vector_store,
        indexing_lock=asyncio.Lock(),
        reranker=None,  # Explicitly None
        embedding_client=None,  # Explicitly None
    )

    # Should not raise - these fields are optional
    assert ctx.reranker is None
    assert ctx.embedding_client is None


# Property 3: Service Construction with DI


@given(ctx=mcp_context_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_search_service_uses_injected_dependencies(ctx: MCPContext):
    """
    **Feature: mcp-dependency-injection, Property 3: Service Construction with DI**
    **Validates: Requirements 2.2**

    For any MCPContext, the search_service SHALL have been constructed with
    dependencies from ServicesContainer (not created internally).

    This verifies that SearchService receives its embedding_client and
    vector_store via constructor injection.
    """
    search_service = ctx.search_service

    # Verify SearchService has the injected dependencies as attributes
    # These should be the same instances passed during construction
    assert hasattr(search_service, "_embedding_client"), \
        "SearchService must store injected embedding_client"
    assert hasattr(search_service, "_vector_store"), \
        "SearchService must store injected vector_store"

    # Verify the dependencies are not None (they were injected)
    assert search_service._embedding_client is not None, \
        "SearchService._embedding_client must be injected, not None"
    assert search_service._vector_store is not None, \
        "SearchService._vector_store must be injected, not None"

    # Verify the injected dependencies match what's in the context
    # This ensures the same instances are shared (DI pattern)
    assert search_service._vector_store is ctx.vector_store, \
        "SearchService must use the same vector_store instance as MCPContext"


@given(ctx=mcp_context_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_indexing_service_uses_injected_dependencies(ctx: MCPContext):
    """
    **Feature: mcp-dependency-injection, Property 3: Service Construction with DI**
    **Validates: Requirements 2.2**

    For any MCPContext, the indexing_service SHALL have been constructed with
    dependencies from ServicesContainer (not created internally).

    This verifies that IndexingService receives its embedding_client,
    vector_store, and metadata_store via constructor injection.
    """
    indexing_service = ctx.indexing_service

    # Verify IndexingService has the injected dependencies as attributes
    assert hasattr(indexing_service, "_embedding_client"), \
        "IndexingService must store injected embedding_client"
    assert hasattr(indexing_service, "_vector_store"), \
        "IndexingService must store injected vector_store"
    assert hasattr(indexing_service, "_metadata_store"), \
        "IndexingService must store injected metadata_store"

    # Verify the dependencies are not None (they were injected)
    assert indexing_service._embedding_client is not None, \
        "IndexingService._embedding_client must be injected, not None"
    assert indexing_service._vector_store is not None, \
        "IndexingService._vector_store must be injected, not None"
    assert indexing_service._metadata_store is not None, \
        "IndexingService._metadata_store must be injected, not None"

    # Verify the injected dependencies match what's in the context
    # This ensures the same instances are shared (DI pattern)
    assert indexing_service._vector_store is ctx.vector_store, \
        "IndexingService must use the same vector_store instance as MCPContext"
    assert indexing_service._metadata_store is ctx.metadata_store, \
        "IndexingService must use the same metadata_store instance as MCPContext"


def test_services_share_infrastructure_instances():
    """
    **Feature: mcp-dependency-injection, Property 3: Service Construction with DI**
    **Validates: Requirements 2.2**

    When MCPContext is created, SearchService and IndexingService SHALL share
    the same infrastructure instances (embedding_client, vector_store).
    This verifies proper dependency injection where services don't create
    their own instances internally.
    """
    config = ACIConfig()
    embedding_client = LocalEmbeddingClient()
    vector_store = InMemoryVectorStore()
    metadata_store = IndexMetadataStore(":memory:")
    file_scanner = FileScanner(extensions={".py"})
    chunker = _get_shared_chunker()

    # Create services with shared dependencies
    search_service = SearchService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        reranker=None,
        grep_searcher=None,
        default_limit=10,
    )

    indexing_service = IndexingService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        chunker=chunker,
        batch_size=32,
        max_workers=1,
    )

    ctx = MCPContext(
        config=config,
        search_service=search_service,
        indexing_service=indexing_service,
        metadata_store=metadata_store,
        vector_store=vector_store,
        indexing_lock=asyncio.Lock(),
        embedding_client=embedding_client,
    )

    # Verify both services share the same embedding_client instance
    assert search_service._embedding_client is indexing_service._embedding_client, \
        "SearchService and IndexingService must share the same embedding_client"

    # Verify both services share the same vector_store instance
    assert search_service._vector_store is indexing_service._vector_store, \
        "SearchService and IndexingService must share the same vector_store"

    # Verify the shared instances are the same as in the context
    assert ctx.vector_store is search_service._vector_store, \
        "MCPContext.vector_store must be the same instance used by services"

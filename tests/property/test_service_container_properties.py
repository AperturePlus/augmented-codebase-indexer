"""
Property-based tests for ServicesContainer completeness.

**Feature: service-initialization-refactor, Property 3: ServicesContainer Completeness**
**Validates: Requirements 1.4**
"""

from dataclasses import fields

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aci.core.chunker import Chunker, create_chunker
from aci.core.config import ACIConfig
from aci.core.file_scanner import FileScanner
from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.metadata_store import IndexMetadataStore
from aci.services.container import ServicesContainer

# Create a shared chunker instance to avoid repeated expensive initialization
_shared_chunker = None


def _get_shared_chunker() -> Chunker:
    """Get or create a shared chunker instance for testing."""
    global _shared_chunker
    if _shared_chunker is None:
        _shared_chunker = create_chunker()
    return _shared_chunker


# Strategy for generating valid file extensions
file_extension_strategy = st.sampled_from([".py", ".js", ".ts", ".go", ".java", ".c", ".cpp"])

# Strategy for generating ignore patterns
ignore_pattern_strategy = st.sampled_from(["__pycache__", "node_modules", ".git", "*.pyc"])


@st.composite
def services_container_strategy(draw):
    """
    Generate valid ServicesContainer instances using fake implementations.

    This strategy creates containers with all required fields populated
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

    # Create file scanner with random extensions
    extensions = draw(
        st.lists(file_extension_strategy, min_size=1, max_size=3, unique=True)
    )
    ignore_patterns = draw(
        st.lists(ignore_pattern_strategy, min_size=0, max_size=2, unique=True)
    )
    file_scanner = FileScanner(extensions=set(extensions), ignore_patterns=ignore_patterns)

    # Use shared chunker to avoid expensive repeated initialization
    chunker = _get_shared_chunker()

    # Optionally include reranker (None is valid)
    include_reranker = draw(st.booleans())
    reranker = None  # Reranker is optional

    return ServicesContainer(
        config=config,
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        chunker=chunker,
        reranker=reranker if include_reranker else None,
    )


@given(container=services_container_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_services_container_has_all_required_fields(container: ServicesContainer):
    """
    **Feature: service-initialization-refactor, Property 3: ServicesContainer Completeness**
    **Validates: Requirements 1.4**

    For any ServicesContainer instance, all required fields (config, embedding_client,
    vector_store, metadata_store, file_scanner, chunker) SHALL have non-None values.
    """
    # Required fields that must be non-None
    required_fields = [
        "config",
        "embedding_client",
        "vector_store",
        "metadata_store",
        "file_scanner",
        "chunker",
    ]

    for field_name in required_fields:
        value = getattr(container, field_name)
        assert value is not None, f"Required field '{field_name}' must not be None"


@given(container=services_container_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_services_container_fields_have_correct_types(container: ServicesContainer):
    """
    **Feature: service-initialization-refactor, Property 3: ServicesContainer Completeness**
    **Validates: Requirements 1.4**

    For any ServicesContainer instance, all fields SHALL have the correct types
    as specified in the dataclass definition.
    """
    # Verify config is ACIConfig
    assert isinstance(container.config, ACIConfig), "config must be ACIConfig"

    # Verify embedding_client implements the interface (duck typing)
    assert hasattr(container.embedding_client, "embed_batch"), \
        "embedding_client must have embed_batch method"
    assert hasattr(container.embedding_client, "get_dimension"), \
        "embedding_client must have get_dimension method"

    # Verify vector_store implements the interface (duck typing)
    assert hasattr(container.vector_store, "search"), \
        "vector_store must have search method"
    assert hasattr(container.vector_store, "upsert"), \
        "vector_store must have upsert method"

    # Verify metadata_store is IndexMetadataStore
    assert isinstance(container.metadata_store, IndexMetadataStore), \
        "metadata_store must be IndexMetadataStore"

    # Verify file_scanner is FileScanner
    assert isinstance(container.file_scanner, FileScanner), \
        "file_scanner must be FileScanner"

    # Verify chunker is Chunker
    assert isinstance(container.chunker, Chunker), \
        "chunker must be Chunker"


def test_services_container_dataclass_has_expected_fields():
    """
    **Feature: service-initialization-refactor, Property 3: ServicesContainer Completeness**
    **Validates: Requirements 1.4**

    The ServicesContainer dataclass SHALL define all expected fields.
    """
    expected_fields = {
        "config",
        "embedding_client",
        "vector_store",
        "metadata_store",
        "file_scanner",
        "chunker",
        "reranker",
    }

    actual_fields = {f.name for f in fields(ServicesContainer)}

    assert expected_fields == actual_fields, \
        f"ServicesContainer fields mismatch. Expected: {expected_fields}, Got: {actual_fields}"


def test_services_container_reranker_is_optional():
    """
    **Feature: service-initialization-refactor, Property 3: ServicesContainer Completeness**
    **Validates: Requirements 1.4**

    The reranker field SHALL be optional (can be None).
    """
    # Create minimal container with None reranker
    config = ACIConfig()
    embedding_client = LocalEmbeddingClient()
    vector_store = InMemoryVectorStore()
    metadata_store = IndexMetadataStore(":memory:")
    file_scanner = FileScanner(extensions={".py"})
    chunker = create_chunker()

    container = ServicesContainer(
        config=config,
        embedding_client=embedding_client,
        vector_store=vector_store,
        metadata_store=metadata_store,
        file_scanner=file_scanner,
        chunker=chunker,
        reranker=None,  # Explicitly None
    )

    # Should not raise - reranker is optional
    assert container.reranker is None

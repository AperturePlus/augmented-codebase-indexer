"""
Property-based tests for ACIConfig round-trip serialization.

**Feature: codebase-semantic-search, Property 20: Configuration Round-Trip**
**Validates: Requirements 7.5**
"""

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.core.config import (
    ACIConfig,
    EmbeddingConfig,
    IndexingConfig,
    LoggingConfig,
    SearchConfig,
    VectorStoreConfig,
)

# Strategies for generating valid configuration values
safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        blacklist_characters="\x00\n\r\t",
    ),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

safe_url = st.from_regex(r"https?://[a-z0-9]+(\.[a-z0-9]+)*(:[0-9]+)?(/[a-z0-9]*)*", fullmatch=True)

file_extension = st.from_regex(r"\.[a-z]{1,5}", fullmatch=True)

ignore_pattern = st.from_regex(r"[a-zA-Z0-9_\-\*\.]+", fullmatch=True).filter(lambda s: len(s) > 0)

log_level = st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])


@st.composite
def embedding_config_strategy(draw):
    """Generate valid EmbeddingConfig instances."""
    return EmbeddingConfig(
        api_key=draw(safe_text),
        api_url=draw(safe_url),
        model=draw(safe_text),
        batch_size=draw(st.integers(min_value=1, max_value=1000)),
        max_retries=draw(st.integers(min_value=0, max_value=10)),
        timeout=draw(
            st.floats(min_value=0.1, max_value=300.0, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def vector_store_config_strategy(draw):
    """Generate valid VectorStoreConfig instances."""
    return VectorStoreConfig(
        host=draw(st.from_regex(r"[a-z0-9\-\.]+", fullmatch=True).filter(lambda s: len(s) > 0)),
        port=draw(st.integers(min_value=1, max_value=65535)),
        collection_name=draw(
            st.from_regex(r"[a-z0-9_]+", fullmatch=True).filter(lambda s: len(s) > 0)
        ),
        vector_size=draw(st.integers(min_value=1, max_value=4096)),
    )


@st.composite
def indexing_config_strategy(draw):
    """Generate valid IndexingConfig instances."""
    return IndexingConfig(
        file_extensions=draw(st.lists(file_extension, min_size=1, max_size=10, unique=True)),
        ignore_patterns=draw(st.lists(ignore_pattern, min_size=0, max_size=20)),
        max_chunk_tokens=draw(st.integers(min_value=100, max_value=32000)),
        chunk_overlap_lines=draw(st.integers(min_value=0, max_value=50)),
        max_workers=draw(st.integers(min_value=1, max_value=32)),
    )


@st.composite
def search_config_strategy(draw):
    """Generate valid SearchConfig instances."""
    return SearchConfig(
        default_limit=draw(st.integers(min_value=1, max_value=100)),
        use_rerank=draw(st.booleans()),
        rerank_model=draw(safe_text),
        rerank_api_key=draw(safe_text),
        rerank_api_url=draw(safe_url),
    )


@st.composite
def logging_config_strategy(draw):
    """Generate valid LoggingConfig instances."""
    return LoggingConfig(
        level=draw(log_level),
        format=draw(safe_text),
    )


@st.composite
def aci_config_strategy(draw):
    """Generate valid ACIConfig instances."""
    return ACIConfig(
        embedding=draw(embedding_config_strategy()),
        vector_store=draw(vector_store_config_strategy()),
        indexing=draw(indexing_config_strategy()),
        search=draw(search_config_strategy()),
        logging=draw(logging_config_strategy()),
    )


@given(config=aci_config_strategy())
@settings(max_examples=100)
def test_config_yaml_round_trip(config: ACIConfig):
    """
    **Feature: codebase-semantic-search, Property 20: Configuration Round-Trip**
    **Validates: Requirements 7.5**

    For any valid ACIConfig object, serializing to YAML and deserializing
    should produce an equivalent configuration object.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"

        # Serialize to YAML
        config.save(yaml_path)

        # Deserialize from YAML
        loaded_config = ACIConfig.from_file(yaml_path)

        # Verify equivalence
        assert config.to_dict() == loaded_config.to_dict()


@given(config=aci_config_strategy())
@settings(max_examples=100)
def test_config_json_round_trip(config: ACIConfig):
    """
    **Feature: codebase-semantic-search, Property 20: Configuration Round-Trip**
    **Validates: Requirements 7.5**

    For any valid ACIConfig object, serializing to JSON and deserializing
    should produce an equivalent configuration object.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "config.json"

        # Serialize to JSON
        config.save(json_path)

        # Deserialize from JSON
        loaded_config = ACIConfig.from_file(json_path)

        # Verify equivalence
        assert config.to_dict() == loaded_config.to_dict()


@given(config=aci_config_strategy())
@settings(max_examples=100)
def test_config_to_yaml_string_round_trip(config: ACIConfig):
    """
    **Feature: codebase-semantic-search, Property 20: Configuration Round-Trip**
    **Validates: Requirements 7.5**

    For any valid ACIConfig object, converting to YAML string and back
    should produce an equivalent configuration object.
    """
    # Serialize to YAML string
    yaml_str = config.to_yaml()

    # Deserialize from YAML string (via temp file)
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"
        yaml_path.write_text(yaml_str, encoding="utf-8")
        loaded_config = ACIConfig.from_file(yaml_path)

        # Verify equivalence
        assert config.to_dict() == loaded_config.to_dict()


@given(config=aci_config_strategy())
@settings(max_examples=100)
def test_config_to_json_string_round_trip(config: ACIConfig):
    """
    **Feature: codebase-semantic-search, Property 20: Configuration Round-Trip**
    **Validates: Requirements 7.5**

    For any valid ACIConfig object, converting to JSON string and back
    should produce an equivalent configuration object.
    """
    # Serialize to JSON string
    json_str = config.to_json()

    # Deserialize from JSON string (via temp file)
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "config.json"
        json_path.write_text(json_str, encoding="utf-8")
        loaded_config = ACIConfig.from_file(json_path)

        # Verify equivalence
        assert config.to_dict() == loaded_config.to_dict()


@given(config=aci_config_strategy())
@settings(max_examples=50)
def test_config_to_dict_safe_redacts_api_keys(config: ACIConfig):
    """
    **Feature: Security - Sensitive Information Protection**
    
    For any valid ACIConfig object with API keys, to_dict_safe() should
    redact sensitive information like API keys while preserving all other
    configuration values.
    """
    # Get the safe dictionary representation
    safe_dict = config.to_dict_safe()
    
    # Verify that API keys are redacted if they exist
    if config.embedding.api_key:
        assert safe_dict["embedding"]["api_key"] == "[REDACTED]", \
            "Embedding API key should be redacted in safe dict"
    
    if config.search.rerank_api_key:
        assert safe_dict["search"]["rerank_api_key"] == "[REDACTED]", \
            "Rerank API key should be redacted in safe dict"
    
    # Verify that non-sensitive fields are preserved
    assert safe_dict["embedding"]["api_url"] == config.embedding.api_url
    assert safe_dict["embedding"]["model"] == config.embedding.model
    assert safe_dict["vector_store"]["host"] == config.vector_store.host
    assert safe_dict["indexing"]["max_chunk_tokens"] == config.indexing.max_chunk_tokens


def test_config_to_dict_safe_empty_keys():
    """
    **Feature: Security - Sensitive Information Protection**
    
    When API keys are empty strings, to_dict_safe() should preserve them
    as empty strings rather than redacting.
    """
    config = ACIConfig(
        embedding=EmbeddingConfig(
            api_key="",
            api_url="https://api.example.com",
            model="test-model"
        )
    )
    
    safe_dict = config.to_dict_safe()
    
    # Empty API key should remain empty, not redacted
    assert safe_dict["embedding"]["api_key"] == ""


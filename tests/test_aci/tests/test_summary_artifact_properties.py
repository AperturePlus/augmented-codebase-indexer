"""
Property-based tests for SummaryArtifact.

**Feature: multi-granularity-indexing, Property 7: Summary artifact JSON round-trip**
**Validates: Requirements 3.4**
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aci.core.summary_artifact import ArtifactType, SummaryArtifact

# Strategies for generating valid SummaryArtifact fields
safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        blacklist_characters="\x00",
    ),
    min_size=0,
    max_size=200,
)

non_empty_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=100,
)

file_path = st.from_regex(r"[a-zA-Z0-9_/\-\.]+\.(py|js|ts|go|java|c|cpp|h)", fullmatch=True)

artifact_type = st.sampled_from(list(ArtifactType))

line_number = st.integers(min_value=0, max_value=100000)


# Strategy for metadata - JSON-serializable values only
json_primitive = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000000, max_value=1000000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    safe_text,
)

json_value = st.recursive(
    json_primitive,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=20,
            ),
            children,
            max_size=5,
        ),
    ),
    max_leaves=10,
)

metadata_strategy = st.dictionaries(
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Pc")),
        min_size=1,
        max_size=30,
    ),
    json_value,
    max_size=10,
)


@st.composite
def summary_artifact_strategy(draw):
    """Generate valid SummaryArtifact instances."""
    start = draw(line_number)
    end = draw(st.integers(min_value=start, max_value=start + 10000))

    return SummaryArtifact(
        artifact_id=draw(st.uuids().map(str)),
        file_path=draw(file_path),
        artifact_type=draw(artifact_type),
        name=draw(non_empty_text),
        content=draw(safe_text),
        start_line=start,
        end_line=end,
        metadata=draw(metadata_strategy),
    )


@given(artifact=summary_artifact_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_summary_artifact_json_round_trip(artifact: SummaryArtifact):
    """
    **Feature: multi-granularity-indexing, Property 7: Summary artifact JSON round-trip**
    **Validates: Requirements 3.4**

    For any valid SummaryArtifact, serializing to JSON and deserializing back
    should produce an equivalent artifact with all fields preserved.
    """
    # Serialize to JSON
    json_str = artifact.to_json()

    # Deserialize from JSON
    restored = SummaryArtifact.from_json(json_str)

    # Verify all fields are preserved
    assert restored.artifact_id == artifact.artifact_id
    assert restored.file_path == artifact.file_path
    assert restored.artifact_type == artifact.artifact_type
    assert restored.name == artifact.name
    assert restored.content == artifact.content
    assert restored.start_line == artifact.start_line
    assert restored.end_line == artifact.end_line
    assert restored.metadata == artifact.metadata


@given(artifact=summary_artifact_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_summary_artifact_dict_round_trip(artifact: SummaryArtifact):
    """
    **Feature: multi-granularity-indexing, Property 7: Summary artifact JSON round-trip**
    **Validates: Requirements 3.4**

    For any valid SummaryArtifact, converting to dict and back
    should produce an equivalent artifact.
    """
    # Convert to dict
    data = artifact.to_dict()

    # Restore from dict
    restored = SummaryArtifact.from_dict(data)

    # Verify equivalence
    assert restored.to_dict() == artifact.to_dict()


@given(artifact=summary_artifact_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_artifact_type_preserved_as_string(artifact: SummaryArtifact):
    """
    **Feature: multi-granularity-indexing, Property 7: Summary artifact JSON round-trip**
    **Validates: Requirements 3.4**

    For any SummaryArtifact, the artifact_type should be serialized as a string
    value (not the enum representation) for JSON compatibility.
    """
    data = artifact.to_dict()

    # artifact_type should be a string value, not enum repr
    assert isinstance(data["artifact_type"], str)
    assert data["artifact_type"] in [t.value for t in ArtifactType]

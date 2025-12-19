"""Shared strategies for vector store property tests."""

from hypothesis import HealthCheck, strategies as st

# Strategies for generating test data
chunk_id_strategy = st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=36)
file_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/._-"),
    min_size=1,
    max_size=100,
).filter(lambda x: x.strip() != "")
line_number_strategy = st.integers(min_value=1, max_value=10000)
content_strategy = st.text(alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")), min_size=0, max_size=500)
vector_strategy = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=128,
    max_size=128,
)


@st.composite
def chunk_payload_strategy(draw):
    """Generate a valid chunk payload."""
    start_line = draw(line_number_strategy)
    end_line = draw(st.integers(min_value=start_line, max_value=start_line + 500))

    return {
        "file_path": draw(file_path_strategy),
        "start_line": start_line,
        "end_line": end_line,
        "content": draw(content_strategy),
        "language": draw(st.sampled_from(["python", "javascript", "go", "unknown"])),
        "chunk_type": draw(st.sampled_from(["function", "class", "method", "fixed"])),
    }


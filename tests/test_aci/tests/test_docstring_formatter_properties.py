"""
Property-based tests for DocstringFormatter.

Uses Hypothesis to verify universal properties across all inputs.

Feature: comment-aware-search
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.core.docstring_formatter import DocstringFormatter

# =============================================================================
# Custom Strategies for Generating Docstrings
# =============================================================================

# Strategy for generating plain text content (no comment syntax)
plain_text_content = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs"),  # Letters, Numbers, Spaces only
        blacklist_characters="\x00\r/*#'\"",  # Exclude problematic chars
    ),
    min_size=1,
    max_size=200,
).filter(lambda x: x.strip() and len(x.strip()) > 0)


@st.composite
def jsdoc_comment(draw):
    """Generate a JSDoc-style comment /** ... */."""
    content = draw(plain_text_content)
    return f"/** {content} */"


@st.composite
def go_doc_comment(draw):
    """Generate Go-style doc comments // ..."""
    content = draw(plain_text_content)
    # Single line, no newlines in content
    safe_content = content.replace("\n", " ").strip()
    return f"// {safe_content}"


@st.composite
def python_docstring(draw):
    """Generate Python-style docstrings."""
    content = draw(plain_text_content)
    return f'"""{content}"""'


@st.composite
def doxygen_line_comment(draw):
    """Generate Doxygen line comments /// ..."""
    content = draw(plain_text_content)
    safe_content = content.replace("\n", " ").strip()
    return f"/// {safe_content}"


# =============================================================================
# Property 7: Round-Trip Consistency Tests
# =============================================================================

class TestRoundTripProperty:
    """
    # Feature: comment-aware-search, Property 7: Docstring Round-Trip Consistency

    For any valid docstring: normalize(pretty_print(normalize(d))) == normalize(d)

    **Validates: Requirements 5.3**
    """

    def setup_method(self):
        self.formatter = DocstringFormatter()

    @given(docstring=jsdoc_comment())
    @settings(max_examples=100)
    def test_round_trip_jsdoc(self, docstring: str):
        """Test round-trip for JSDoc comments."""
        normalized = self.formatter.normalize(docstring, "javascript")
        assume(normalized)

        pretty = self.formatter.pretty_print(normalized)
        normalized_again = self.formatter.normalize(pretty, "unknown")

        assert normalized == normalized_again

    @given(docstring=go_doc_comment())
    @settings(max_examples=100)
    def test_round_trip_go(self, docstring: str):
        """Test round-trip for Go doc comments."""
        normalized = self.formatter.normalize(docstring, "go")
        assume(normalized)

        pretty = self.formatter.pretty_print(normalized)
        normalized_again = self.formatter.normalize(pretty, "unknown")

        assert normalized == normalized_again

    @given(docstring=python_docstring())
    @settings(max_examples=100)
    def test_round_trip_python(self, docstring: str):
        """Test round-trip for Python docstrings."""
        normalized = self.formatter.normalize(docstring, "python")
        assume(normalized)

        pretty = self.formatter.pretty_print(normalized)
        normalized_again = self.formatter.normalize(pretty, "unknown")

        assert normalized == normalized_again

    @given(docstring=doxygen_line_comment())
    @settings(max_examples=100)
    def test_round_trip_doxygen(self, docstring: str):
        """Test round-trip for Doxygen line comments."""
        normalized = self.formatter.normalize(docstring, "cpp")
        assume(normalized)

        pretty = self.formatter.pretty_print(normalized)
        normalized_again = self.formatter.normalize(pretty, "unknown")

        assert normalized == normalized_again


# =============================================================================
# Property 6: Normalization Consistency Tests
# =============================================================================

class TestNormalizationConsistencyProperty:
    """
    # Feature: comment-aware-search, Property 6: Docstring Normalization Consistency

    Normalized output SHALL be plain text with comment syntax removed.

    **Validates: Requirements 5.1, 5.2**
    """

    def setup_method(self):
        self.formatter = DocstringFormatter()

    @given(content=plain_text_content)
    @settings(max_examples=100)
    def test_jsdoc_syntax_removed(self, content: str):
        """Test that JSDoc syntax is removed."""
        docstring = f"/** {content} */"
        normalized = self.formatter.normalize(docstring, "javascript")

        assert "/**" not in normalized
        assert "*/" not in normalized

    @given(content=plain_text_content)
    @settings(max_examples=100)
    def test_go_syntax_removed(self, content: str):
        """Test that Go comment syntax is removed."""
        safe_content = content.replace("\n", " ").strip()
        docstring = f"// {safe_content}"
        normalized = self.formatter.normalize(docstring, "go")

        assert not normalized.startswith("//")

    @given(content=plain_text_content)
    @settings(max_examples=100)
    def test_python_quotes_removed(self, content: str):
        """Test that Python docstring quotes are removed."""
        docstring = f'"""{content}"""'
        normalized = self.formatter.normalize(docstring, "python")

        assert '"""' not in normalized

    @given(content=plain_text_content)
    @settings(max_examples=100)
    def test_equivalent_content_similar_output(self, content: str):
        """Test that same content in different formats produces similar output."""
        safe_content = content.replace("\n", " ").strip()

        jsdoc = f"/** {safe_content} */"
        go_doc = f"// {safe_content}"
        py_doc = f'"""{safe_content}"""'

        norm_js = self.formatter.normalize(jsdoc, "javascript")
        norm_go = self.formatter.normalize(go_doc, "go")
        norm_py = self.formatter.normalize(py_doc, "python")

        # All should contain the core content
        assert safe_content in norm_js or norm_js == safe_content
        assert safe_content in norm_go or norm_go == safe_content
        assert safe_content in norm_py or norm_py == safe_content


# =============================================================================
# Format for Embedding Property Tests
# =============================================================================

class TestFormatForEmbeddingProperty:
    """Property tests for format_for_embedding method."""

    def setup_method(self):
        self.formatter = DocstringFormatter()

    @given(
        docstring=jsdoc_comment(),
        code=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=50)
    def test_delimiter_present_when_docstring_exists(self, docstring: str, code: str):
        """Test that delimiter is present when docstring normalizes to non-empty."""
        result = self.formatter.format_for_embedding(docstring, code, "javascript")
        normalized = self.formatter.normalize(docstring, "javascript")

        if normalized:
            assert self.formatter.DELIMITER in result

    @given(code=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    @settings(max_examples=50)
    def test_no_delimiter_without_docstring(self, code: str):
        """Test that no delimiter when docstring is empty."""
        result = self.formatter.format_for_embedding("", code, "javascript")
        assert result == code
        assert self.formatter.DELIMITER not in result

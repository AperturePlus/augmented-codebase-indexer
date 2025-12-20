"""
Unit tests for DocstringFormatter.

Tests normalization, formatting, and pretty-printing of docstrings
from various programming languages.
"""


from aci.core.docstring_formatter import DocstringFormatter


class TestDocstringFormatterNormalize:
    """Test docstring normalization for different languages."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DocstringFormatter()

    def test_normalize_jsdoc_single_line(self):
        """Test normalizing single-line JSDoc comment."""
        docstring = "/** Simple description */"
        result = self.formatter.normalize(docstring, "javascript")
        assert result == "Simple description"

    def test_normalize_jsdoc_multi_line(self):
        """Test normalizing multi-line JSDoc comment."""
        docstring = """/**
         * Authenticates a user.
         * @param username - The user's login name
         * @param password - The user's password
         * @returns Promise<string> - JWT token
         */"""
        result = self.formatter.normalize(docstring, "javascript")
        assert "Authenticates a user" in result
        assert "@param username" in result
        assert "@param password" in result
        assert "@returns Promise<string>" in result

    def test_normalize_jsdoc_with_asterisks(self):
        """Test normalizing JSDoc with leading asterisks on each line."""
        docstring = """/**
         * Line 1
         * Line 2
         * Line 3
         */"""
        result = self.formatter.normalize(docstring, "javascript")
        assert result == "Line 1\nLine 2\nLine 3"

    def test_normalize_go_doc_single_line(self):
        """Test normalizing single-line Go doc comment."""
        docstring = "// HandleRequest processes an HTTP request"
        result = self.formatter.normalize(docstring, "go")
        assert result == "HandleRequest processes an HTTP request"

    def test_normalize_go_doc_multi_line(self):
        """Test normalizing multi-line Go doc comment."""
        docstring = """// User represents a user in the system.
// It contains authentication and profile information.
// The struct is used throughout the application."""
        result = self.formatter.normalize(docstring, "go")
        expected = "User represents a user in the system.\nIt contains authentication and profile information.\nThe struct is used throughout the application."
        assert result == expected

    def test_normalize_python_docstring_triple_double(self):
        """Test normalizing Python docstring with triple double quotes."""
        docstring = '''"""
        Calculate the sum of two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            The sum of a and b
        """'''
        result = self.formatter.normalize(docstring, "python")
        assert "Calculate the sum of two numbers" in result
        assert "Args:" in result
        assert "Returns:" in result

    def test_normalize_python_docstring_triple_single(self):
        """Test normalizing Python docstring with triple single quotes."""
        docstring = """'''Simple docstring'''"""
        result = self.formatter.normalize(docstring, "python")
        assert result == "Simple docstring"

    def test_normalize_javadoc(self):
        """Test normalizing Javadoc comment."""
        docstring = """/**
         * Processes a payment transaction.
         * @param amount The payment amount
         * @param currency The currency code
         * @return Transaction ID
         * @throws PaymentException if payment fails
         */"""
        result = self.formatter.normalize(docstring, "java")
        assert "Processes a payment transaction" in result
        assert "@param amount" in result
        assert "@throws PaymentException" in result

    def test_normalize_doxygen_block(self):
        """Test normalizing Doxygen block comment."""
        docstring = """/**
         * @brief Initializes the system
         * @param config Configuration object
         * @return Status code
         */"""
        result = self.formatter.normalize(docstring, "cpp")
        assert "@brief Initializes the system" in result
        assert "@param config" in result

    def test_normalize_doxygen_line(self):
        """Test normalizing Doxygen line comments."""
        docstring = """/// Calculate the square root
/// @param x Input value
/// @return Square root of x"""
        result = self.formatter.normalize(docstring, "cpp")
        assert "Calculate the square root" in result
        assert "@param x" in result

    def test_normalize_empty_docstring(self):
        """Test normalizing empty docstring."""
        assert self.formatter.normalize("", "python") == ""
        assert self.formatter.normalize("   ", "javascript") == ""
        assert self.formatter.normalize("/** */", "javascript") == ""

    def test_normalize_unicode_content(self):
        """Test normalizing docstring with Unicode characters."""
        docstring = "/** 处理用户请求 🚀 */"""
        result = self.formatter.normalize(docstring, "javascript")
        assert result == "处理用户请求 🚀"

    def test_normalize_auto_detect_jsdoc(self):
        """Test auto-detection of JSDoc format."""
        docstring = "/** Auto-detected JSDoc */"
        result = self.formatter.normalize(docstring, "unknown")
        assert result == "Auto-detected JSDoc"

    def test_normalize_auto_detect_line_comment(self):
        """Test auto-detection of line comment format."""
        docstring = "// Auto-detected line comment"
        result = self.formatter.normalize(docstring, "unknown")
        assert result == "Auto-detected line comment"

    def test_normalize_auto_detect_python(self):
        """Test auto-detection of Python docstring."""
        docstring = '"""Auto-detected Python docstring"""'
        result = self.formatter.normalize(docstring, "unknown")
        assert result == "Auto-detected Python docstring"

    def test_normalize_preserves_structure(self):
        """Test that normalization preserves paragraph structure."""
        docstring = """/**
         * First paragraph with important info.
         *
         * Second paragraph with more details.
         *
         * Third paragraph with examples.
         */"""
        result = self.formatter.normalize(docstring, "javascript")
        # Should have double newlines between paragraphs
        assert "First paragraph with important info.\n\nSecond paragraph" in result


class TestDocstringFormatterFormatForEmbedding:
    """Test formatting docstrings for embedding generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DocstringFormatter()

    def test_format_with_docstring(self):
        """Test formatting with both docstring and code."""
        docstring = "/** Adds two numbers */"
        code = "function add(a, b) { return a + b; }"
        result = self.formatter.format_for_embedding(docstring, code, "javascript")

        assert "Adds two numbers" in result
        assert "---" in result
        assert "function add(a, b)" in result
        assert result.index("Adds two numbers") < result.index("---")
        assert result.index("---") < result.index("function add")

    def test_format_without_docstring(self):
        """Test formatting with no docstring."""
        code = "function add(a, b) { return a + b; }"
        result = self.formatter.format_for_embedding("", code, "javascript")
        assert result == code
        assert "---" not in result

    def test_format_with_empty_docstring_after_normalization(self):
        """Test formatting when docstring is empty after normalization."""
        docstring = "/** */"  # Empty JSDoc
        code = "function test() {}"
        result = self.formatter.format_for_embedding(docstring, code, "javascript")
        assert result == code
        assert "---" not in result

    def test_format_delimiter_is_correct(self):
        """Test that the delimiter is exactly as specified."""
        docstring = "/** Test */"
        code = "code"
        result = self.formatter.format_for_embedding(docstring, code, "javascript")
        assert "\n---\n" in result

    def test_format_with_multiline_docstring(self):
        """Test formatting with multi-line docstring."""
        docstring = """/**
         * Complex function
         * @param x Input
         * @returns Output
         */"""
        code = "function complex(x) { return x * 2; }"
        result = self.formatter.format_for_embedding(docstring, code, "javascript")

        result.split("\n")
        # Should have docstring, delimiter, and code
        assert "Complex function" in result
        assert "@param x" in result
        assert "---" in result
        assert "function complex" in result


class TestDocstringFormatterPrettyPrint:
    """Test pretty-printing of normalized docstrings."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DocstringFormatter()

    def test_pretty_print_simple(self):
        """Test pretty-printing simple docstring."""
        docstring = "Simple docstring"
        result = self.formatter.pretty_print(docstring)
        assert result == "Simple docstring"

    def test_pretty_print_multiline(self):
        """Test pretty-printing multi-line docstring."""
        docstring = "Line 1\nLine 2\nLine 3"
        result = self.formatter.pretty_print(docstring)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_pretty_print_removes_leading_blank_lines(self):
        """Test that pretty-print removes leading blank lines."""
        docstring = "\n\nContent here"
        result = self.formatter.pretty_print(docstring)
        assert result == "Content here"

    def test_pretty_print_removes_trailing_blank_lines(self):
        """Test that pretty-print removes trailing blank lines."""
        docstring = "Content here\n\n"
        result = self.formatter.pretty_print(docstring)
        assert result == "Content here"

    def test_pretty_print_empty(self):
        """Test pretty-printing empty string."""
        assert self.formatter.pretty_print("") == ""
        assert self.formatter.pretty_print("   ") == ""

    def test_pretty_print_strips_trailing_whitespace(self):
        """Test that pretty-print strips trailing whitespace from lines."""
        docstring = "Line 1   \nLine 2  \nLine 3"
        result = self.formatter.pretty_print(docstring)
        assert result == "Line 1\nLine 2\nLine 3"


class TestDocstringFormatterRoundTrip:
    """Test round-trip consistency: normalize(pretty_print(normalize(d))) == normalize(d)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DocstringFormatter()

    def test_round_trip_jsdoc(self):
        """Test round-trip with JSDoc."""
        original = "/** Test function */"
        normalized = self.formatter.normalize(original, "javascript")
        pretty = self.formatter.pretty_print(normalized)
        normalized_again = self.formatter.normalize(pretty, "javascript")
        assert normalized == normalized_again

    def test_round_trip_go_doc(self):
        """Test round-trip with Go doc comment."""
        original = "// HandleRequest processes requests"
        normalized = self.formatter.normalize(original, "go")
        pretty = self.formatter.pretty_print(normalized)
        normalized_again = self.formatter.normalize(pretty, "go")
        assert normalized == normalized_again

    def test_round_trip_python(self):
        """Test round-trip with Python docstring."""
        original = '"""Calculate sum"""'
        normalized = self.formatter.normalize(original, "python")
        pretty = self.formatter.pretty_print(normalized)
        normalized_again = self.formatter.normalize(pretty, "python")
        assert normalized == normalized_again

    def test_round_trip_multiline(self):
        """Test round-trip with multi-line docstring."""
        original = """/**
         * Line 1
         * Line 2
         * Line 3
         */"""
        normalized = self.formatter.normalize(original, "javascript")
        pretty = self.formatter.pretty_print(normalized)
        normalized_again = self.formatter.normalize(pretty, "javascript")
        assert normalized == normalized_again

    def test_round_trip_with_unicode(self):
        """Test round-trip with Unicode content."""
        original = "/** 测试函数 🎉 */"
        normalized = self.formatter.normalize(original, "javascript")
        pretty = self.formatter.pretty_print(normalized)
        normalized_again = self.formatter.normalize(pretty, "javascript")
        assert normalized == normalized_again

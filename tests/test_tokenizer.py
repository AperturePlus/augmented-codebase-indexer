"""
Tests for the Tokenizer module.
"""

import pytest
from aci.core.tokenizer import (
    TokenizerInterface,
    TiktokenTokenizer,
    get_default_tokenizer,
)


class TestTiktokenTokenizer:
    """Unit tests for TiktokenTokenizer."""

    def test_implements_interface(self):
        """Verify TiktokenTokenizer implements TokenizerInterface."""
        tokenizer = TiktokenTokenizer()
        assert isinstance(tokenizer, TokenizerInterface)

    def test_count_tokens_empty_string(self):
        """Empty string should return 0 tokens."""
        tokenizer = TiktokenTokenizer()
        assert tokenizer.count_tokens("") == 0

    def test_count_tokens_simple_text(self):
        """Simple text should return positive token count."""
        tokenizer = TiktokenTokenizer()
        count = tokenizer.count_tokens("Hello, world!")
        assert count > 0

    def test_count_tokens_code(self):
        """Code should be tokenized correctly."""
        tokenizer = TiktokenTokenizer()
        code = "def hello():\n    print('Hello')"
        count = tokenizer.count_tokens(code)
        assert count > 0

    def test_truncate_empty_string(self):
        """Empty string should return empty string."""
        tokenizer = TiktokenTokenizer()
        assert tokenizer.truncate_to_tokens("", 100) == ""

    def test_truncate_zero_max_tokens(self):
        """Zero max_tokens should return empty string."""
        tokenizer = TiktokenTokenizer()
        assert tokenizer.truncate_to_tokens("Hello, world!", 0) == ""

    def test_truncate_negative_max_tokens(self):
        """Negative max_tokens should return empty string."""
        tokenizer = TiktokenTokenizer()
        assert tokenizer.truncate_to_tokens("Hello, world!", -5) == ""

    def test_truncate_text_fits(self):
        """Text that fits should be returned unchanged."""
        tokenizer = TiktokenTokenizer()
        text = "Hello, world!"
        result = tokenizer.truncate_to_tokens(text, 1000)
        assert result == text

    def test_truncate_preserves_line_integrity(self):
        """Truncation should not cut in the middle of a line."""
        tokenizer = TiktokenTokenizer()
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        
        # Get a max_tokens that will require truncation
        total_tokens = tokenizer.count_tokens(text)
        max_tokens = total_tokens // 2
        
        result = tokenizer.truncate_to_tokens(text, max_tokens)
        
        # Result should end with a complete line (no partial lines)
        assert result.endswith("Line 1") or result.endswith("Line 2") or result.endswith("Line 3")
        # Result should not contain partial text
        for line in result.split("\n"):
            assert line.startswith("Line ")

    def test_truncate_respects_token_limit(self):
        """Truncated text should not exceed max_tokens."""
        tokenizer = TiktokenTokenizer()
        text = "\n".join([f"This is line number {i} with some content" for i in range(100)])
        max_tokens = 50
        
        result = tokenizer.truncate_to_tokens(text, max_tokens)
        result_tokens = tokenizer.count_tokens(result)
        
        assert result_tokens <= max_tokens

    def test_truncate_multiline_code(self):
        """Truncation should work correctly with code."""
        tokenizer = TiktokenTokenizer()
        code = """def function_one():
    print("Hello")
    return 1

def function_two():
    print("World")
    return 2

def function_three():
    print("Test")
    return 3
"""
        # Use a small token limit
        max_tokens = 20
        result = tokenizer.truncate_to_tokens(code, max_tokens)
        
        # Should not exceed limit
        assert tokenizer.count_tokens(result) <= max_tokens
        # Should contain complete lines only
        lines = result.split("\n")
        for line in lines:
            # Each line should be a valid Python line (not cut mid-statement)
            assert not line.endswith("pri")  # Not cut in middle of "print"


class TestGetDefaultTokenizer:
    """Tests for get_default_tokenizer factory function."""

    def test_returns_tokenizer_interface(self):
        """Should return a TokenizerInterface instance."""
        tokenizer = get_default_tokenizer()
        assert isinstance(tokenizer, TokenizerInterface)

    def test_returns_tiktoken_tokenizer(self):
        """Should return a TiktokenTokenizer instance."""
        tokenizer = get_default_tokenizer()
        assert isinstance(tokenizer, TiktokenTokenizer)

    def test_uses_cl100k_base_encoding(self):
        """Should use cl100k_base encoding by default."""
        tokenizer = get_default_tokenizer()
        assert tokenizer._encoding_name == "cl100k_base"

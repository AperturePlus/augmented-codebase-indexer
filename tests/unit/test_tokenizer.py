"""
Tests for the Tokenizer module.
"""

import pytest

from aci.core.tokenizer import (
    CharacterTokenizer,
    SimpleTokenizer,
    TiktokenTokenizer,
    TokenizerInterface,
    get_default_tokenizer,
)


class FakeEncoding:
    """Offline-safe encoding stub for unit tests."""

    def encode(self, text: str) -> list[str]:
        if not text:
            return []
        # Approximate tokenization: split on whitespace boundaries
        return text.replace("\n", " \n ").split()


def make_tiktoken_tokenizer() -> TiktokenTokenizer:
    tokenizer = TiktokenTokenizer()
    tokenizer._encoding = FakeEncoding()
    return tokenizer


class TestTiktokenTokenizer:
    """Unit tests for TiktokenTokenizer."""

    def test_implements_interface(self):
        tokenizer = make_tiktoken_tokenizer()
        assert isinstance(tokenizer, TokenizerInterface)

    def test_count_tokens_empty_string(self):
        tokenizer = make_tiktoken_tokenizer()
        assert tokenizer.count_tokens("") == 0

    def test_count_tokens_simple_text(self):
        tokenizer = make_tiktoken_tokenizer()
        assert tokenizer.count_tokens("Hello, world!") > 0

    def test_count_tokens_code(self):
        tokenizer = make_tiktoken_tokenizer()
        assert tokenizer.count_tokens("def hello():\n    print('Hello')") > 0

    def test_truncate_empty_string(self):
        tokenizer = make_tiktoken_tokenizer()
        assert tokenizer.truncate_to_tokens("", 100) == ""

    def test_truncate_zero_max_tokens(self):
        tokenizer = make_tiktoken_tokenizer()
        assert tokenizer.truncate_to_tokens("Hello, world!", 0) == ""

    def test_truncate_negative_max_tokens(self):
        tokenizer = make_tiktoken_tokenizer()
        assert tokenizer.truncate_to_tokens("Hello, world!", -5) == ""

    def test_truncate_text_fits(self):
        tokenizer = make_tiktoken_tokenizer()
        text = "Hello, world!"
        assert tokenizer.truncate_to_tokens(text, 1000) == text

    def test_truncate_preserves_line_integrity(self):
        tokenizer = make_tiktoken_tokenizer()
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        max_tokens = max(1, tokenizer.count_tokens(text) // 2)

        result = tokenizer.truncate_to_tokens(text, max_tokens)
        for line in result.split("\n"):
            assert line.startswith("Line ")

    def test_truncate_respects_token_limit(self):
        tokenizer = make_tiktoken_tokenizer()
        text = "\n".join([f"This is line number {i} with some content" for i in range(100)])
        max_tokens = 50
        result = tokenizer.truncate_to_tokens(text, max_tokens)
        assert tokenizer.count_tokens(result) <= max_tokens


class TestAlternativeTokenizers:
    def test_character_tokenizer_counts_and_truncates(self):
        tokenizer = CharacterTokenizer(chars_per_token=4)
        text = "abcd\nefgh\nijkl"
        assert tokenizer.count_tokens(text) == 4
        truncated = tokenizer.truncate_to_tokens(text, 2)
        assert truncated == "abcd"
        assert tokenizer.count_tokens(truncated) <= 2

    def test_simple_tokenizer_counts_and_truncates(self):
        tokenizer = SimpleTokenizer()
        text = "one two\nthree four five"
        assert tokenizer.count_tokens(text) == 5
        truncated = tokenizer.truncate_to_tokens(text, 2)
        assert truncated == "one two"
        assert tokenizer.count_tokens(truncated) <= 2


class TestGetDefaultTokenizer:
    def test_returns_tokenizer_interface(self):
        assert isinstance(get_default_tokenizer(), TokenizerInterface)

    def test_returns_tiktoken_tokenizer(self):
        assert isinstance(get_default_tokenizer("tiktoken"), TiktokenTokenizer)

    def test_returns_character_tokenizer(self):
        assert isinstance(get_default_tokenizer("character"), CharacterTokenizer)

    def test_returns_simple_tokenizer(self):
        assert isinstance(get_default_tokenizer("simple"), SimpleTokenizer)

    def test_uses_cl100k_base_encoding(self):
        tokenizer = get_default_tokenizer()
        assert tokenizer._encoding_name == "cl100k_base"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unsupported tokenizer strategy"):
            get_default_tokenizer("bert")

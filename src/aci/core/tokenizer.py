"""
Tokenizer module for token counting and text truncation.

Uses tiktoken library for accurate token counting compatible with OpenAI models.
"""

from abc import ABC, abstractmethod
from math import ceil

import tiktoken


class TokenizerInterface(ABC):
    """Abstract interface for tokenization operations."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The text to tokenize.

        Returns:
            The number of tokens in the text.
        """
        pass

    @abstractmethod
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within the specified token limit while preserving line integrity.

        The truncation will not cut in the middle of a line. If a line would cause
        the token count to exceed max_tokens, that line and all subsequent lines
        are excluded.

        Args:
            text: The text to truncate.
            max_tokens: The maximum number of tokens allowed.

        Returns:
            The truncated text with complete lines only.
        """
        pass


class TiktokenTokenizer(TokenizerInterface):
    """
    Tokenizer implementation using tiktoken library.

    Supports various encoding schemes used by OpenAI models.
    Default encoding is 'cl100k_base' which is used by text-embedding-3-* models.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the tokenizer with the specified encoding.

        Args:
            encoding_name: The tiktoken encoding name. Common options:
                - 'cl100k_base': Used by text-embedding-3-*, gpt-4, gpt-3.5-turbo
                - 'p50k_base': Used by older models like text-davinci-003
                - 'r50k_base': Used by older GPT-3 models
        """
        self._encoding_name = encoding_name
        self._encoding: tiktoken.Encoding | None = None

    @property
    def encoding(self) -> tiktoken.Encoding:
        """Lazy-load the encoding to avoid initialization overhead."""
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding(self._encoding_name)
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The text to tokenize.

        Returns:
            The number of tokens in the text.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within the specified token limit while preserving line integrity.

        This method ensures that:
        1. The returned text does not exceed max_tokens
        2. Lines are not cut in the middle
        3. If the entire text fits, it is returned unchanged

        Args:
            text: The text to truncate.
            max_tokens: The maximum number of tokens allowed.

        Returns:
            The truncated text with complete lines only.
        """
        if not text:
            return ""

        if max_tokens <= 0:
            return ""

        # Fast path: if entire text fits, return as-is
        total_tokens = self.count_tokens(text)
        if total_tokens <= max_tokens:
            return text

        # Split into lines and accumulate until we exceed the limit
        lines = text.split("\n")
        result_lines: list[str] = []
        current_tokens = 0

        for line in lines:
            # Calculate tokens for this line (including newline if not first line)
            if result_lines:
                # Account for the newline character that will join lines
                line_with_newline = "\n" + line
                line_tokens = self.count_tokens(line_with_newline)
            else:
                line_tokens = self.count_tokens(line)

            # Check if adding this line would exceed the limit
            if current_tokens + line_tokens > max_tokens:
                break

            result_lines.append(line)
            current_tokens += line_tokens

        return "\n".join(result_lines)


class CharacterTokenizer(TokenizerInterface):
    """Conservative tokenizer that estimates tokens using character length."""

    def __init__(self, chars_per_token: int = 4):
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be greater than 0")
        self._chars_per_token = chars_per_token

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return ceil(len(text) / self._chars_per_token)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""

        if self.count_tokens(text) <= max_tokens:
            return text

        lines = text.split("\n")
        result_lines: list[str] = []
        current_tokens = 0

        for line in lines:
            line_with_newline = f"\n{line}" if result_lines else line
            line_tokens = self.count_tokens(line_with_newline)

            if current_tokens + line_tokens > max_tokens:
                break

            result_lines.append(line)
            current_tokens += line_tokens

        return "\n".join(result_lines)


class SimpleTokenizer(TokenizerInterface):
    """Simple whitespace tokenizer primarily for generic non-BPE models."""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""

        if self.count_tokens(text) <= max_tokens:
            return text

        lines = text.split("\n")
        result_lines: list[str] = []
        current_tokens = 0

        for line in lines:
            line_with_newline = f"\n{line}" if result_lines else line
            line_tokens = self.count_tokens(line_with_newline)

            if current_tokens + line_tokens > max_tokens:
                break

            result_lines.append(line)
            current_tokens += line_tokens

        return "\n".join(result_lines)


def get_default_tokenizer(strategy: str = "tiktoken") -> TokenizerInterface:
    """
    Get the default tokenizer instance.

    Returns:
        A tokenizer implementation matching the configured strategy.
    """
    normalized = strategy.strip().lower()
    if normalized == "tiktoken":
        return TiktokenTokenizer(encoding_name="cl100k_base")
    if normalized == "character":
        return CharacterTokenizer(chars_per_token=4)
    if normalized == "simple":
        return SimpleTokenizer()
    raise ValueError(
        f"Unsupported tokenizer strategy '{strategy}'. "
        "Expected one of: tiktoken, character, simple"
    )

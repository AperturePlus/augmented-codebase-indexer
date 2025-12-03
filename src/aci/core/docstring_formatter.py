"""
Docstring Formatter - Normalizes and formats documentation strings.

This module provides functionality to normalize docstrings from different
programming languages (Python, JavaScript/TypeScript, Go, Java, C/C++) into
a canonical format for embedding generation and search.
"""

import logging
import re

logger = logging.getLogger(__name__)


class DocstringFormatter:
    """
    Formats docstrings for inclusion in chunk content.

    Responsibilities:
    - Normalize docstrings from different languages to a canonical format
    - Format content as "docstring + delimiter + code" for embedding
    - Provide pretty-printing for round-trip testing

    The formatter handles various comment syntaxes:
    - Python: triple-quoted strings
    - JavaScript/TypeScript: JSDoc (/** ... */)
    - Go: line comments (//)
    - Java: Javadoc (/** ... */)
    - C/C++: Doxygen (/** ... */, ///)
    """

    # Clear separator between docstring and code
    DELIMITER = "\n---\n"

    # Comment syntax patterns for different languages
    _BLOCK_COMMENT_START = re.compile(r"^\s*/\*\*?\s*")  # /** or /*
    _BLOCK_COMMENT_END = re.compile(r"\s*\*/\s*$")  # */
    _DOXYGEN_LINE_PREFIX = re.compile(r"^\s*///\s?")  # /// (Doxygen line comment)
    _GO_LINE_PREFIX = re.compile(r"^\s*//\s?")  # // (Go doc comment)

    def normalize(self, docstring: str, language: str = "unknown") -> str:
        """
        Normalize a docstring to canonical text format.

        Strips comment syntax while preserving semantic content and structure.
        Handles language-specific comment formats and documentation tags.

        Args:
            docstring: Raw docstring with comment syntax
            language: Programming language identifier (for language-specific handling)

        Returns:
            Normalized plain text docstring with comment syntax removed

        Examples:
            >>> formatter = DocstringFormatter()
            >>> formatter.normalize("/** Hello world */", "javascript")
            'Hello world'
            >>> formatter.normalize("// Line 1\\n// Line 2", "go")
            'Line 1\\nLine 2'
        """
        if not docstring or not docstring.strip():
            return ""

        # Handle different comment syntaxes based on language
        if language in ("javascript", "typescript", "java"):
            return self._normalize_block_comment(docstring)
        elif language in ("c", "cpp"):
            return self._normalize_doxygen(docstring)
        elif language == "go":
            return self._normalize_go_doc(docstring)
        elif language == "python":
            return self._normalize_python_docstring(docstring)
        else:
            # Unknown language: try to detect and normalize
            return self._normalize_auto_detect(docstring)

    def _normalize_block_comment(self, docstring: str) -> str:
        """
        Normalize block comments (/** ... */).

        Used for JSDoc and Javadoc comments.

        Args:
            docstring: Block comment string

        Returns:
            Normalized text without comment delimiters
        """
        lines = docstring.split("\n")
        normalized_lines = []

        for line in lines:
            # Remove /** or /* at start
            line = self._BLOCK_COMMENT_START.sub("", line)
            # Remove */ at end
            line = self._BLOCK_COMMENT_END.sub("", line)
            # Remove leading * from continuation lines (but preserve content after *)
            line = re.sub(r"^\s*\*\s?", "", line)

            normalized_lines.append(line.rstrip())

        # Join lines and clean up excessive blank lines
        result = "\n".join(normalized_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)  # Max 2 consecutive newlines
        return result.strip()

    def _normalize_doxygen(self, docstring: str) -> str:
        """
        Normalize Doxygen comments (/** ... */ or ///).

        Used for C/C++ documentation.

        Args:
            docstring: Doxygen comment string

        Returns:
            Normalized text without comment delimiters
        """
        stripped = docstring.strip()

        # Check if it's a line comment style (///)
        if stripped.startswith("///"):
            return self._normalize_doxygen_line(docstring)
        else:
            # Block comment style
            return self._normalize_block_comment(docstring)

    def _normalize_doxygen_line(self, docstring: str) -> str:
        """
        Normalize Doxygen line comments (///).

        Args:
            docstring: Doxygen line comment string

        Returns:
            Normalized text without /// prefix
        """
        lines = docstring.split("\n")
        normalized_lines = []

        for line in lines:
            # Remove /// prefix
            match = self._DOXYGEN_LINE_PREFIX.match(line)
            if match:
                line = line[match.end():]
            normalized_lines.append(line.rstrip())

        result = "\n".join(normalized_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def _normalize_go_doc(self, docstring: str) -> str:
        """
        Normalize Go doc comments (//).

        Args:
            docstring: Go doc comment string

        Returns:
            Normalized text without // prefix
        """
        lines = docstring.split("\n")
        normalized_lines = []

        for line in lines:
            # Remove // prefix
            match = self._GO_LINE_PREFIX.match(line)
            if match:
                line = line[match.end():]
            normalized_lines.append(line.rstrip())

        result = "\n".join(normalized_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def _normalize_python_docstring(self, docstring: str) -> str:
        """
        Normalize Python docstrings (triple-quoted strings).

        Args:
            docstring: Python docstring

        Returns:
            Normalized text without quotes
        """
        result = docstring.strip()

        # Remove """ or ''' at start and end
        if result.startswith('"""'):
            result = result[3:]
        elif result.startswith("'''"):
            result = result[3:]

        if result.endswith('"""'):
            result = result[:-3]
        elif result.endswith("'''"):
            result = result[:-3]

        # Clean up and normalize whitespace
        result = result.strip()
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result

    def _normalize_auto_detect(self, docstring: str) -> str:
        """
        Auto-detect comment syntax and normalize.

        Tries to intelligently detect the comment format when language is unknown.

        Args:
            docstring: Docstring with unknown format

        Returns:
            Best-effort normalized text
        """
        stripped = docstring.strip()

        # Check for block comments (JSDoc/Javadoc/Doxygen block)
        if stripped.startswith("/**") or stripped.startswith("/*"):
            return self._normalize_block_comment(docstring)

        # Check for Doxygen line comments (///)
        if stripped.startswith("///"):
            return self._normalize_doxygen_line(docstring)

        # Check for Go-style line comments (//)
        if stripped.startswith("//"):
            return self._normalize_go_doc(docstring)

        # Check for Python docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            return self._normalize_python_docstring(docstring)

        # No recognized syntax, return as-is (already plain text)
        return stripped

    def format_for_embedding(self, docstring: str, code: str, language: str = "unknown") -> str:
        """
        Format docstring and code for embedding generation.

        Combines normalized docstring with code using a clear delimiter.
        This format is used to generate vector embeddings that capture both
        documentation and implementation.

        Args:
            docstring: Raw docstring (will be normalized)
            code: Source code content
            language: Programming language identifier

        Returns:
            Formatted string: "{normalized_docstring}{DELIMITER}{code}"

        Examples:
            >>> formatter = DocstringFormatter()
            >>> result = formatter.format_for_embedding(
            ...     "/** Adds two numbers */",
            ...     "function add(a, b) { return a + b; }",
            ...     "javascript"
            ... )
            >>> print(result)
            Adds two numbers
            ---
            function add(a, b) { return a + b; }
        """
        if not docstring or not docstring.strip():
            return code

        normalized = self.normalize(docstring, language)

        if not normalized:
            return code

        return f"{normalized}{self.DELIMITER}{code}"

    def pretty_print(self, docstring: str) -> str:
        """
        Pretty-print a normalized docstring.

        Used for round-trip testing and debugging. The output should be
        semantically equivalent to the input when both are normalized.

        Args:
            docstring: Normalized docstring (plain text)

        Returns:
            Pretty-printed docstring

        Note:
            For round-trip consistency: normalize(pretty_print(normalize(d))) == normalize(d)
        """
        if not docstring or not docstring.strip():
            return ""

        lines = docstring.split("\n")

        # Ensure consistent line endings and whitespace
        formatted_lines = [line.rstrip() for line in lines]

        # Remove leading/trailing blank lines
        while formatted_lines and not formatted_lines[0]:
            formatted_lines.pop(0)
        while formatted_lines and not formatted_lines[-1]:
            formatted_lines.pop()

        return "\n".join(formatted_lines)

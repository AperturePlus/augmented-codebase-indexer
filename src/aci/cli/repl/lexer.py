"""
Command lexer module for ACI REPL syntax highlighting.

Provides a prompt_toolkit Lexer that tokenizes REPL commands for
real-time syntax highlighting of command names, arguments, and options.
"""

from collections.abc import Callable

from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.lexers import Lexer


class CommandLexer(Lexer):
    """
    Lexer for REPL command syntax highlighting.

    Tokenizes input into:
    - command: Valid command names (green)
    - argument: Command arguments (cyan)
    - option.name: Option names like --limit (magenta)
    - option.value: Option values (white)
    - unknown: Unrecognized commands (yellow)
    """

    def __init__(self, commands: list[str]) -> None:
        """
        Initialize the lexer with valid command names.

        Args:
            commands: List of valid command names to recognize.
        """
        self._commands = {cmd.lower() for cmd in commands}

    def lex_document(self, document: Document) -> Callable[[int], StyleAndTextTuples]:
        """
        Return a callable that returns tokens for a given line.

        Args:
            document: The document to lex.

        Returns:
            A callable that takes a line number and returns styled tokens.
        """
        lines = document.lines

        def get_line_tokens(line_number: int) -> StyleAndTextTuples:
            """Get tokens for a specific line."""
            if line_number >= len(lines):
                return []

            line = lines[line_number]
            return self._tokenize_line(line)

        return get_line_tokens

    def _tokenize_line(self, line: str) -> StyleAndTextTuples:
        """
        Tokenize a single line into styled tokens.

        Args:
            line: The line text to tokenize.

        Returns:
            List of (style, text) tuples.
        """
        if not line or not line.strip():
            return [("", line)]

        tokens: StyleAndTextTuples = []
        parts = self._split_preserving_whitespace(line)

        is_first_word = True
        for part in parts:
            if not part:
                continue

            # Preserve whitespace
            if part.isspace():
                tokens.append(("", part))
                continue

            if is_first_word:
                # First non-whitespace word is the command
                style = self._get_command_style(part)
                tokens.append((style, part))
                is_first_word = False
            elif part.startswith("--"):
                # Option: --name or --name=value
                tokens.extend(self._tokenize_option(part))
            elif part.startswith("-") and len(part) > 1 and not part[1].isdigit():
                # Short option: -n or -n=value
                tokens.extend(self._tokenize_option(part))
            else:
                # Regular argument
                tokens.append(("class:argument", part))

        return tokens

    def _split_preserving_whitespace(self, text: str) -> list[str]:
        """
        Split text into words while preserving whitespace as separate tokens.

        Args:
            text: The text to split.

        Returns:
            List of words and whitespace segments.
        """
        result: list[str] = []
        current = ""
        in_whitespace = False

        for char in text:
            is_space = char.isspace()
            if is_space != in_whitespace:
                if current:
                    result.append(current)
                current = char
                in_whitespace = is_space
            else:
                current += char

        if current:
            result.append(current)

        return result

    def _get_command_style(self, word: str) -> str:
        """
        Get the style class for a command word.

        Args:
            word: The command word to check.

        Returns:
            Style class string.
        """
        if word.lower() in self._commands:
            return "class:command"
        return "class:unknown"

    def _tokenize_option(self, option: str) -> StyleAndTextTuples:
        """
        Tokenize an option string (--name or --name=value).

        Args:
            option: The option string to tokenize.

        Returns:
            List of (style, text) tuples.
        """
        if "=" in option:
            name, value = option.split("=", 1)
            return [
                ("class:option.name", name + "="),
                ("class:option.value", value),
            ]
        return [("class:option.name", option)]

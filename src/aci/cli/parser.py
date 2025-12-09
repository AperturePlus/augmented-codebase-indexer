"""
Command parser module for ACI interactive REPL.

Provides parsing functionality to convert user input strings into
structured command objects with proper handling of quoted strings,
escape characters, and keyword arguments.
"""

import shlex
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedCommand:
    """
    Represents a parsed command from user input.

    Attributes:
        name: The command name (first token, lowercased).
        args: Positional arguments following the command.
        kwargs: Keyword arguments in the form key=value.
    """

    name: str
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


class CommandParseError(Exception):
    """Raised when command parsing fails."""

    pass


def parse_command(input_str: str) -> ParsedCommand:
    """
    Parse user input into a structured command.

    Uses shlex for proper tokenization, handling:
    - Quoted strings (single and double quotes)
    - Escape characters
    - Keyword arguments (key=value format)

    Args:
        input_str: Raw user input string.

    Returns:
        ParsedCommand with extracted name, args, and kwargs.

    Raises:
        CommandParseError: If the input cannot be parsed (e.g., unclosed quotes).

    Examples:
        >>> parse_command("search hello world")
        ParsedCommand(name='search', args=['hello', 'world'], kwargs={})

        >>> parse_command('index "/path/with spaces"')
        ParsedCommand(name='index', args=['/path/with spaces'], kwargs={})

        >>> parse_command("search query --limit=10")
        ParsedCommand(name='search', args=['query'], kwargs={'limit': '10'})
    """
    stripped = input_str.strip()

    if not stripped:
        return ParsedCommand(name="", args=[], kwargs={})

    try:
        # Use shlex for proper tokenization with quote handling
        # posix=False preserves backslashes (important for Windows paths)
        tokens = shlex.split(stripped, posix=False)
        # Remove surrounding quotes that shlex preserves in non-POSIX mode
        tokens = [t.strip('"').strip("'") for t in tokens]
    except ValueError as e:
        raise CommandParseError(f"Failed to parse command: {e}") from e

    if not tokens:
        return ParsedCommand(name="", args=[], kwargs={})

    # First token is the command name (case-insensitive)
    name = tokens[0].lower()
    args: list[str] = []
    kwargs: dict[str, Any] = {}

    # Process remaining tokens
    for token in tokens[1:]:
        # Check for keyword argument patterns
        if token.startswith("--") and "=" in token:
            # --key=value format
            key_value = token[2:]  # Remove --
            key, _, value = key_value.partition("=")
            if key:
                kwargs[key] = value
        elif token.startswith("-") and len(token) > 1 and "=" in token:
            # -k=value format (short form with value)
            key_value = token[1:]  # Remove -
            key, _, value = key_value.partition("=")
            if key:
                kwargs[key] = value
        elif "=" in token and not token.startswith("-"):
            # key=value format (without dashes)
            key, _, value = token.partition("=")
            if key:
                kwargs[key] = value
        else:
            # Regular positional argument
            args.append(token)

    return ParsedCommand(name=name, args=args, kwargs=kwargs)


def format_command(command: ParsedCommand) -> str:
    """
    Format a ParsedCommand back into a command string.

    This is primarily used for round-trip testing to verify
    that parsing preserves command semantics.

    Args:
        command: The parsed command to format.

    Returns:
        A string representation of the command.

    Examples:
        >>> cmd = ParsedCommand(name='search', args=['hello'], kwargs={'limit': '10'})
        >>> format_command(cmd)
        'search hello --limit=10'
    """
    if not command.name:
        return ""

    parts = [command.name]

    # Add positional arguments, quoting if necessary
    for arg in command.args:
        if " " in arg or '"' in arg or "'" in arg:
            # Quote arguments containing spaces or quotes
            escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'"{escaped}"')
        else:
            parts.append(arg)

    # Add keyword arguments
    for key, value in sorted(command.kwargs.items()):
        value_str = str(value)
        if " " in value_str or '"' in value_str or "'" in value_str:
            escaped = value_str.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'--{key}="{escaped}"')
        else:
            parts.append(f"--{key}={value_str}")

    return " ".join(parts)

"""
Property-based tests for command parser.

**Feature: interactive-repl, Property 1: Command parsing correctly identifies command and arguments**
**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.cli.parser import CommandParseError, ParsedCommand, format_command, parse_command

# Strategy for valid command names (lowercase alphanumeric, no spaces)
command_name = st.from_regex(r"[a-z][a-z0-9_]*", fullmatch=True).filter(
    lambda s: 1 <= len(s) <= 20
)

# Strategy for simple arguments (no special characters that need escaping)
simple_arg = st.from_regex(r"[a-zA-Z0-9_\-\.\/]+", fullmatch=True).filter(
    lambda s: 1 <= len(s) <= 50 and s not in ("", "-", "--")
)

# Strategy for arguments that contain spaces (will be quoted)
# Generates "word word" patterns to guarantee spaces
arg_with_spaces = st.tuples(
    st.from_regex(r"[a-zA-Z0-9_\-\.]+", fullmatch=True).filter(lambda s: 1 <= len(s) <= 20),
    st.from_regex(r"[a-zA-Z0-9_\-\.]+", fullmatch=True).filter(lambda s: 1 <= len(s) <= 20),
).map(lambda t: f"{t[0]} {t[1]}")

# Strategy for keyword argument keys
kwarg_key = st.from_regex(r"[a-z][a-z0-9_]*", fullmatch=True).filter(
    lambda s: 1 <= len(s) <= 20
)

# Strategy for keyword argument values (simple, no spaces)
kwarg_value = st.from_regex(r"[a-zA-Z0-9_\-\.]+", fullmatch=True).filter(
    lambda s: 1 <= len(s) <= 30
)


@st.composite
def parsed_command_strategy(draw):
    """Generate valid ParsedCommand instances."""
    name = draw(command_name)
    args = draw(st.lists(simple_arg, min_size=0, max_size=5))
    kwargs = draw(st.dictionaries(kwarg_key, kwarg_value, min_size=0, max_size=3))
    return ParsedCommand(name=name, args=args, kwargs=kwargs)


@given(cmd=parsed_command_strategy())
@settings(max_examples=100)
def test_command_parsing_round_trip(cmd: ParsedCommand):
    """
    **Feature: interactive-repl, Property 1: Command parsing correctly identifies command and arguments**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

    For any valid ParsedCommand, formatting then parsing should produce
    an equivalent command with the same name, args, and kwargs.
    """
    # Format the command to a string
    formatted = format_command(cmd)

    # Parse it back
    parsed = parse_command(formatted)

    # Verify the command name matches (case-insensitive)
    assert parsed.name == cmd.name.lower()

    # Verify positional arguments match
    assert parsed.args == cmd.args

    # Verify keyword arguments match
    assert parsed.kwargs == cmd.kwargs


@given(name=command_name, args=st.lists(simple_arg, min_size=0, max_size=5))
@settings(max_examples=100)
def test_command_name_extraction(name: str, args: list[str]):
    """
    **Feature: interactive-repl, Property 1: Command parsing correctly identifies command and arguments**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

    For any command string, the parser should correctly extract the command name
    as the first token (lowercased) and remaining tokens as arguments.
    """
    # Build command string
    parts = [name] + args
    input_str = " ".join(parts)

    # Parse
    parsed = parse_command(input_str)

    # Command name should be lowercased first token
    assert parsed.name == name.lower()

    # Arguments should match remaining tokens
    assert parsed.args == args


@given(name=command_name, key=kwarg_key, value=kwarg_value)
@settings(max_examples=100)
def test_kwarg_extraction_double_dash(name: str, key: str, value: str):
    """
    **Feature: interactive-repl, Property 1: Command parsing correctly identifies command and arguments**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

    For any command with --key=value format, the parser should correctly
    extract the keyword argument.
    """
    input_str = f"{name} --{key}={value}"
    parsed = parse_command(input_str)

    assert parsed.name == name.lower()
    assert key in parsed.kwargs
    assert parsed.kwargs[key] == value


@given(name=command_name, arg=arg_with_spaces)
@settings(max_examples=100)
def test_quoted_argument_handling(name: str, arg: str):
    """
    **Feature: interactive-repl, Property 1: Command parsing correctly identifies command and arguments**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

    For any argument containing spaces, when properly quoted, the parser
    should preserve the argument as a single token.
    """
    # Quote the argument
    input_str = f'{name} "{arg}"'
    parsed = parse_command(input_str)

    assert parsed.name == name.lower()
    assert len(parsed.args) == 1
    assert parsed.args[0] == arg


def test_empty_input():
    """Test that empty input returns empty command."""
    parsed = parse_command("")
    assert parsed.name == ""
    assert parsed.args == []
    assert parsed.kwargs == {}


def test_whitespace_only_input():
    """Test that whitespace-only input returns empty command."""
    parsed = parse_command("   \t  ")
    assert parsed.name == ""
    assert parsed.args == []
    assert parsed.kwargs == {}


def test_known_commands_parsing():
    """
    **Feature: interactive-repl, Property 1: Command parsing correctly identifies command and arguments**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

    Test parsing of known REPL commands matches expected structure.
    """
    # index command (Req 2.1)
    parsed = parse_command("index /path/to/code")
    assert parsed.name == "index"
    assert parsed.args == ["/path/to/code"]

    # search command (Req 2.2)
    parsed = parse_command("search hello world")
    assert parsed.name == "search"
    assert parsed.args == ["hello", "world"]

    # status command (Req 2.3)
    parsed = parse_command("status")
    assert parsed.name == "status"
    assert parsed.args == []

    # update command (Req 2.4)
    parsed = parse_command("update /path/to/code")
    assert parsed.name == "update"
    assert parsed.args == ["/path/to/code"]

    # list command (Req 2.5)
    parsed = parse_command("list")
    assert parsed.name == "list"
    assert parsed.args == []


def test_case_insensitive_command_names():
    """Test that command names are case-insensitive."""
    assert parse_command("SEARCH query").name == "search"
    assert parse_command("Search query").name == "search"
    assert parse_command("INDEX /path").name == "index"


def test_unclosed_quote_raises_error():
    """Test that unclosed quotes raise CommandParseError."""
    try:
        parse_command('search "unclosed')
        raise AssertionError("Should have raised CommandParseError")
    except CommandParseError:
        pass  # Expected

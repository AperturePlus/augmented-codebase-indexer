"""
Property-based tests for REPL enhancements.

Tests for EventLoopManager, CommandLexer, and PromptBuilder components.
"""

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.cli.repl.event_loop import EventLoopManager
from aci.cli.repl.lexer import CommandLexer


# Strategy for number of async operations to run
num_operations = st.integers(min_value=1, max_value=10)


async def simple_async_operation(value: int) -> int:
    """A simple async operation that returns the input value."""
    await asyncio.sleep(0)  # Yield control to event loop
    return value


@given(num_ops=num_operations)
@settings(max_examples=100)
def test_event_loop_persistence_across_async_commands(num_ops: int):
    """
    **Feature: repl-enhancements, Property 1: Event loop persistence across async commands**
    **Validates: Requirements 1.1, 1.2**

    For any sequence of async commands executed in a REPL session without
    codebase switching, the EventLoopManager SHALL use the same event loop
    instance for all executions.
    """
    manager = EventLoopManager()
    initial_loop = manager.loop
    
    try:
        # Execute multiple async operations
        for i in range(num_ops):
            result = manager.run_async(simple_async_operation(i))
            assert result == i
            # Verify same loop is used
            assert manager.loop is initial_loop
            assert not manager.loop.is_closed()
    finally:
        manager.close()


@given(num_resets=st.integers(min_value=1, max_value=5))
@settings(max_examples=100)
def test_event_loop_recreation_on_codebase_switch(num_resets: int):
    """
    **Feature: repl-enhancements, Property 2: Event loop recreation on codebase switch**
    **Validates: Requirements 1.3**

    For any codebase switch operation, the EventLoopManager SHALL close
    the previous event loop and create a new functional event loop.
    """
    manager = EventLoopManager()
    
    try:
        for _ in range(num_resets):
            # Get the current loop
            old_loop = manager.loop
            
            # Simulate codebase switch by resetting
            manager.reset()
            
            # Verify new loop is created
            new_loop = manager.loop
            assert new_loop is not old_loop
            assert not new_loop.is_closed()
            
            # Verify old loop is closed
            assert old_loop.is_closed()
            
            # Verify new loop is functional
            result = manager.run_async(simple_async_operation(42))
            assert result == 42
    finally:
        manager.close()


# ============================================================================
# CommandLexer Property Tests
# ============================================================================

# Strategy for valid command names
valid_command_names = st.lists(
    st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz"),
        min_size=1,
        max_size=10,
    ),
    min_size=1,
    max_size=10,
    unique=True,
)

# Strategy for argument text (non-option, non-empty)
argument_text = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_./"),
    min_size=1,
    max_size=20,
).filter(lambda s: not s.startswith("-") and s.strip() == s)

# Strategy for option names
option_name = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz"),
    min_size=1,
    max_size=10,
)

# Strategy for option values
option_value = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789"),
    min_size=1,
    max_size=10,
)


def extract_token_styles(tokens: list[tuple[str, str]]) -> list[str]:
    """Extract just the style classes from tokens, ignoring whitespace."""
    return [style for style, text in tokens if text.strip()]


@given(
    commands=valid_command_names,
    cmd_index=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=100)
def test_lexer_known_command_tokenization(commands: list[str], cmd_index: int):
    """
    **Feature: repl-enhancements, Property 3: Lexer tokenization correctness**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

    For any valid REPL input string with a known command, the CommandLexer
    SHALL produce tokens where known command names receive the "command" token type.
    """
    lexer = CommandLexer(commands)
    # Pick a command from the list
    cmd = commands[cmd_index % len(commands)]

    # Create a mock document with just the command
    from prompt_toolkit.document import Document
    doc = Document(cmd)
    
    get_tokens = lexer.lex_document(doc)
    tokens = get_tokens(0)
    
    styles = extract_token_styles(tokens)
    assert len(styles) >= 1
    assert styles[0] == "class:command"


@given(
    commands=valid_command_names,
    unknown_cmd=st.text(
        alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        min_size=1,
        max_size=10,
    ),
)
@settings(max_examples=100)
def test_lexer_unknown_command_tokenization(commands: list[str], unknown_cmd: str):
    """
    **Feature: repl-enhancements, Property 3: Lexer tokenization correctness**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

    For any valid REPL input string with an unknown command, the CommandLexer
    SHALL produce tokens where unknown command names receive the "unknown" token type.
    """
    # Ensure unknown_cmd is not in commands (case-insensitive)
    if unknown_cmd.lower() in [c.lower() for c in commands]:
        return  # Skip this case
    
    lexer = CommandLexer(commands)
    
    from prompt_toolkit.document import Document
    doc = Document(unknown_cmd)
    
    get_tokens = lexer.lex_document(doc)
    tokens = get_tokens(0)
    
    styles = extract_token_styles(tokens)
    assert len(styles) >= 1
    assert styles[0] == "class:unknown"


@given(
    commands=valid_command_names,
    cmd_index=st.integers(min_value=0, max_value=100),
    args=st.lists(argument_text, min_size=1, max_size=5),
)
@settings(max_examples=100)
def test_lexer_argument_tokenization(
    commands: list[str], cmd_index: int, args: list[str]
):
    """
    **Feature: repl-enhancements, Property 3: Lexer tokenization correctness**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

    For any valid REPL input string, the CommandLexer SHALL produce tokens
    where arguments receive the "argument" token type.
    """
    lexer = CommandLexer(commands)
    cmd = commands[cmd_index % len(commands)]
    
    # Build input: command followed by arguments
    input_text = cmd + " " + " ".join(args)
    
    from prompt_toolkit.document import Document
    doc = Document(input_text)
    
    get_tokens = lexer.lex_document(doc)
    tokens = get_tokens(0)
    
    styles = extract_token_styles(tokens)
    # First should be command, rest should be arguments
    assert styles[0] == "class:command"
    for style in styles[1:]:
        assert style == "class:argument"


@given(
    commands=valid_command_names,
    cmd_index=st.integers(min_value=0, max_value=100),
    opt_name=option_name,
    opt_value=option_value,
)
@settings(max_examples=100)
def test_lexer_option_tokenization(
    commands: list[str], cmd_index: int, opt_name: str, opt_value: str
):
    """
    **Feature: repl-enhancements, Property 3: Lexer tokenization correctness**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

    For any valid REPL input string with options (--name=value), the CommandLexer
    SHALL produce tokens where option names receive "option.name" and values
    receive "option.value" token types.
    """
    lexer = CommandLexer(commands)
    cmd = commands[cmd_index % len(commands)]
    
    # Build input: command with option
    input_text = f"{cmd} --{opt_name}={opt_value}"
    
    from prompt_toolkit.document import Document
    doc = Document(input_text)
    
    get_tokens = lexer.lex_document(doc)
    tokens = get_tokens(0)
    
    # Find option tokens (skip whitespace)
    non_ws_tokens = [(s, t) for s, t in tokens if t.strip()]
    
    assert len(non_ws_tokens) >= 2
    assert non_ws_tokens[0][0] == "class:command"
    
    # Find the option tokens
    option_tokens = [(s, t) for s, t in non_ws_tokens[1:]]
    assert len(option_tokens) >= 1
    
    # Check that we have option.name and option.value
    styles = [s for s, _ in option_tokens]
    assert "class:option.name" in styles
    assert "class:option.value" in styles


# ============================================================================
# PromptBuilder Property Tests
# ============================================================================

from pathlib import Path

from aci.cli.repl.context import REPLContext
from aci.cli.repl.prompt import PromptBuilder


# Strategy for valid directory names (last component of a path)
directory_name = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-"),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip() == s and s not in (".", ".."))

# Strategy for path components (for building full paths)
path_component = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-"),
    min_size=1,
    max_size=15,
).filter(lambda s: s.strip() == s and s not in (".", ".."))


@given(
    parent_parts=st.lists(path_component, min_size=0, max_size=5),
    dir_name=directory_name,
)
@settings(max_examples=100)
def test_prompt_generation_with_codebase_name(
    parent_parts: list[str], dir_name: str
):
    """
    **Feature: repl-enhancements, Property 4: Prompt generation with codebase name**
    **Validates: Requirements 3.2, 3.3, 3.5**

    For any codebase path set in REPLContext, the PromptBuilder SHALL generate
    a prompt containing only the last path component (directory name) of that path.
    """
    # Build a path with the given components
    if parent_parts:
        full_path = Path("/") / Path(*parent_parts) / dir_name
    else:
        full_path = Path("/") / dir_name
    
    # Create context and set codebase
    context = REPLContext()
    context.set_codebase(full_path)
    
    # Create prompt builder
    builder = PromptBuilder(context)
    
    # Verify get_codebase_display returns only the directory name
    display = builder.get_codebase_display()
    assert display == dir_name
    
    # Verify prompt contains the codebase name
    prompt_tokens = builder.get_prompt()
    
    # Find the codebase token
    codebase_tokens = [
        (style, text) for style, text in prompt_tokens
        if style == "class:prompt.codebase"
    ]
    
    assert len(codebase_tokens) == 1
    assert codebase_tokens[0][1] == dir_name
    
    # Verify prompt structure: codebase, separator, "aci", "> "
    styles = [style for style, _ in prompt_tokens]
    assert "class:prompt.codebase" in styles
    assert "class:prompt" in styles
    assert "class:prompt.arrow" in styles


@given(st.just(None))
@settings(max_examples=10)
def test_prompt_generation_without_codebase(_):
    """
    **Feature: repl-enhancements, Property 4: Prompt generation with codebase name**
    **Validates: Requirements 3.1**

    When no codebase is explicitly set, the PromptBuilder SHALL generate
    a prompt without a codebase indicator (just "aci> ").
    """
    # Create context without setting codebase
    context = REPLContext()
    
    # Create prompt builder
    builder = PromptBuilder(context)
    
    # Verify get_codebase_display returns empty string
    display = builder.get_codebase_display()
    assert display == ""
    
    # Verify prompt does not contain codebase token
    prompt_tokens = builder.get_prompt()
    
    styles = [style for style, _ in prompt_tokens]
    assert "class:prompt.codebase" not in styles
    
    # Verify basic prompt structure: "aci", "> "
    assert "class:prompt" in styles
    assert "class:prompt.arrow" in styles


@given(
    dir_name=directory_name,
)
@settings(max_examples=100)
def test_prompt_updates_on_codebase_change(dir_name: str):
    """
    **Feature: repl-enhancements, Property 4: Prompt generation with codebase name**
    **Validates: Requirements 3.4, 3.5**

    When the codebase changes, the PromptBuilder SHALL update immediately
    for the next input.
    """
    context = REPLContext()
    builder = PromptBuilder(context)
    
    # Initially no codebase
    prompt1 = builder.get_prompt()
    styles1 = [style for style, _ in prompt1]
    assert "class:prompt.codebase" not in styles1
    
    # Set codebase
    context.set_codebase(Path("/some/path") / dir_name)
    
    # Prompt should now include codebase
    prompt2 = builder.get_prompt()
    codebase_tokens = [
        text for style, text in prompt2
        if style == "class:prompt.codebase"
    ]
    assert len(codebase_tokens) == 1
    assert codebase_tokens[0] == dir_name
    
    # Clear codebase
    context.clear_codebase()
    
    # Prompt should revert to default
    prompt3 = builder.get_prompt()
    styles3 = [style for style, _ in prompt3]
    assert "class:prompt.codebase" not in styles3

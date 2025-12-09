"""
Property-based tests for command router.

Tests Properties 2, 3, and 6 from the interactive-repl design.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.cli.parser import ParsedCommand
from aci.cli.router import ArgumentInfo, CommandInfo, CommandResult, CommandRouter


# Known command names that will be registered
KNOWN_COMMANDS = ["help", "exit", "quit", "q", "index", "search", "status", "update", "list", "reset", "?"]

# Strategy for unknown command names (not in known commands)
unknown_command_name = st.from_regex(r"[a-z][a-z0-9_]*", fullmatch=True).filter(
    lambda s: s not in KNOWN_COMMANDS and 1 <= len(s) <= 20
)

# Strategy for any string that could be a command name
any_command_name = st.from_regex(r"[a-z][a-z0-9_]*", fullmatch=True).filter(
    lambda s: 1 <= len(s) <= 20
)


def create_test_router() -> CommandRouter:
    """Create a router with test commands registered."""
    router = CommandRouter()

    # Register test commands
    router.register(
        name="help",
        handler=lambda cmd: CommandResult(success=True, message="help", data=router.get_available_commands()),
        description="Display help",
        usage="help",
        aliases=["?"],
    )

    router.register(
        name="exit",
        handler=lambda cmd: CommandResult(success=True, should_exit=True),
        description="Exit the shell",
        usage="exit",
        aliases=["quit", "q"],
    )

    router.register(
        name="index",
        handler=lambda cmd: CommandResult(
            success=False,
            message="Usage: index <path>\n\nIndex a directory.",
        ) if not cmd.args else CommandResult(success=True, message=f"indexing {cmd.args[0]}"),
        description="Index a directory",
        usage="index <path>",
        arguments=[ArgumentInfo(name="path", description="Directory to index", required=True)],
    )

    router.register(
        name="search",
        handler=lambda cmd: CommandResult(
            success=False,
            message="Usage: search <query>\n\nSearch the codebase.",
        ) if not cmd.args else CommandResult(success=True, message=f"searching {cmd.args[0]}"),
        description="Search the codebase",
        usage="search <query>",
        arguments=[ArgumentInfo(name="query", description="Search query", required=True)],
    )

    router.register(
        name="status",
        handler=lambda cmd: CommandResult(success=True, message="status ok"),
        description="Show status",
        usage="status",
    )

    router.register(
        name="update",
        handler=lambda cmd: CommandResult(
            success=False,
            message="Usage: update <path>\n\nUpdate the index.",
        ) if not cmd.args else CommandResult(success=True, message=f"updating {cmd.args[0]}"),
        description="Update the index",
        usage="update <path>",
        arguments=[ArgumentInfo(name="path", description="Directory to update", required=True)],
    )

    router.register(
        name="list",
        handler=lambda cmd: CommandResult(success=True, message="listing repos"),
        description="List repositories",
        usage="list",
    )

    router.register(
        name="reset",
        handler=lambda cmd: CommandResult(success=True, message="reset"),
        description="Reset the index",
        usage="reset",
    )

    return router


@given(name=unknown_command_name)
@settings(max_examples=100)
def test_unknown_command_returns_error_without_crash(name: str):
    """
    **Feature: interactive-repl, Property 2: Unknown command handling returns error without crash**
    **Validates: Requirements 3.2, 3.3**

    For any string that is not a recognized command name, the router
    should return an error result and the session should remain active
    (should_exit=False).
    """
    router = create_test_router()
    command = ParsedCommand(name=name, args=[], kwargs={})

    result = router.route(command)

    # Should return error (not crash)
    assert isinstance(result, CommandResult)
    assert result.success is False
    assert result.should_exit is False
    assert result.message is not None
    assert "help" in result.message.lower()


def test_help_output_contains_all_registered_commands():
    """
    **Feature: interactive-repl, Property 3: Help output contains all registered commands**
    **Validates: Requirements 3.1**

    For any command registered in the router, the help output should
    contain that command's name and description.
    """
    router = create_test_router()

    # Get all registered commands
    commands = router.get_available_commands()

    # Should have all primary commands (not aliases)
    command_names = {cmd.name for cmd in commands}
    expected_names = {"help", "exit", "index", "search", "status", "update", "list", "reset"}

    assert command_names == expected_names

    # Each command should have name and description
    for cmd in commands:
        assert cmd.name
        assert cmd.description
        assert cmd.usage


@given(cmd_name=st.sampled_from(["index", "search", "update"]))
@settings(max_examples=100)
def test_missing_required_arguments_returns_usage(cmd_name: str):
    """
    **Feature: interactive-repl, Property 6: Missing required arguments returns usage information**
    **Validates: Requirements 3.4**

    For any command that requires arguments, invoking it without arguments
    should return a result containing usage information for that command.
    """
    router = create_test_router()
    command = ParsedCommand(name=cmd_name, args=[], kwargs={})

    result = router.route(command)

    # Should return error with usage info
    assert isinstance(result, CommandResult)
    assert result.success is False
    assert result.message is not None
    assert "usage" in result.message.lower()


def test_empty_command_returns_success():
    """Test that empty command (no-op) returns success."""
    router = create_test_router()
    command = ParsedCommand(name="", args=[], kwargs={})

    result = router.route(command)

    assert result.success is True
    assert result.should_exit is False


def test_exit_commands_signal_termination():
    """
    **Feature: interactive-repl, Property 5: Exit commands signal session termination**
    **Validates: Requirements 1.4**

    For any exit command variant (exit, quit, q), the router should
    return a result with should_exit=True.
    """
    router = create_test_router()

    for exit_cmd in ["exit", "quit", "q"]:
        command = ParsedCommand(name=exit_cmd, args=[], kwargs={})
        result = router.route(command)

        assert result.success is True
        assert result.should_exit is True


def test_command_with_arguments_succeeds():
    """Test that commands with required arguments succeed when provided."""
    router = create_test_router()

    # Index with path
    result = router.route(ParsedCommand(name="index", args=["/path/to/code"], kwargs={}))
    assert result.success is True

    # Search with query
    result = router.route(ParsedCommand(name="search", args=["hello"], kwargs={}))
    assert result.success is True


def test_handler_exception_returns_error():
    """Test that handler exceptions are caught and returned as errors."""
    router = CommandRouter()

    def failing_handler(cmd: ParsedCommand) -> CommandResult:
        raise ValueError("Test error")

    router.register(
        name="fail",
        handler=failing_handler,
        description="A failing command",
        usage="fail",
    )

    result = router.route(ParsedCommand(name="fail", args=[], kwargs={}))

    assert result.success is False
    assert "error" in result.message.lower()
    assert "Test error" in result.message

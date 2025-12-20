"""
Integration tests for REPL workflow.

Tests the complete workflow: start session, execute commands, exit.
**Validates: Requirements 1.1, 1.4**

Note: Tests avoid direct REPLController instantiation due to prompt_toolkit
requiring a real console. Instead, we test the core logic through the
router, parser, and handlers which REPLController delegates to.
"""

from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from aci.cli.handlers import register_all_commands
from aci.cli.parser import CommandParseError, ParsedCommand, parse_command
from aci.cli.router import CommandRouter
from aci.cli.services import ServicesContainer
from aci.cli.ui import render_error, render_help, render_welcome_banner


@pytest.fixture
def mock_services() -> ServicesContainer:
    """Create a mock services container for testing."""
    mock_config = MagicMock()
    mock_config.embedding.api_url = "http://test"
    mock_config.embedding.model = "test-model"
    mock_config.search.default_limit = 10
    mock_config.indexing.max_workers = 2
    mock_config.embedding.batch_size = 32

    return ServicesContainer(
        config=mock_config,
        embedding_client=MagicMock(),
        vector_store=MagicMock(),
        metadata_store=MagicMock(),
        file_scanner=MagicMock(),
        chunker=MagicMock(),
        reranker=None,
    )


@pytest.fixture
def test_console() -> Console:
    """Create a console that captures output."""
    return Console(file=StringIO(), force_terminal=True, width=80)


@pytest.fixture
def router_with_handlers(mock_services) -> CommandRouter:
    """Create a router with all handlers registered."""
    router = CommandRouter()
    register_all_commands(router, mock_services)
    return router


class TestREPLWorkflowLogic:
    """
    Integration tests for REPL workflow logic.

    Tests the core command processing logic that REPLController uses,
    simulating the handle_input flow without prompt_toolkit dependencies.
    """

    def test_empty_input_handling(self, router_with_handlers):
        """
        **Validates: Requirements 1.1**

        Empty input should be handled gracefully (no-op).
        """
        # Empty command name results in no-op
        cmd = parse_command("")
        result = router_with_handlers.route(cmd)
        assert result.success is True
        assert result.should_exit is False

        # Whitespace-only also results in empty command
        cmd = parse_command("   ")
        result = router_with_handlers.route(cmd)
        assert result.success is True
        assert result.should_exit is False

    def test_exit_command_signals_termination(self, router_with_handlers):
        """
        **Validates: Requirements 1.4**

        Exit commands should signal session termination.
        """
        for exit_cmd in ["exit", "quit", "q"]:
            cmd = parse_command(exit_cmd)
            result = router_with_handlers.route(cmd)
            assert result.success is True
            assert result.should_exit is True, f"'{exit_cmd}' should signal exit"

    def test_help_command_continues_session(self, router_with_handlers):
        """
        **Validates: Requirements 1.1, 3.1**

        Help command should return success and not exit.
        """
        for help_cmd in ["help", "?"]:
            cmd = parse_command(help_cmd)
            result = router_with_handlers.route(cmd)
            assert result.success is True
            assert result.should_exit is False
            assert result.data is not None  # Should have command list

    def test_unknown_command_continues_session(self, router_with_handlers):
        """
        **Validates: Requirements 3.2, 3.3**

        Unknown commands should show error but not exit.
        """
        cmd = parse_command("unknowncommand")
        result = router_with_handlers.route(cmd)
        assert result.success is False
        assert result.should_exit is False
        assert "help" in result.message.lower()

    def test_status_command_continues_session(self, router_with_handlers):
        """
        **Validates: Requirements 1.1, 2.3**

        Status command should execute and not exit.
        """
        cmd = parse_command("status")
        result = router_with_handlers.route(cmd)
        assert result.success is True
        assert result.should_exit is False

    def test_list_command_continues_session(self, router_with_handlers):
        """
        **Validates: Requirements 1.1, 2.5**

        List command should execute and not exit.
        """
        cmd = parse_command("list")
        result = router_with_handlers.route(cmd)
        assert result.success is True
        assert result.should_exit is False

    def test_command_sequence_workflow(self, router_with_handlers):
        """
        **Validates: Requirements 1.1, 1.4**

        Test a complete workflow: multiple commands then exit.
        """
        # Simulate a session with multiple commands
        commands = ["help", "status", "list", "?"]

        for cmd_str in commands:
            cmd = parse_command(cmd_str)
            result = router_with_handlers.route(cmd)
            assert result.success is True, f"Command '{cmd_str}' should succeed"
            assert result.should_exit is False, f"Command '{cmd_str}' should not exit"

        # Exit should terminate
        cmd = parse_command("exit")
        result = router_with_handlers.route(cmd)
        assert result.success is True
        assert result.should_exit is True


class TestCommandRouterIntegration:
    """Integration tests for command router with handlers."""

    def test_all_commands_registered(self, router_with_handlers):
        """
        **Validates: Requirements 1.1**

        All expected commands should be registered.
        """
        expected_commands = {"help", "exit", "index", "search", "status", "update", "list", "reset"}
        registered = {cmd.name for cmd in router_with_handlers.get_available_commands()}

        assert expected_commands == registered

    def test_aliases_work(self, router_with_handlers):
        """
        **Validates: Requirements 1.4, 3.1**

        Command aliases should route to correct handlers.
        """
        # ? should work like help
        result = router_with_handlers.route(ParsedCommand(name="?", args=[], kwargs={}))
        assert result.success is True

        # quit and q should work like exit
        for alias in ["quit", "q"]:
            result = router_with_handlers.route(ParsedCommand(name=alias, args=[], kwargs={}))
            assert result.success is True
            assert result.should_exit is True


class TestCommandParserIntegration:
    """Integration tests for command parsing in REPL context."""

    def test_parse_and_route_index(self, router_with_handlers):
        """
        **Validates: Requirements 2.1**

        Index command should parse and route correctly.
        """
        cmd = parse_command("index /path/to/code")
        result = router_with_handlers.route(cmd)

        assert result.success is True
        assert result.data["path"] == "/path/to/code"

    def test_parse_and_route_search(self, router_with_handlers):
        """
        **Validates: Requirements 2.2**

        Search command should parse and route correctly.
        """
        cmd = parse_command("search authentication handler")
        result = router_with_handlers.route(cmd)

        assert result.success is True
        assert "authentication handler" in result.data["query"]

    def test_parse_and_route_search_with_limit(self, router_with_handlers):
        """
        **Validates: Requirements 2.2**

        Search command with limit option should parse correctly.
        """
        cmd = parse_command("search query --limit=5")
        result = router_with_handlers.route(cmd)

        assert result.success is True
        assert result.data["limit"] == "5"

    def test_missing_required_args_returns_usage(self, router_with_handlers):
        """
        **Validates: Requirements 3.4**

        Commands missing required args should return usage info.
        """
        for cmd_name in ["index", "search", "update"]:
            cmd = parse_command(cmd_name)
            result = router_with_handlers.route(cmd)

            assert result.success is False
            assert "usage" in result.message.lower()

    def test_quoted_path_parsing(self, router_with_handlers):
        """
        **Validates: Requirements 2.1**

        Paths with spaces should be handled via quoting.
        """
        cmd = parse_command('index "/path/with spaces/code"')
        result = router_with_handlers.route(cmd)

        assert result.success is True
        assert result.data["path"] == "/path/with spaces/code"

    def test_invalid_syntax_raises_error(self):
        """
        **Validates: Requirements 3.3**

        Invalid command syntax should raise parse error.
        """
        with pytest.raises(CommandParseError):
            parse_command('search "unclosed quote')


class TestUIComponentsIntegration:
    """Integration tests for UI components."""

    def test_welcome_banner_renders(self):
        """
        **Validates: Requirements 6.1**

        Welcome banner should render without errors.
        """
        # Use console without markup/ANSI for easier assertion
        console = Console(file=StringIO(), force_terminal=False, no_color=True, width=80)
        render_welcome_banner(console)
        output = console.file.getvalue()
        # Check for key content in the banner
        assert "Augmented Codebase Indexer" in output
        assert "help" in output
        assert "exit" in output

    def test_help_renders_all_commands(self, test_console, router_with_handlers):
        """
        **Validates: Requirements 6.4**

        Help should render all commands in a table.
        """
        commands = router_with_handlers.get_available_commands()
        render_help(commands, test_console)
        output = test_console.file.getvalue()

        # Check that command names appear in output
        for cmd in commands:
            assert cmd.name in output

    def test_error_renders_message(self, test_console):
        """
        **Validates: Requirements 6.5**

        Error messages should render distinctly.
        """
        render_error("Test error message", test_console)
        output = test_console.file.getvalue()
        assert "Test error message" in output




class TestREPLModuleStructure:
    """
    Integration tests verifying REPL module refactoring preserved functionality.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    """

    def test_all_modules_importable(self):
        """
        **Validates: Requirements 5.1, 5.2**

        All refactored REPL modules should be importable.
        """
        # Import all modules from the repl package
        from aci.cli.repl import (
            CommandCompleter,
            IndexingOperations,
            REPLContext,
            REPLController,
            SearchOperations,
        )

        # Verify classes are accessible
        assert CommandCompleter is not None
        assert IndexingOperations is not None
        assert REPLContext is not None
        assert REPLController is not None
        assert SearchOperations is not None

    def test_repl_controller_from_package(self):
        """
        **Validates: Requirements 5.3**

        REPLController should be importable from the repl package.
        """
        from aci.cli.repl import REPLController

        # Verify it's the correct class
        assert hasattr(REPLController, 'run')
        assert hasattr(REPLController, 'handle_input')
        assert hasattr(REPLController, '_create_history')

    def test_context_integration_with_controller(self, mock_services):
        """
        **Validates: Requirements 5.4**

        REPLContext should integrate correctly with REPLController.
        """
        from aci.cli.repl import REPLContext

        # Create context and verify it works
        context = REPLContext()
        context.set_metadata_store(mock_services.metadata_store)

        # Verify context methods work
        assert context.get_codebase() is not None
        assert context.has_explicit_codebase() is False

        # Set and verify codebase
        from pathlib import Path
        test_path = Path("/test/path")
        context.set_codebase(test_path)
        assert context.get_codebase() == test_path
        assert context.has_explicit_codebase() is True

        # Clear and verify
        context.clear_codebase()
        assert context.has_explicit_codebase() is False

    def test_indexing_ops_integration(self, mock_services, test_console):
        """
        **Validates: Requirements 5.4**

        IndexingOperations should work correctly after refactoring.
        """
        from aci.cli.repl import IndexingOperations

        # Create indexing ops
        indexing_ops = IndexingOperations(mock_services, test_console, verbose=False)

        # Verify properties work
        assert indexing_ops.verbose is False
        indexing_ops.verbose = True
        assert indexing_ops.verbose is True

    def test_search_ops_integration(self, mock_services, test_console):
        """
        **Validates: Requirements 5.4**

        SearchOperations should work correctly after refactoring.
        """
        from aci.cli.repl import REPLContext, SearchOperations

        # Create context and search ops
        context = REPLContext()
        context.set_metadata_store(mock_services.metadata_store)
        search_ops = SearchOperations(mock_services, test_console, context)

        # Verify it was created successfully
        assert search_ops.context is context
        assert search_ops.console is test_console

    def test_completer_integration(self, router_with_handlers):
        """
        **Validates: Requirements 5.4**

        CommandCompleter should work correctly after refactoring.
        """
        from aci.cli.repl import CommandCompleter

        # Create completer with router
        completer = CommandCompleter(router_with_handlers)

        # Verify command names are populated
        assert len(completer._command_names) > 0
        assert "help" in completer._command_names
        assert "exit" in completer._command_names
        assert "index" in completer._command_names

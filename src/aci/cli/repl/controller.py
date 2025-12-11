"""
REPL controller module for ACI interactive shell.

Provides the main Read-Eval-Print Loop controller that integrates
prompt_toolkit for input handling, command parsing, routing, and UI rendering.
"""

from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console

from aci.cli.handlers import register_all_commands
from aci.cli.parser import CommandParseError, parse_command
from aci.cli.repl.completer import CommandCompleter
from aci.cli.repl.context import REPLContext
from aci.cli.repl.event_loop import EventLoopManager
from aci.cli.repl.indexing_ops import IndexingOperations
from aci.cli.repl.lexer import CommandLexer
from aci.cli.repl.prompt import PromptBuilder
from aci.cli.repl.search_ops import SearchOperations
from aci.cli.router import CommandRouter
from aci.cli.services import ServicesContainer
from aci.cli.ui import render_error, render_help, render_welcome_banner
from aci.core.path_utils import ensure_directory_exists


# Prompt style for prompt_toolkit with command highlighting and codebase colors
PROMPT_STYLE = Style.from_dict({
    "prompt": "#00aa00 bold",
    "prompt-arrow": "#ffffff",
    "prompt.codebase": "#00aaaa",
    "prompt.separator": "#888888",
    "command": "#00aa00 bold",
    "argument": "#00aaaa",
    "option.name": "#aa00aa",
    "option.value": "#ffffff",
    "unknown": "#aaaa00",
})


class REPLController:
    """
    Interactive REPL session controller.

    Manages the main loop, integrating prompt_toolkit for input,
    command parsing, routing, and Rich for output rendering.

    Attributes:
        services: Container with all initialized services.
        console: Rich console for styled output.
        router: Command router for dispatching commands.
        session: prompt_toolkit session for input handling.
    """

    def __init__(
        self,
        services: ServicesContainer,
        console: Optional[Console] = None,
        history_file: Optional[str] = None,
    ):
        """
        Initialize the REPL controller.

        Args:
            services: Services container with initialized services.
            console: Rich Console for output. Creates new one if None.
            history_file: Path to history file. Defaults to '.aci/history'.
        """
        self.services = services
        self.console = console or Console()
        self.router = CommandRouter()
        self._running = False
        self._verbose = False  # Verbose output mode

        # Create REPL context for codebase management
        self.context = REPLContext()
        self.context.set_metadata_store(services.metadata_store)

        # Create event loop manager for persistent async operations
        self._event_loop_manager = EventLoopManager()

        # Create prompt builder for dynamic prompts
        self._prompt_builder = PromptBuilder(self.context)

        # Create indexing operations handler with event loop manager
        self._indexing_ops = IndexingOperations(
            services, self.console, self._verbose, self._event_loop_manager
        )

        # Create search operations handler with event loop manager
        self._search_ops = SearchOperations(
            services, self.console, self.context, self._event_loop_manager
        )

        # Register all command handlers with context
        register_all_commands(self.router, services, context=self.context)

        # Create command lexer with valid command names
        command_names = self._get_command_names()
        self._lexer = CommandLexer(command_names)

        # Set up prompt_toolkit session with history, completion, and lexer
        history_path = history_file or ".aci/history"
        history = self._create_history(history_path)
        self.session: PromptSession = PromptSession(
            history=history,
            completer=CommandCompleter(self.router),
            lexer=self._lexer,
            style=PROMPT_STYLE,
            complete_while_typing=False,
        )

    def _get_command_names(self) -> list[str]:
        """
        Get list of valid command names from the router.
        
        Returns:
            List of registered command names.
        """
        return [cmd.name for cmd in self.router.get_available_commands()]

    def _create_history(self, history_path: str):
        """
        Create history storage, ensuring parent directory exists.
        
        Falls back to in-memory history if directory creation fails.
        
        Args:
            history_path: Path to the history file.
            
        Returns:
            FileHistory if directory exists/created, InMemoryHistory otherwise.
        """
        from pathlib import Path
        
        path = Path(history_path)
        parent_dir = path.parent
        
        # Ensure parent directory exists
        if not ensure_directory_exists(parent_dir):
            self.console.print(
                "[yellow]Warning: Could not create history directory, "
                "using in-memory history[/yellow]"
            )
            return InMemoryHistory()
        
        try:
            return FileHistory(history_path)
        except (OSError, PermissionError) as e:
            self.console.print(
                f"[yellow]Warning: Could not create history file ({e}), "
                "using in-memory history[/yellow]"
            )
            return InMemoryHistory()

    def run(self) -> None:
        """
        Start the REPL main loop.

        Displays welcome banner, then enters the read-eval-print loop
        until an exit command is received or Ctrl+C/Ctrl+D is pressed.
        """
        # Display welcome banner
        render_welcome_banner(self.console)
        self.console.print()

        self._running = True

        try:
            while self._running:
                try:
                    # Get user input with dynamic prompt from PromptBuilder
                    user_input = self.session.prompt(
                        self._prompt_builder.get_prompt(),
                    )

                    # Handle the input
                    should_continue = self.handle_input(user_input)
                    if not should_continue:
                        break

                except KeyboardInterrupt:
                    # Ctrl+C - show message and continue or exit
                    self.console.print("\n[yellow]Use 'exit' or 'quit' to leave.[/yellow]")
                    continue

                except EOFError:
                    # Ctrl+D - graceful exit
                    self.console.print("\n[cyan]Goodbye![/cyan]")
                    break
        finally:
            # Clean up event loop on REPL exit
            self._event_loop_manager.close()


    def handle_input(self, user_input: str) -> bool:
        """
        Process user input and execute the command.

        Args:
            user_input: Raw input string from the user.

        Returns:
            True to continue the REPL, False to exit.
        """
        # Skip empty input
        stripped = user_input.strip()
        if not stripped:
            return True

        try:
            # Parse the command
            command = parse_command(stripped)
        except CommandParseError as e:
            render_error(str(e), self.console)
            return True

        # Route and execute the command
        result = self.router.route(command)

        # Handle the result
        if result.should_exit:
            self.console.print("[cyan]Goodbye![/cyan]")
            return False

        if not result.success:
            render_error(result.message or "Command failed", self.console)
            return True

        # Handle successful commands with special rendering
        self._render_result(command.name, result)

        return True

    def _render_result(self, command_name: str, result) -> None:
        """
        Render command result based on command type.

        Args:
            command_name: Name of the executed command.
            result: CommandResult from the handler.
        """
        if command_name in ("help", "?"):
            # Render help table
            if result.data:
                render_help(result.data, self.console)
        elif command_name == "verbose":
            self._toggle_verbose()
        elif command_name == "status":
            self._render_status(result)
        elif command_name == "list":
            self._render_list(result)
        elif command_name == "use":
            # Reset event loop when codebase changes
            if result.data and result.data.get("codebase_changed"):
                self._reset_async_clients()
                self._event_loop_manager.reset()
            if result.message:
                self.console.print(f"[green]{result.message}[/green]")
        elif command_name in ("index", "search", "update", "reset"):
            # These commands need async execution
            self._execute_async_command(command_name, result)
        elif result.message:
            self.console.print(f"[green]{result.message}[/green]")

    def _toggle_verbose(self) -> None:
        """Toggle verbose output mode."""
        self._verbose = not self._verbose
        self._indexing_ops.verbose = self._verbose
        mode = "verbose" if self._verbose else "compact"
        self.console.print(f"[cyan]Output mode:[/cyan] {mode}")

    def _reset_async_clients(self) -> None:
        """
        Reset async HTTP clients before event loop reset.
        
        This ensures that httpx.AsyncClient instances are properly closed
        before the event loop is reset, preventing "Event loop is closed" errors.
        """
        # Close embedding client's internal HTTP client
        if hasattr(self.services.embedding_client, 'close'):
            try:
                self._event_loop_manager.run_async(
                    self.services.embedding_client.close()
                )
            except Exception:
                # Best effort - client may already be closed
                pass
        
        # Close reranker's internal HTTP client if it has one
        if self.services.reranker and hasattr(self.services.reranker, 'close'):
            try:
                self._event_loop_manager.run_async(
                    self.services.reranker.close()
                )
            except Exception:
                pass

    def _execute_async_command(self, command_name: str, result) -> None:
        """
        Execute commands that require async service calls.

        Args:
            command_name: Name of the command.
            result: CommandResult containing command data.
        """
        data = result.data or {}

        if command_name == "index":
            self._indexing_ops.run_index(data.get("path"))
        elif command_name == "search":
            self._search_ops.run_search(
                data.get("query"),
                data.get("limit"),
                data.get("artifact_types"),
            )
        elif command_name == "update":
            self._indexing_ops.run_update(data.get("path"))
        elif command_name == "reset":
            self._indexing_ops.run_reset()


    def _render_status(self, result) -> None:
        """Render status command output."""
        from rich.panel import Panel
        from rich.table import Table

        try:
            stats = self.services.metadata_store.get_stats()

            # Get vector count from vector store
            vector_count = 0
            try:
                vector_stats = self._event_loop_manager.run_async(
                    self.services.vector_store.get_stats()
                )
                vector_count = vector_stats.get("total_vectors", 0)
            except Exception:
                pass  # Will show 0 if unavailable

            # Per-collection statistics grid (Requirements 3.1, 3.2)
            grid = Table.grid(padding=1)
            grid.add_column(style="bold")
            grid.add_column()

            grid.add_row("File Count:", str(stats["total_files"]))
            grid.add_row("Vector Count:", str(vector_count))
            grid.add_row("Total Chunks:", str(stats["total_chunks"]))
            grid.add_row("Total Lines:", str(stats["total_lines"]))

            self.console.print(
                Panel(grid, title="Index Statistics", border_style="blue", expand=False)
            )

            if stats.get("languages"):
                lang_table = Table(title="Languages", box=None, show_header=True)
                lang_table.add_column("Language", style="cyan")
                lang_table.add_column("Files", justify="right")

                for lang, count in stats["languages"].items():
                    lang_table.add_row(lang, str(count))

                self.console.print(
                    Panel(lang_table, border_style="blue", expand=False)
                )

            # Staleness information (Requirements 3.3, 3.4)
            try:
                stale_files = self.services.metadata_store.get_stale_files(limit=5)
                stale_count = len(self.services.metadata_store.get_stale_files())
                
                if stale_count > 0:
                    self.console.print("\n[bold yellow]Stale Files:[/bold yellow]")
                    self.console.print(f"  Total stale files: [yellow]{stale_count}[/yellow]")
                    
                    if stale_files:
                        self.console.print("  Examples:")
                        for file_path, staleness_seconds in stale_files:
                            staleness_hours = staleness_seconds / 3600
                            if staleness_hours >= 1:
                                staleness_str = f"{staleness_hours:.1f}h"
                            else:
                                staleness_minutes = staleness_seconds / 60
                                staleness_str = f"{staleness_minutes:.1f}m"
                            self.console.print(f"    • {file_path} ({staleness_str} stale)")
                else:
                    self.console.print("\n[green]✓ All files are up to date[/green]")
            except Exception:
                pass  # Staleness info is optional

            # Health checks
            self.console.print("\n[bold]System Health:[/bold]")

            if vector_count > 0:
                self.console.print(f"  [green]✓[/green] Vector Store: Connected ({vector_count} vectors)")
            else:
                try:
                    # Re-check connection if count was 0
                    self._event_loop_manager.run_async(
                        self.services.vector_store.get_stats()
                    )
                    self.console.print(f"  [green]✓[/green] Vector Store: Connected ({vector_count} vectors)")
                except Exception as e:
                    self.console.print(f"  [red]✗[/red] Vector Store: Error - {e}")

            self.console.print(
                f"  [green]✓[/green] Embedding API: "
                f"{self.services.config.embedding.api_url} ({self.services.config.embedding.model})"
            )

        except Exception as e:
            render_error(str(e), self.console)

    def _render_list(self, result) -> None:
        """Render list command output."""
        from rich.table import Table

        try:
            repos = self.services.metadata_store.get_repositories()

            if not repos:
                self.console.print("[yellow]No repositories indexed.[/yellow]")
                return

            table = Table(title="Indexed Repositories", border_style="blue")
            table.add_column("Root Path", style="cyan", no_wrap=True)
            table.add_column("Last Updated", style="magenta")

            for repo in repos:
                table.add_row(repo["root_path"], str(repo["updated_at"]))

            self.console.print(table)

        except Exception as e:
            render_error(str(e), self.console)

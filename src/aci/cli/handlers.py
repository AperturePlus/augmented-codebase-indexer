"""
Command handlers for ACI interactive REPL.

Provides handler functions for each REPL command, bridging the
interactive interface with the underlying services.
"""

from typing import TYPE_CHECKING

from aci.cli.parser import ParsedCommand
from aci.cli.router import ArgumentInfo, CommandResult, CommandRouter

if TYPE_CHECKING:
    from aci.cli.services import ServicesContainer


def create_help_handler(router: CommandRouter) -> callable:
    """
    Create a help command handler.

    Args:
        router: The command router to get command info from.

    Returns:
        Handler function for the help command.
    """

    def handle_help(command: ParsedCommand) -> CommandResult:
        """Display help information."""
        commands = router.get_available_commands()
        return CommandResult(
            success=True,
            message="help",
            data=commands,
        )

    return handle_help


def handle_exit(command: ParsedCommand) -> CommandResult:
    """Handle exit/quit commands."""
    return CommandResult(
        success=True,
        message="Goodbye!",
        should_exit=True,
    )


def create_index_handler(services: "ServicesContainer", context=None) -> callable:
    """
    Create an index command handler.

    Args:
        services: Services container with initialized services.
        context: Optional REPLContext for getting current codebase.

    Returns:
        Handler function for the index command.
    """

    def handle_index(command: ParsedCommand) -> CommandResult:
        """Handle index command."""
        # Use provided path or fall back to current codebase from context
        if command.args:
            path = command.args[0]
        elif context is not None and context.has_explicit_codebase():
            path = str(context.get_codebase())
        else:
            return CommandResult(
                success=False,
                message="Usage: index [path]\n\nIndex a directory for semantic search.\n"
                "Tip: Use 'use <path>' to set a codebase, then 'index' without arguments.",
            )

        return CommandResult(
            success=True,
            message=f"index:{path}",
            data={"path": path, "services": services},
        )

    return handle_index


def create_search_handler(services: "ServicesContainer") -> callable:
    """
    Create a search command handler.

    Args:
        services: Services container with initialized services.

    Returns:
        Handler function for the search command.
    """

    def handle_search(command: ParsedCommand) -> CommandResult:
        """Handle search command."""
        if not command.args:
            return CommandResult(
                success=False,
                message="Usage: search <query> [--limit=N] [--type=TYPE]\n\nSearch the indexed codebase.\n"
                "Types: chunk, function_summary, class_summary, file_summary",
            )

        query = " ".join(command.args)
        limit = command.kwargs.get("limit", command.kwargs.get("n"))
        artifact_type = command.kwargs.get("type", command.kwargs.get("t"))

        # Parse artifact types (can be comma-separated or multiple --type flags)
        artifact_types = None
        if artifact_type:
            if isinstance(artifact_type, list):
                artifact_types = artifact_type
            else:
                artifact_types = [t.strip() for t in artifact_type.split(",")]

        return CommandResult(
            success=True,
            message=f"search:{query}",
            data={"query": query, "limit": limit, "artifact_types": artifact_types, "services": services},
        )

    return handle_search


def create_status_handler(services: "ServicesContainer") -> callable:
    """
    Create a status command handler.

    Args:
        services: Services container with initialized services.

    Returns:
        Handler function for the status command.
    """

    def handle_status(command: ParsedCommand) -> CommandResult:
        """Handle status command."""
        return CommandResult(
            success=True,
            message="status",
            data={"services": services},
        )

    return handle_status


def create_update_handler(services: "ServicesContainer", context=None) -> callable:
    """
    Create an update command handler.

    Args:
        services: Services container with initialized services.
        context: Optional REPLContext for getting current codebase.

    Returns:
        Handler function for the update command.
    """

    def handle_update(command: ParsedCommand) -> CommandResult:
        """Handle update command."""
        # Use provided path or fall back to current codebase from context
        if command.args:
            path = command.args[0]
        elif context is not None and context.has_explicit_codebase():
            path = str(context.get_codebase())
        else:
            return CommandResult(
                success=False,
                message="Usage: update [path]\n\nIncrementally update the index.\n"
                "Tip: Use 'use <path>' to set a codebase, then 'update' without arguments.",
            )

        return CommandResult(
            success=True,
            message=f"update:{path}",
            data={"path": path, "services": services},
        )

    return handle_update


def create_list_handler(services: "ServicesContainer") -> callable:
    """
    Create a list command handler.

    Args:
        services: Services container with initialized services.

    Returns:
        Handler function for the list command.
    """

    def handle_list(command: ParsedCommand) -> CommandResult:
        """Handle list command."""
        return CommandResult(
            success=True,
            message="list",
            data={"services": services},
        )

    return handle_list


def create_reset_handler(services: "ServicesContainer") -> callable:
    """
    Create a reset command handler.

    Args:
        services: Services container with initialized services.

    Returns:
        Handler function for the reset command.
    """

    def handle_reset(command: ParsedCommand) -> CommandResult:
        """Handle reset command."""
        return CommandResult(
            success=True,
            message="reset",
            data={"services": services, "needs_confirmation": True},
        )

    return handle_reset


def create_use_handler(services: "ServicesContainer", context) -> callable:
    """
    Create a use command handler for setting the current codebase.

    Args:
        services: Services container with initialized services.
        context: REPLContext for managing codebase state.

    Returns:
        Handler function for the use command.
    """
    from pathlib import Path

    def handle_use(command: ParsedCommand) -> CommandResult:
        """Handle use command."""
        # No args - display current codebase
        if not command.args:
            current = context.get_codebase()
            if context.has_explicit_codebase():
                return CommandResult(
                    success=True,
                    message=f"Current codebase: {current}",
                )
            else:
                return CommandResult(
                    success=True,
                    message=f"No codebase set. Using current directory: {current}",
                )

        # With args - set the codebase
        path_str = command.args[0]
        path = Path(path_str).resolve()

        # Validate path exists
        if not path.exists():
            return CommandResult(
                success=False,
                message=f"Path '{path_str}' does not exist",
            )

        # Validate path is a directory
        if not path.is_dir():
            return CommandResult(
                success=False,
                message=f"Path '{path_str}' is not a directory",
            )

        # Check if path is indexed
        if not context.is_path_indexed(path):
            return CommandResult(
                success=False,
                message=f"Path '{path}' has not been indexed. Run 'index {path}' first.",
            )

        # Set the codebase
        context.set_codebase(path)
        return CommandResult(
            success=True,
            message=f"Codebase set to: {path}",
            data={"codebase_changed": True},
        )

    return handle_use


def register_all_commands(
    router: CommandRouter,
    services: "ServicesContainer",
    context=None,
) -> None:
    """
    Register all REPL commands with the router.

    Args:
        router: Command router to register with.
        services: Services container for handlers that need it.
        context: Optional REPLContext for codebase management.
    """
    # Help command
    router.register(
        name="help",
        handler=create_help_handler(router),
        description="Display available commands and their usage",
        usage="help",
        aliases=["?"],
    )

    # Exit commands
    router.register(
        name="exit",
        handler=handle_exit,
        description="Exit the interactive shell",
        usage="exit",
        aliases=["quit", "q"],
    )

    # Index command
    router.register(
        name="index",
        handler=create_index_handler(services, context),
        description="Index a directory for semantic search (uses current codebase if set)",
        usage="index [path]",
        arguments=[
            ArgumentInfo(
                name="path",
                description="Directory path to index (optional if codebase is set)",
                required=False,
            ),
        ],
    )

    # Search command
    router.register(
        name="search",
        handler=create_search_handler(services),
        description="Search the indexed codebase",
        usage="search <query> [--limit=N] [--type=TYPE]",
        arguments=[
            ArgumentInfo(
                name="query",
                description="Search query",
                required=True,
            ),
            ArgumentInfo(
                name="limit",
                description="Maximum number of results",
                required=False,
                default=10,
            ),
            ArgumentInfo(
                name="type",
                description="Artifact type filter (chunk, function_summary, class_summary, file_summary)",
                required=False,
                default=None,
            ),
        ],
    )

    # Status command
    router.register(
        name="status",
        handler=create_status_handler(services),
        description="Show index status and statistics",
        usage="status",
    )

    # Update command
    router.register(
        name="update",
        handler=create_update_handler(services, context),
        description="Incrementally update the index (uses current codebase if set)",
        usage="update [path]",
        arguments=[
            ArgumentInfo(
                name="path",
                description="Directory path to update (optional if codebase is set)",
                required=False,
            ),
        ],
    )

    # List command
    router.register(
        name="list",
        handler=create_list_handler(services),
        description="List all indexed repositories",
        usage="list",
    )

    # Reset command
    router.register(
        name="reset",
        handler=create_reset_handler(services),
        description="Clear the index (requires confirmation)",
        usage="reset",
    )

    # Use command (only if context is provided)
    if context is not None:
        router.register(
            name="use",
            handler=create_use_handler(services, context),
            description="Set or display the current codebase for search",
            usage="use [path]",
            arguments=[
                ArgumentInfo(
                    name="path",
                    description="Directory path to set as current codebase",
                    required=False,
                ),
            ],
        )

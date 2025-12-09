"""
Command router module for ACI interactive REPL.

Routes parsed commands to their appropriate handlers and manages
command registration, help generation, and error handling.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from aci.cli.parser import ParsedCommand


@dataclass
class ArgumentInfo:
    """
    Information about a command argument.

    Attributes:
        name: Argument name.
        description: Brief description of the argument.
        required: Whether the argument is required.
        default: Default value if not required.
    """

    name: str
    description: str
    required: bool = True
    default: Optional[Any] = None


@dataclass
class CommandInfo:
    """
    Information about a registered command.

    Attributes:
        name: Command name (lowercase).
        description: Brief description of what the command does.
        usage: Usage string showing syntax.
        arguments: List of argument information.
    """

    name: str
    description: str
    usage: str
    arguments: list[ArgumentInfo] = field(default_factory=list)


@dataclass
class CommandResult:
    """
    Result of command execution.

    Attributes:
        success: Whether the command executed successfully.
        message: Optional message to display.
        data: Optional data returned by the command.
        should_exit: Whether the REPL should exit after this command.
    """

    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
    should_exit: bool = False


# Type alias for command handlers
CommandHandler = Callable[[ParsedCommand], CommandResult]


class CommandRouter:
    """
    Routes commands to their handlers.

    Manages command registration, routing, and help generation.
    Handlers are registered with their metadata for help display.
    """

    def __init__(self) -> None:
        """Initialize the router with empty command registry."""
        self._handlers: dict[str, CommandHandler] = {}
        self._command_info: dict[str, CommandInfo] = {}

    def register(
        self,
        name: str,
        handler: CommandHandler,
        description: str,
        usage: str,
        arguments: Optional[list[ArgumentInfo]] = None,
        aliases: Optional[list[str]] = None,
    ) -> None:
        """
        Register a command handler.

        Args:
            name: Primary command name.
            handler: Function to handle the command.
            description: Brief description for help.
            usage: Usage string showing syntax.
            arguments: List of argument info.
            aliases: Alternative names for the command.
        """
        info = CommandInfo(
            name=name,
            description=description,
            usage=usage,
            arguments=arguments or [],
        )

        self._handlers[name] = handler
        self._command_info[name] = info

        # Register aliases pointing to same handler
        for alias in aliases or []:
            self._handlers[alias] = handler

    def route(self, command: ParsedCommand) -> CommandResult:
        """
        Route a command to its handler.

        Args:
            command: Parsed command to route.

        Returns:
            CommandResult from the handler, or error result if unknown.
        """
        if not command.name:
            # Empty command, just return success (no-op)
            return CommandResult(success=True)

        handler = self._handlers.get(command.name)
        if handler is None:
            return CommandResult(
                success=False,
                message=f"Unknown command: '{command.name}'. Type 'help' for available commands.",
            )

        try:
            return handler(command)
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error executing '{command.name}': {e}",
            )

    def get_available_commands(self) -> list[CommandInfo]:
        """
        Get information about all registered commands.

        Returns:
            List of CommandInfo for all registered commands (no aliases).
        """
        return list(self._command_info.values())

    def get_command_info(self, name: str) -> Optional[CommandInfo]:
        """
        Get information about a specific command.

        Args:
            name: Command name to look up.

        Returns:
            CommandInfo if found, None otherwise.
        """
        return self._command_info.get(name)

    def is_registered(self, name: str) -> bool:
        """
        Check if a command name is registered.

        Args:
            name: Command name to check.

        Returns:
            True if registered (including aliases).
        """
        return name in self._handlers

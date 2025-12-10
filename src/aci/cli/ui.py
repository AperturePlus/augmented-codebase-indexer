"""
UI components module for ACI interactive REPL.

Provides styled terminal output using Rich library for welcome banners,
prompts, help display, and error rendering.
"""

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from aci.cli.router import CommandInfo


# ASCII art banner for ACI
ACI_BANNER = r"""
    _    ____ ___   ____  _          _ _ 
   / \  / ___|_ _| / ___|| |__   ___| | |
  / _ \| |    | |  \___ \| '_ \ / _ \ | |
 / ___ \ |___ | |   ___) | | | |  __/ | |
/_/   \_\____|___| |____/|_| |_|\___|_|_|
"""

VERSION = "0.2.0"


def render_welcome_banner(console: Console) -> None:
    """
    Render the welcome banner with ASCII art and styled text.

    Displays the ACI logo, version, and quick start hints.

    Args:
        console: Rich Console instance for output.
    """
    # Create styled banner text
    banner_text = Text(ACI_BANNER, style="bold cyan")

    # Create welcome message
    welcome_content = Text()
    welcome_content.append("Augmented Codebase Indexer", style="bold white")
    welcome_content.append(f" v{VERSION}\n\n", style="dim")
    welcome_content.append("Type ", style="white")
    welcome_content.append("help", style="bold green")
    welcome_content.append(" for available commands, ", style="white")
    welcome_content.append("exit", style="bold yellow")
    welcome_content.append(" to quit.", style="white")

    # Combine banner and welcome message
    full_content = Text()
    full_content.append_text(banner_text)
    full_content.append("\n")
    full_content.append_text(welcome_content)

    console.print(
        Panel(
            full_content,
            border_style="cyan",
            padding=(0, 2),
        )
    )


def render_prompt() -> str:
    """
    Generate the colored prompt string for user input.

    Returns:
        Prompt string with ANSI color codes for prompt_toolkit.
    """
    # Return a simple styled prompt for prompt_toolkit
    # Using ANSI escape codes compatible with prompt_toolkit
    return "aci> "


def get_prompt_style() -> list[tuple[str, str]]:
    """
    Get prompt style tuples for prompt_toolkit.

    Returns:
        List of (style, text) tuples for styled prompt.
    """
    return [
        ("class:prompt", "aci"),
        ("class:prompt-arrow", "> "),
    ]


def render_help(commands: list["CommandInfo"], console: Console) -> None:
    """
    Render help information as a styled table.

    Displays all available commands with their descriptions and usage.

    Args:
        commands: List of CommandInfo objects to display.
        console: Rich Console instance for output.
    """
    table = Table(
        title="Available Commands",
        title_style="bold cyan",
        border_style="blue",
        show_header=True,
        header_style="bold white",
    )

    table.add_column("Command", style="green", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Usage", style="dim cyan")

    for cmd in sorted(commands, key=lambda c: c.name):
        table.add_row(cmd.name, cmd.description, cmd.usage)

    console.print(table)
    console.print()


def render_error(message: str, console: Console) -> None:
    """
    Render an error message in a visually distinct red panel.

    Args:
        message: Error message to display.
        console: Rich Console instance for output.
    """
    error_text = Text()
    error_text.append("Error: ", style="bold red")
    error_text.append(message, style="red")

    console.print(
        Panel(
            error_text,
            border_style="red",
            title="[bold red]Error[/bold red]",
            expand=False,
        )
    )


def render_success(message: str, console: Console) -> None:
    """
    Render a success message in a green panel.

    Args:
        message: Success message to display.
        console: Rich Console instance for output.
    """
    success_text = Text(message, style="green")

    console.print(
        Panel(
            success_text,
            border_style="green",
            expand=False,
        )
    )


def render_warning(message: str, console: Console) -> None:
    """
    Render a warning message in yellow.

    Args:
        message: Warning message to display.
        console: Rich Console instance for output.
    """
    console.print(f"[yellow]Warning:[/yellow] {message}")


def render_info(message: str, console: Console) -> None:
    """
    Render an informational message.

    Args:
        message: Info message to display.
        console: Rich Console instance for output.
    """
    console.print(f"[blue]Info:[/blue] {message}")

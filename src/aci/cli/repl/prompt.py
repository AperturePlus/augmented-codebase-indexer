"""
REPL prompt builder module.

Provides dynamic prompt generation that displays the current codebase name.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aci.cli.repl.context import REPLContext


class PromptBuilder:
    """
    Builds dynamic prompts showing current codebase.

    Generates prompt tokens for prompt_toolkit that display the current
    codebase name (directory name only) when a codebase is set.

    Attributes:
        _context: The REPL context containing codebase state.
    """

    def __init__(self, context: "REPLContext") -> None:
        """
        Initialize with REPL context.

        Args:
            context: The REPL context to read codebase state from.
        """
        self._context = context

    def get_prompt(self) -> list[tuple[str, str]]:
        """
        Return prompt tokens for prompt_toolkit.

        Returns a list of (style, text) tuples that form the prompt.
        When a codebase is set, shows: "codebase_name aci> "
        When no codebase is set, shows: "aci> "

        Returns:
            List of (style_class, text) tuples for prompt_toolkit.
        """
        tokens: list[tuple[str, str]] = []

        if self._context.has_explicit_codebase():
            codebase_display = self.get_codebase_display()
            tokens.append(("class:prompt.codebase", codebase_display))
            tokens.append(("class:prompt.separator", " "))

        tokens.append(("class:prompt", "aci"))
        tokens.append(("class:prompt.arrow", "> "))

        return tokens

    def get_codebase_display(self) -> str:
        """
        Get the display name for current codebase (directory name only).

        Extracts just the last component of the codebase path to keep
        the prompt concise.

        Returns:
            The directory name of the current codebase, or empty string
            if no codebase is set.
        """
        if not self._context.has_explicit_codebase():
            return ""

        codebase_path = self._context.get_codebase()
        return codebase_path.name

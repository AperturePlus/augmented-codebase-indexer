"""
Command completer module for ACI REPL.

Provides tab completion for commands and file paths in the interactive shell.
"""

import os
import sys
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion, PathCompleter

from aci.cli.router import CommandRouter


class CommandCompleter(Completer):
    """
    Custom completer for REPL commands.

    Provides completion for command names and file paths.
    """

    def __init__(self, router: CommandRouter):
        """
        Initialize the completer.

        Args:
            router: Command router to get available commands from.
        """
        self.router = router
        self.path_completer = PathCompleter(expanduser=True)
        self._command_names: list[str] = []
        self._update_commands()

    def _update_commands(self) -> None:
        """Update the list of available command names."""
        commands = self.router.get_available_commands()
        self._command_names = [cmd.name for cmd in commands]
        # Add aliases and built-in commands
        self._command_names.extend(["?", "q", "quit", "verbose", "use"])

    def get_completions(self, document, complete_event):
        """Generate completions for the current input."""
        text = document.text_before_cursor
        words = text.split()

        if not words or (len(words) == 1 and not text.endswith(" ")):
            # Complete command name
            word = words[0] if words else ""
            for cmd in self._command_names:
                if cmd.startswith(word.lower()):
                    yield Completion(cmd, start_position=-len(word))
        elif len(words) >= 1:
            # After command, check if we need path completion
            cmd_name = words[0].lower()
            if cmd_name in ("index", "update", "use"):
                # Extract the path portion for completion
                # Find where the path argument starts
                cmd_end = text.find(cmd_name) + len(cmd_name)
                path_text = text[cmd_end:].lstrip()

                # Generate path completions
                yield from self._complete_path(path_text, document)

    def _complete_path(self, path_text: str, document):
        """
        Generate path completions for the given partial path.

        Supports:
        - Relative paths (./foo, foo/bar)
        - Absolute paths (D:\\, /home/, C:\\Users)
        - Home directory expansion (~/)

        Args:
            path_text: The partial path text to complete.
            document: The original document for position calculation.
        """
        # Handle ~ expansion
        if path_text.startswith("~"):
            path_text = os.path.expanduser(path_text)

        # Handle empty path - complete from current directory
        if not path_text:
            base_dir = Path(".")
            prefix = ""
            replace_len = 0
        # Windows drive letter without path (e.g., "D:" or "D")
        elif sys.platform == "win32" and len(path_text) <= 2 and path_text[0].isalpha():
            if len(path_text) == 1:
                # Just "D" - offer "D:\"
                drive_letter = path_text.upper()
                drive_path = Path(f"{drive_letter}:/")
                if drive_path.exists():
                    yield Completion(
                        f"{drive_letter}:\\",
                        start_position=-1,
                        display=f"{drive_letter}:\\",
                    )
                return
            elif path_text.endswith(":"):
                # "D:" - first offer "D:\" then list root contents
                drive_letter = path_text[0].upper()
                base_dir = Path(f"{drive_letter}:/")
                prefix = ""
                replace_len = 0
            else:
                base_dir = Path(".")
                prefix = path_text
                replace_len = len(path_text)
        else:
            path = Path(path_text)
            # Check if path_text ends with separator - user wants contents of directory
            if path_text.endswith(os.sep) or path_text.endswith("/"):
                base_dir = path
                prefix = ""
                replace_len = 0
            elif path.exists() and path.is_dir():
                # Existing directory without trailing separator - list its contents
                base_dir = path
                prefix = ""
                replace_len = 0
            else:
                # Partial name - complete in parent directory
                if path.is_absolute():
                    base_dir = path.parent
                else:
                    base_dir = path.parent if str(path.parent) != "." else Path(".")
                prefix = path.name
                replace_len = len(prefix)

        # Get completions from the base directory
        try:
            if not base_dir.exists():
                return

            for entry in sorted(base_dir.iterdir()):
                name = entry.name

                # Skip hidden paths (starting with . or $, or Windows hidden attribute)
                if self._is_hidden_path(entry):
                    continue

                if prefix and not name.lower().startswith(prefix.lower()):
                    continue

                # Add trailing separator for directories
                is_dir = entry.is_dir()
                display = name + os.sep if is_dir else name

                yield Completion(
                    name + (os.sep if is_dir else ""),
                    start_position=-replace_len,
                    display=display,
                )
        except (OSError, PermissionError):
            # Silently ignore permission errors during completion
            return

    def _is_hidden_path(self, path: Path) -> bool:
        """
        Check if a path is hidden and should be excluded from completion.

        Detects:
        - Paths starting with '.' (Unix hidden files)
        - Paths starting with '$' (Windows system folders like $RECYCLE.BIN)
        - Windows hidden file attribute

        Args:
            path: Path to check.

        Returns:
            True if the path is hidden.
        """
        name = path.name

        # Skip paths starting with . or $ (common hidden/system patterns)
        if name.startswith(".") or name.startswith("$"):
            return True

        # On Windows, also check the hidden file attribute
        if sys.platform == "win32":
            try:
                import stat
                attrs = path.stat().st_file_attributes  # type: ignore[attr-defined]
                if attrs & stat.FILE_ATTRIBUTE_HIDDEN:
                    return True
            except (OSError, AttributeError):
                pass

        return False

"""
History manager module for ACI interactive REPL.

Provides functionality to persist and load command history across
interactive sessions, enabling users to recall previous commands.
"""

from pathlib import Path


class HistoryManager:
    """
    Manages command history persistence for the interactive REPL.

    Handles loading history from disk on session start and saving
    history when commands are executed or the session ends.

    Attributes:
        history_file: Path to the history file.
        _history: In-memory list of command strings.
    """

    DEFAULT_HISTORY_FILE = Path(".aci/history")

    def __init__(self, history_file: Path | None = None):
        """
        Initialize the history manager.

        Args:
            history_file: Path to the history file. If None, uses
                         the default location '.aci/history'.
        """
        self.history_file = history_file or self.DEFAULT_HISTORY_FILE
        self._history: list[str] = []

    def load(self) -> list[str]:
        """
        Load command history from the history file.

        Handles file not found gracefully on first run by returning
        an empty list. Each line in the file represents one command.

        Returns:
            List of command strings from history, in chronological order.
        """
        if not self.history_file.exists():
            self._history = []
            return []

        try:
            content = self.history_file.read_text(encoding="utf-8")
            # Filter out empty lines
            self._history = [
                line for line in content.splitlines() if line.strip()
            ]
            return self._history.copy()
        except (OSError, UnicodeDecodeError):
            # Handle read errors gracefully
            self._history = []
            return []

    def save(self, history: list[str]) -> None:
        """
        Save command history to the history file.

        Creates the parent directory if it doesn't exist.
        Each command is written on a separate line.

        Args:
            history: List of command strings to save.
        """
        # Filter out empty commands
        filtered = [cmd for cmd in history if cmd.strip()]

        # Ensure parent directory exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.history_file.write_text(
                "\n".join(filtered) + ("\n" if filtered else ""),
                encoding="utf-8",
            )
            self._history = filtered.copy()
        except OSError:
            # Silently fail on write errors to not disrupt the session
            pass

    def append(self, command: str) -> None:
        """
        Append a single command to the history.

        The command is added to both in-memory history and persisted
        to disk immediately.

        Args:
            command: The command string to append.
        """
        if not command.strip():
            return

        self._history.append(command)

        # Ensure parent directory exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Append to file
            with self.history_file.open("a", encoding="utf-8") as f:
                f.write(command + "\n")
        except OSError:
            # Silently fail on write errors
            pass

    def get_history(self) -> list[str]:
        """
        Get the current in-memory history.

        Returns:
            Copy of the current history list.
        """
        return self._history.copy()

    def clear(self) -> None:
        """
        Clear all history from memory and disk.
        """
        self._history = []
        if self.history_file.exists():
            try:
                self.history_file.unlink()
            except OSError:
                pass

"""
Watch configuration module for file watching service.

Provides configuration for the file watcher including debounce settings,
ignore patterns, and verbose logging options.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _get_default_debounce_ms() -> int:
    """Get default debounce delay from environment or use default."""
    env_value = os.environ.get("ACI_WATCH_DEBOUNCE_MS")
    if env_value is not None:
        try:
            return int(env_value)
        except ValueError:
            pass
    return 2000


@dataclass
class WatchConfig:
    """
    Configuration for the file watching service.

    Attributes:
        watch_path: Directory path to watch for file changes
        debounce_ms: Debounce delay in milliseconds (default: 2000ms or ACI_WATCH_DEBOUNCE_MS)
        ignore_patterns: Additional patterns to ignore beyond default gitignore
        verbose: Enable verbose logging output
    """

    watch_path: Path
    debounce_ms: int = field(default_factory=_get_default_debounce_ms)
    ignore_patterns: list[str] = field(default_factory=list)
    verbose: bool = False

    def __post_init__(self) -> None:
        """Ensure watch_path is a Path object."""
        if isinstance(self.watch_path, str):
            self.watch_path = Path(self.watch_path)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize configuration to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "watch_path": str(self.watch_path),
            "debounce_ms": self.debounce_ms,
            "ignore_patterns": list(self.ignore_patterns),
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchConfig":
        """
        Create WatchConfig from a dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            WatchConfig instance
        """
        return cls(
            watch_path=Path(data["watch_path"]),
            debounce_ms=data.get("debounce_ms", _get_default_debounce_ms()),
            ignore_patterns=list(data.get("ignore_patterns", [])),
            verbose=data.get("verbose", False),
        )

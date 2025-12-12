"""
Abstract interfaces for file scanning operations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Set

from .models import ScannedFile


class FileScannerInterface(ABC):
    """
    Abstract interface for file scanning operations.

    Implementations should provide recursive directory scanning with
    configurable file extension filtering and ignore patterns.
    """

    @abstractmethod
    def scan(self, root_path: Path) -> Iterator[ScannedFile]:
        """
        Recursively scan a directory and yield ScannedFile objects.

        Args:
            root_path: Root directory to scan

        Yields:
            ScannedFile objects for each matching file

        Notes:
            - Skips files matching ignore patterns
            - Only yields files with configured extensions
            - Logs errors and continues on unreadable files
        """
        pass

    @abstractmethod
    def set_extensions(self, extensions: Set[str]) -> None:
        """
        Set the file extensions to include in scanning.

        Args:
            extensions: Set of extensions including the dot (e.g., {'.py', '.js'})
        """
        pass

    @abstractmethod
    def set_ignore_patterns(self, patterns: list[str]) -> None:
        """
        Set ignore patterns using gitignore syntax.

        Args:
            patterns: List of gitignore-style patterns
        """
        pass

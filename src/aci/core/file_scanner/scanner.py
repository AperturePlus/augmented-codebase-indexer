"""
FileScanner implementation for recursive directory scanning.
"""

import fnmatch
import hashlib
import logging
import sys
from pathlib import Path
from typing import Iterator, Set

from aci.core.gitignore_manager import GitignoreManager
from aci.core.symlink_validator import SymlinkValidator

from .interfaces import FileScannerInterface
from .language_registry import LanguageRegistry, get_default_registry
from .models import SENSITIVE_DENYLIST, ScannedFile

logger = logging.getLogger(__name__)


def _compute_sha256(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _detect_language(file_path: Path, registry: LanguageRegistry | None = None) -> str:
    """
    Detect programming language from file extension.

    Args:
        file_path: Path to the file
        registry: Optional LanguageRegistry to use. If None, uses default registry.

    Returns:
        Language identifier or 'unknown'
    """
    reg = registry or get_default_registry()
    return reg.detect_from_path(file_path)


class FileScanner(FileScannerInterface):
    """
    Concrete implementation of FileScannerInterface.

    Provides recursive directory scanning with:
    - File extension filtering
    - Gitignore-style pattern matching via GitignoreManager
    - SHA-256 content hashing
    - Graceful error handling for unreadable files
    - Extensible language detection via LanguageRegistry
    - Symlink validation for security
    """

    # Default ignore patterns applied before .gitignore patterns
    DEFAULT_IGNORE_PATTERNS: list[str] = [
        # Hidden files/directories (Unix-style and Windows system folders)
        ".*",
        "$*",
        # Python
        "__pycache__",
        "*.pyc",
        "*.egg-info",
        ".venv",
        "venv",
        ".tox",
        ".pytest_cache",
        ".uv_cache",
        ".ruff_cache",
        ".hypothesis",
        # JavaScript/Node
        "node_modules",
        # Build outputs
        "dist",
        "build",
    ]

    def __init__(
        self,
        extensions: Set[str] | None = None,
        ignore_patterns: list[str] | None = None,
        language_registry: LanguageRegistry | None = None,
        follow_symlinks: bool = False,
        case_sensitive: bool | None = None,
    ):
        """
        Initialize the FileScanner.

        Args:
            extensions: Set of file extensions to include (e.g., {'.py', '.js'}).
                       If None, defaults to common code file extensions.
            ignore_patterns: List of gitignore-style patterns to exclude.
                            If None, defaults to common ignore patterns.
            language_registry: Custom LanguageRegistry for language detection.
                              If None, uses the global default registry.
            follow_symlinks: Whether to follow symlinks (default: False for security).
                            When True, symlinks are validated before following.
            case_sensitive: Override case sensitivity for pattern matching.
                           If None, auto-detects based on platform (Windows=insensitive).
        """
        self._extensions: Set[str] = extensions or {".py", ".js", ".ts", ".go"}
        self._ignore_patterns: list[str] = (
            ignore_patterns if ignore_patterns is not None 
            else list(self.DEFAULT_IGNORE_PATTERNS)
        )
        self._language_registry = language_registry or get_default_registry()
        self._follow_symlinks = follow_symlinks

        # Auto-detect case sensitivity based on platform if not specified
        if case_sensitive is None:
            self._case_sensitive = sys.platform != "win32"
        else:
            self._case_sensitive = case_sensitive

        # GitignoreManager and SymlinkValidator are initialized per-scan
        self._gitignore_manager: GitignoreManager | None = None
        self._symlink_validator: SymlinkValidator | None = None
        self._root_path: Path | None = None

    def _matches_sensitive_denylist(self, path: Path) -> bool:
        """
        Check if a path matches any pattern in the sensitive denylist.

        This check cannot be overridden by user configuration and ensures
        that sensitive files (credentials, keys, etc.) are never indexed.
        """
        name = path.name

        for pattern in SENSITIVE_DENYLIST:
            if name == pattern:
                return True
            if fnmatch.fnmatch(name, pattern):
                return True

        return False

    def set_extensions(self, extensions: Set[str]) -> None:
        """Set the file extensions to include in scanning."""
        self._extensions = extensions

    def set_ignore_patterns(self, patterns: list[str]) -> None:
        """Set ignore patterns using gitignore syntax."""
        self._ignore_patterns = patterns

    def _should_ignore(self, path: Path, is_dir: bool = False) -> bool:
        """
        Check if a path should be ignored based on patterns.

        The sensitive denylist is always checked first and cannot be overridden.
        """
        if self._matches_sensitive_denylist(path):
            logger.debug(f"Ignoring sensitive file/directory: {path}")
            return True

        if self._gitignore_manager is None:
            return False

        return self._gitignore_manager.matches(path, is_dir=is_dir)

    def _has_matching_extension(self, path: Path) -> bool:
        """Check if file has a matching extension."""
        return path.suffix.lower() in self._extensions

    def scan(self, root_path: Path) -> Iterator[ScannedFile]:
        """
        Recursively scan a directory and yield ScannedFile objects.

        Args:
            root_path: Root directory to scan

        Yields:
            ScannedFile objects for each matching file
        """
        root_path = Path(root_path).resolve()
        self._root_path = root_path

        if not root_path.exists():
            logger.error(f"Root path does not exist: {root_path}")
            return

        if not root_path.is_dir():
            logger.error(f"Root path is not a directory: {root_path}")
            return

        # Initialize GitignoreManager for this scan
        self._gitignore_manager = GitignoreManager(root_path, case_sensitive=self._case_sensitive)

        # Add default ignore patterns first (lowest precedence)
        self._gitignore_manager.add_default_patterns(self._ignore_patterns)

        # Load .gitignore hierarchy from root and all subdirectories
        total_patterns = self._gitignore_manager.load_gitignore_hierarchy(root_path)
        logger.debug(f"Loaded {total_patterns} patterns from .gitignore hierarchy")

        # Initialize SymlinkValidator for this scan
        self._symlink_validator = SymlinkValidator(root_path, SENSITIVE_DENYLIST)

        # Track visited real paths to prevent cycles
        visited: set[Path] = set()
        yield from self._scan_directory(root_path, visited)

    def _scan_directory(
        self, current_path: Path, visited: set[Path]
    ) -> Iterator[ScannedFile]:
        """
        Recursively scan a directory.

        Args:
            current_path: Current directory being scanned
            visited: Set of resolved paths already visited to prevent cycles

        Yields:
            ScannedFile objects for matching files
        """
        try:
            # Resolve symlinks to check for cycles
            real_path = current_path.resolve()
            if real_path in visited:
                logger.debug(f"Skipping recursive cycle: {current_path} -> {real_path}")
                return
            visited.add(real_path)

            entries = sorted(current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError as e:
            logger.warning(f"Permission denied accessing directory: {current_path} - {e}")
            return
        except OSError as e:
            logger.warning(f"Error accessing directory: {current_path} - {e}")
            return

        for entry in entries:
            is_dir = entry.is_dir()
            is_symlink = entry.is_symlink()

            # Handle symlinks
            if is_symlink:
                if not self._follow_symlinks:
                    logger.debug(f"Skipping symlink (follow_symlinks=False): {entry}")
                    continue

                # Validate symlink before following
                if self._symlink_validator is not None:
                    result = self._symlink_validator.is_safe_symlink(entry, visited)
                    if not result.safe:
                        logger.warning(f"Skipping unsafe symlink: {entry} - {result.reason}")
                        continue

            # Check if this entry should be ignored
            if self._should_ignore(entry, is_dir=is_dir):
                logger.debug(f"Ignoring: {entry}")
                continue

            if is_dir:
                # Recursively scan subdirectories
                yield from self._scan_directory(entry, visited)
            elif entry.is_file():
                # Check extension and process file
                if self._has_matching_extension(entry):
                    scanned = self._scan_file(entry)
                    if scanned is not None:
                        yield scanned

        # Remove from visited when backtracking to allow other paths to visit this real dir
        visited.remove(real_path)

    def _scan_file(self, file_path: Path) -> ScannedFile | None:
        """
        Scan a single file and return its ScannedFile representation.

        Args:
            file_path: Path to the file to scan

        Returns:
            ScannedFile object or None if the file couldn't be read
        """
        try:
            # Get file stats
            stat = file_path.stat()
            size_bytes = stat.st_size
            modified_time = stat.st_mtime

            # Check file size (limit to 10MB to prevent OOM)
            if size_bytes > 10 * 1024 * 1024:  # 10 MB
                logger.warning(f"Skipping large file ({size_bytes} bytes): {file_path}")
                return None

            # Read content
            content = file_path.read_text(encoding="utf-8")

            # Compute hash
            content_hash = _compute_sha256(content)

            # Detect language using the configured registry
            language = _detect_language(file_path, self._language_registry)

            return ScannedFile(
                path=file_path.resolve(),
                content=content,
                language=language,
                size_bytes=size_bytes,
                modified_time=modified_time,
                content_hash=content_hash,
            )
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to decode file as UTF-8: {file_path} - {e}")
            return None
        except PermissionError as e:
            logger.warning(f"Permission denied reading file: {file_path} - {e}")
            return None
        except OSError as e:
            logger.warning(f"Error reading file: {file_path} - {e}")
            return None

"""
FileScanner module for Project ACI.

Provides recursive directory scanning with file extension filtering,
gitignore pattern support, and content hashing.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Set

import pathspec
import yaml

logger = logging.getLogger(__name__)

# Default path to the languages configuration file
_DEFAULT_LANGUAGES_CONFIG = Path(__file__).parent / "languages.yaml"


class LanguageRegistry:
    """
    Extensible registry for mapping file extensions to programming languages.

    Supports loading from YAML configuration and runtime registration of new
    languages and their extensions, making it easy to add support for additional
    languages without modifying the core scanner code.

    Example:
        >>> registry = LanguageRegistry()
        >>> registry.register("elixir", [".ex", ".exs"])
        >>> registry.detect(".ex")
        'elixir'

        >>> # Load from custom config
        >>> registry = LanguageRegistry.from_yaml("custom_languages.yaml")
    """

    def __init__(self, load_defaults: bool = True):
        """
        Initialize the language registry.

        Args:
            load_defaults: If True, load default language mappings from languages.yaml.
        """
        self._extension_to_language: dict[str, str] = {}
        self._language_to_extensions: dict[str, set[str]] = {}

        if load_defaults:
            self._load_from_yaml(_DEFAULT_LANGUAGES_CONFIG)

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "LanguageRegistry":
        """
        Create a LanguageRegistry from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            LanguageRegistry instance with loaded mappings

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config file format is invalid
        """
        registry = cls(load_defaults=False)
        registry._load_from_yaml(Path(config_path))
        return registry

    def _load_from_yaml(self, config_path: Path) -> None:
        """
        Load language mappings from a YAML file.

        Expected format:
            language_name:
              - .ext1
              - .ext2
        """
        if not config_path.exists():
            logger.warning(f"Languages config not found: {config_path}, using empty registry")
            return

        try:
            content = config_path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)

            if data is None:
                return

            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid languages config format: expected dict, got {type(data)}"
                )

            for language, extensions in data.items():
                if not isinstance(extensions, list):
                    logger.warning(
                        f"Invalid extensions for {language}: expected list, got {type(extensions)}"
                    )
                    continue
                for ext in extensions:
                    self._add_mapping(str(ext), str(language))

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse languages config: {e}")
            raise ValueError(f"Invalid YAML in languages config: {e}") from e

    def _add_mapping(self, extension: str, language: str) -> None:
        """Internal method to add a single extension-language mapping."""
        ext_lower = extension.lower()
        self._extension_to_language[ext_lower] = language

        if language not in self._language_to_extensions:
            self._language_to_extensions[language] = set()
        self._language_to_extensions[language].add(ext_lower)

    def register(self, language: str, extensions: list[str]) -> "LanguageRegistry":
        """
        Register a new language with its file extensions.

        Args:
            language: Language identifier (e.g., 'rust', 'ruby')
            extensions: List of file extensions including the dot (e.g., ['.rs'])

        Returns:
            Self for method chaining

        Example:
            >>> registry.register("elixir", [".ex", ".exs"])
        """
        for ext in extensions:
            self._add_mapping(ext, language)
        return self

    def unregister(self, language: str) -> "LanguageRegistry":
        """
        Remove a language and all its extensions from the registry.

        Args:
            language: Language identifier to remove

        Returns:
            Self for method chaining
        """
        if language in self._language_to_extensions:
            for ext in self._language_to_extensions[language]:
                self._extension_to_language.pop(ext, None)
            del self._language_to_extensions[language]
        return self

    def detect(self, extension: str) -> str:
        """
        Detect language from file extension.

        Args:
            extension: File extension including the dot (e.g., '.py')

        Returns:
            Language identifier or 'unknown' if not recognized
        """
        return self._extension_to_language.get(extension.lower(), "unknown")

    def detect_from_path(self, file_path: Path) -> str:
        """
        Detect language from a file path.

        Args:
            file_path: Path to the file

        Returns:
            Language identifier or 'unknown' if not recognized
        """
        return self.detect(file_path.suffix)

    def get_extensions(self, language: str) -> set[str]:
        """
        Get all registered extensions for a language.

        Args:
            language: Language identifier

        Returns:
            Set of extensions (empty if language not registered)
        """
        return self._language_to_extensions.get(language, set()).copy()

    def get_all_extensions(self) -> set[str]:
        """Get all registered file extensions."""
        return set(self._extension_to_language.keys())

    def get_all_languages(self) -> set[str]:
        """Get all registered language identifiers."""
        return set(self._language_to_extensions.keys())

    def is_supported(self, extension: str) -> bool:
        """Check if an extension is registered."""
        return extension.lower() in self._extension_to_language


# Global default registry instance
_default_registry = LanguageRegistry()


def get_default_registry() -> LanguageRegistry:
    """Get the global default language registry."""
    return _default_registry


# Legacy compatibility - maps to default registry
EXTENSION_TO_LANGUAGE: dict[str, str] = _default_registry._extension_to_language


@dataclass
class ScannedFile:
    """
    Represents a scanned file with its metadata and content.

    Attributes:
        path: Absolute path to the file
        content: File content as UTF-8 string
        language: Detected language identifier ('python', 'javascript', 'typescript', 'go', 'unknown')
        size_bytes: File size in bytes
        modified_time: File modification timestamp (Unix epoch)
        content_hash: SHA-256 hash of the file content
    """

    path: Path
    content: str
    language: str
    size_bytes: int
    modified_time: float
    content_hash: str


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
    reg = registry or _default_registry
    return reg.detect_from_path(file_path)


class FileScanner(FileScannerInterface):
    """
    Concrete implementation of FileScannerInterface.

    Provides recursive directory scanning with:
    - File extension filtering
    - Gitignore-style pattern matching (using pathspec library)
    - SHA-256 content hashing
    - Graceful error handling for unreadable files
    - Extensible language detection via LanguageRegistry
    """

    def __init__(
        self,
        extensions: Set[str] | None = None,
        ignore_patterns: list[str] | None = None,
        language_registry: LanguageRegistry | None = None,
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
        """
        self._extensions: Set[str] = extensions or {".py", ".js", ".ts", ".go"}
        self._ignore_patterns: list[str] = ignore_patterns or [
            "__pycache__",
            "*.pyc",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "*.egg-info",
            "dist",
            "build",
            ".tox",
            ".pytest_cache",
            ".uv_cache",
            ".ruff_cache",
            ".hypothesis",
        ]
        self._language_registry = language_registry or _default_registry
        self._pathspec: pathspec.PathSpec | None = None
        self._update_pathspec()

    def _update_pathspec(self) -> None:
        """Update the pathspec matcher from current ignore patterns."""
        if self._ignore_patterns:
            self._pathspec = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, self._ignore_patterns
            )
        else:
            self._pathspec = None

    def set_extensions(self, extensions: Set[str]) -> None:
        """Set the file extensions to include in scanning."""
        self._extensions = extensions

    def set_ignore_patterns(self, patterns: list[str]) -> None:
        """Set ignore patterns using gitignore syntax."""
        self._ignore_patterns = patterns
        self._update_pathspec()

    def _should_ignore(self, path: Path, root_path: Path) -> bool:
        """
        Check if a path should be ignored based on patterns.

        Args:
            path: Path to check
            root_path: Root directory for relative path calculation

        Returns:
            True if the path should be ignored
        """
        if self._pathspec is None:
            return False

        # Get relative path from root for pattern matching
        try:
            rel_path = path.relative_to(root_path)
            rel_path_str = str(rel_path).replace("\\", "/")

            # For directories, also check with trailing slash
            if path.is_dir():
                return self._pathspec.match_file(rel_path_str) or self._pathspec.match_file(
                    rel_path_str + "/"
                )
            return self._pathspec.match_file(rel_path_str)
        except ValueError:
            # Path is not relative to root_path
            return False

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

        Notes:
            - Skips files/directories matching ignore patterns
            - Only yields files with configured extensions
            - Logs errors and continues on unreadable files
            - Detects and prevents infinite recursion from symlink loops
            - Automatically loads .gitignore patterns from root_path
        """
        root_path = Path(root_path).resolve()

        if not root_path.exists():
            logger.error(f"Root path does not exist: {root_path}")
            return

        if not root_path.is_dir():
            logger.error(f"Root path is not a directory: {root_path}")
            return

        # Load .gitignore patterns from root path if present
        self._load_gitignore(root_path)

        # Track visited real paths to prevent cycles
        visited = set()
        yield from self._scan_directory(root_path, root_path, visited)

    def _load_gitignore(self, root_path: Path) -> None:
        """
        Load additional ignore patterns from .gitignore file.

        Args:
            root_path: Root directory containing .gitignore
        """
        gitignore_path = root_path / ".gitignore"
        if not gitignore_path.exists():
            return

        try:
            content = gitignore_path.read_text(encoding="utf-8")
            gitignore_patterns = []
            for line in content.splitlines():
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    gitignore_patterns.append(line)

            if gitignore_patterns:
                # Merge with existing patterns (avoid duplicates)
                existing = set(self._ignore_patterns)
                for pattern in gitignore_patterns:
                    if pattern not in existing:
                        self._ignore_patterns.append(pattern)
                self._update_pathspec()
                logger.debug(f"Loaded {len(gitignore_patterns)} patterns from .gitignore")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to read .gitignore: {e}")

    def _scan_directory(
        self, current_path: Path, root_path: Path, visited: Set[Path]
    ) -> Iterator[ScannedFile]:
        """
        Recursively scan a directory.

        Args:
            current_path: Current directory being scanned
            root_path: Original root directory for relative path calculation
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
            # Check if this entry should be ignored
            if self._should_ignore(entry, root_path):
                logger.debug(f"Ignoring: {entry}")
                continue

            if entry.is_dir():
                # Recursively scan subdirectories
                # Pass a copy of visited to avoid polluting sibling branches? 
                # No, for cycle detection we want to track path from root.
                # Actually, for general graph traversal, we keep visited.
                # But to avoid excluding same real dir reached via different valid non-cyclic paths?
                # For file system tree, usually we just want to avoid LOOPS.
                # So adding to visited is correct for current recursion stack.
                # However, to support same dir reachable via different paths (if not cyclic),
                # strict visited might be too aggressive if structure is DAG.
                # But for typical "scan tree", strict visited is safest and simplest.
                yield from self._scan_directory(entry, root_path, visited)
            elif entry.is_file():
                # Check extension and process file
                if self._has_matching_extension(entry):
                    scanned = self._scan_file(entry)
                    if scanned is not None:
                        yield scanned
        
        # Remove from visited when backtracking to allow other paths to visit this real dir
        # (if we want to support DAG structures, though typically unnecessary for pure tree scan)
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

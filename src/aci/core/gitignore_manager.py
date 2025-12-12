"""
GitignoreManager module for Project ACI.

Provides proper gitignore pattern parsing and matching with support for:
- Nested .gitignore files with proper scoping
- Pattern precedence (later patterns override earlier ones)
- Negation patterns (!)
- Directory-only patterns (trailing /)
- Anchored patterns (leading /)
- Double-star globs (**)
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)


@dataclass
class GitignorePattern:
    """
    A parsed gitignore pattern with metadata.

    Attributes:
        raw: Original pattern string (e.g., "!/important.py")
        pattern: Normalized pattern for matching (e.g., "important.py")
        negation: True if pattern starts with ! (re-includes files)
        directory_only: True if pattern ends with / (matches only directories)
        anchored: True if pattern starts with / (root-relative only)
        source_path: Path to .gitignore file containing this pattern
        source_depth: Depth of .gitignore from root (0 = root)
    """

    raw: str
    pattern: str
    negation: bool
    directory_only: bool
    anchored: bool
    source_path: Path
    source_depth: int = 0

    @classmethod
    def parse(cls, raw_line: str, source_path: Path, source_depth: int = 0) -> "GitignorePattern":
        """
        Parse a raw gitignore line into a GitignorePattern.

        Args:
            raw_line: Raw line from .gitignore file (already stripped)
            source_path: Path to the .gitignore file
            source_depth: Depth of .gitignore from root (0 = root)

        Returns:
            Parsed GitignorePattern instance
        """
        pattern = raw_line
        negation = False
        directory_only = False
        anchored = False

        # Handle negation (!)
        if pattern.startswith("!"):
            negation = True
            pattern = pattern[1:]

        # Handle directory-only (trailing /)
        if pattern.endswith("/"):
            directory_only = True
            pattern = pattern[:-1]

        # Handle anchored patterns (leading /)
        if pattern.startswith("/"):
            anchored = True
            pattern = pattern[1:]

        return cls(
            raw=raw_line,
            pattern=pattern,
            negation=negation,
            directory_only=directory_only,
            anchored=anchored,
            source_path=source_path,
            source_depth=source_depth,
        )


class GitignoreManager:
    """
    Manages gitignore patterns with proper scoping and precedence.

    Supports:
    - Multiple .gitignore files at different directory levels
    - Pattern precedence (later patterns override earlier ones)
    - Negation patterns (!)
    - Directory-only patterns (trailing /)
    - Anchored patterns (leading /)
    - Double-star globs (**)
    """

    def __init__(self, root_path: Path, case_sensitive: bool | None = None):
        """
        Initialize the GitignoreManager.

        Args:
            root_path: Root directory for pattern matching
            case_sensitive: Override case sensitivity (None = auto-detect from platform)
        """
        self._root_path = Path(root_path).resolve()
        
        # Auto-detect case sensitivity based on platform if not specified
        if case_sensitive is None:
            # Windows is case-insensitive, POSIX is case-sensitive
            self._case_sensitive = sys.platform != "win32"
        else:
            self._case_sensitive = case_sensitive

        # Store patterns with their metadata
        self._patterns: list[GitignorePattern] = []
        
        # Compiled pathspec for efficient matching (rebuilt when patterns change)
        self._pathspec: pathspec.PathSpec | None = None
        self._pathspec_dirty = True

    def load_gitignore(self, gitignore_path: Path) -> int:
        """
        Load patterns from a .gitignore file.

        Args:
            gitignore_path: Path to the .gitignore file

        Returns:
            Number of patterns loaded

        Raises:
            No exceptions - errors are logged and the method returns 0
        """
        gitignore_path = Path(gitignore_path).resolve()
        
        if not gitignore_path.exists():
            logger.debug(f"Gitignore file not found: {gitignore_path}")
            return 0

        # Calculate depth from root
        try:
            rel_path = gitignore_path.parent.relative_to(self._root_path)
            source_depth = len(rel_path.parts)
        except ValueError:
            # gitignore_path is not under root_path
            source_depth = 0

        patterns_loaded = 0

        try:
            content = gitignore_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            logger.warning(f"Invalid UTF-8 encoding in {gitignore_path}: {e}")
            return 0
        except PermissionError as e:
            logger.warning(f"Permission denied reading {gitignore_path}: {e}")
            return 0
        except OSError as e:
            logger.warning(f"Error reading {gitignore_path}: {e}")
            return 0

        for line in content.splitlines():
            # Strip whitespace
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip comments (lines starting with #)
            if line.startswith("#"):
                continue

            try:
                pattern = GitignorePattern.parse(line, gitignore_path, source_depth)
                self._patterns.append(pattern)
                patterns_loaded += 1
            except Exception as e:
                logger.warning(f"Malformed pattern '{line}' in {gitignore_path}: {e}")
                continue

        if patterns_loaded > 0:
            self._pathspec_dirty = True
            logger.debug(f"Loaded {patterns_loaded} patterns from {gitignore_path}")

        return patterns_loaded

    def add_default_patterns(self, patterns: list[str]) -> None:
        """
        Add default ignore patterns (applied before .gitignore patterns).

        These patterns are treated as if they came from a virtual .gitignore
        at the root with depth -1 (lowest precedence).

        Args:
            patterns: List of gitignore-style patterns
        """
        virtual_source = self._root_path / ".gitignore.defaults"
        
        for raw_pattern in patterns:
            raw_pattern = raw_pattern.strip()
            if not raw_pattern or raw_pattern.startswith("#"):
                continue
            
            pattern = GitignorePattern.parse(raw_pattern, virtual_source, source_depth=-1)
            # Insert at the beginning so they have lowest precedence
            self._patterns.insert(0, pattern)

        if patterns:
            self._pathspec_dirty = True

    def load_gitignore_hierarchy(self, root_path: Path | None = None) -> int:
        """
        Load .gitignore files from root and all subdirectories.

        This method walks the directory tree and loads all .gitignore files,
        properly scoping patterns relative to their containing directory.
        Patterns are loaded in order from root to deepest directory, ensuring
        proper precedence (later/deeper patterns override earlier/shallower ones).

        Args:
            root_path: Root directory to scan (defaults to manager's root_path)

        Returns:
            Total number of patterns loaded from all .gitignore files
        """
        if root_path is None:
            root_path = self._root_path
        else:
            root_path = Path(root_path).resolve()

        total_patterns = 0
        gitignore_files: list[tuple[int, Path]] = []  # (depth, path)

        # Walk the directory tree to find all .gitignore files
        try:
            for dirpath, dirnames, filenames in root_path.walk():
                if ".gitignore" in filenames:
                    gitignore_path = dirpath / ".gitignore"
                    try:
                        rel_path = dirpath.relative_to(self._root_path)
                        depth = len(rel_path.parts)
                    except ValueError:
                        depth = 0
                    gitignore_files.append((depth, gitignore_path))

                # Skip hidden directories and common non-code directories
                # to avoid unnecessary traversal (but still process their .gitignore)
                dirnames[:] = [
                    d for d in dirnames
                    if not d.startswith(".") and d not in {"node_modules", "__pycache__", "venv", ".venv"}
                ]
        except PermissionError as e:
            logger.warning(f"Permission denied scanning directory: {e}")
        except OSError as e:
            logger.warning(f"Error scanning directory: {e}")

        # Sort by depth to ensure proper precedence (root first, then deeper)
        gitignore_files.sort(key=lambda x: x[0])

        # Load each .gitignore file in order
        for depth, gitignore_path in gitignore_files:
            patterns_loaded = self.load_gitignore(gitignore_path)
            total_patterns += patterns_loaded
            if patterns_loaded > 0:
                logger.debug(
                    f"Loaded {patterns_loaded} patterns from {gitignore_path} (depth={depth})"
                )

        logger.debug(f"Total patterns loaded from hierarchy: {total_patterns}")
        return total_patterns

    def load_gitignore_for_path(self, file_path: Path) -> int:
        """
        Load .gitignore files that apply to a specific file path.

        This is useful for lazy loading - only load gitignore files
        along the path to a specific file rather than the entire tree.

        Args:
            file_path: Path to the file being checked

        Returns:
            Number of new patterns loaded
        """
        file_path = Path(file_path)
        if file_path.is_absolute():
            try:
                file_path = file_path.relative_to(self._root_path)
            except ValueError:
                return 0

        total_patterns = 0
        current_dir = self._root_path

        # Load root .gitignore first
        root_gitignore = current_dir / ".gitignore"
        if root_gitignore.exists() and not self._has_loaded_gitignore(root_gitignore):
            total_patterns += self.load_gitignore(root_gitignore)

        # Walk down the path and load any .gitignore files
        for part in file_path.parts[:-1]:  # Exclude the filename itself
            current_dir = current_dir / part
            gitignore_path = current_dir / ".gitignore"
            if gitignore_path.exists() and not self._has_loaded_gitignore(gitignore_path):
                total_patterns += self.load_gitignore(gitignore_path)

        return total_patterns

    def _has_loaded_gitignore(self, gitignore_path: Path) -> bool:
        """Check if a .gitignore file has already been loaded."""
        gitignore_path = gitignore_path.resolve()
        return any(p.source_path == gitignore_path for p in self._patterns)

    def _rebuild_pathspec(self) -> None:
        """Rebuild the pathspec matcher from current patterns."""
        if not self._pathspec_dirty:
            return

        if not self._patterns:
            self._pathspec = None
        else:
            # Build pattern strings for pathspec
            # Note: pathspec handles negation, directory-only, and globs
            pattern_strings = [p.raw for p in self._patterns]
            self._pathspec = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, pattern_strings
            )

        self._pathspec_dirty = False

    @property
    def pattern_count(self) -> int:
        """Return the number of loaded patterns."""
        return len(self._patterns)

    @property
    def patterns(self) -> list[GitignorePattern]:
        """Return a copy of the loaded patterns."""
        return list(self._patterns)


    def matches(self, path: Path, is_dir: bool = False) -> bool:
        """
        Check if a path matches any gitignore pattern.

        Patterns are applied in order, with negation patterns able to
        re-include previously excluded files. The last matching pattern wins.

        Args:
            path: Path to check (can be absolute or relative to root)
            is_dir: True if the path is a directory

        Returns:
            True if the path should be ignored
        """
        # Normalize path to be relative to root
        try:
            if path.is_absolute():
                rel_path = path.relative_to(self._root_path)
            else:
                rel_path = path
        except ValueError:
            # Path is not under root - don't ignore
            return False

        # Normalize path separators to forward slashes for matching
        rel_path_str = str(rel_path).replace("\\", "/")

        # Apply case sensitivity
        if not self._case_sensitive:
            rel_path_str_match = rel_path_str.lower()
        else:
            rel_path_str_match = rel_path_str

        # Track whether the path is currently ignored
        ignored = False

        for pattern in self._patterns:
            if self._pattern_matches(pattern, rel_path_str, rel_path_str_match, is_dir):
                # Pattern matches - update ignored status based on negation
                ignored = not pattern.negation

        return ignored

    def _pattern_matches(
        self,
        pattern: GitignorePattern,
        rel_path_str: str,
        rel_path_str_match: str,
        is_dir: bool,
    ) -> bool:
        """
        Check if a single pattern matches the given path.

        For nested gitignore files, patterns are scoped relative to the directory
        containing the .gitignore file. A pattern in /subdir/.gitignore only
        applies to files under /subdir/.

        Args:
            pattern: The GitignorePattern to check
            rel_path_str: Original relative path string (for display)
            rel_path_str_match: Path string for matching (may be lowercased)
            is_dir: True if the path is a directory

        Returns:
            True if the pattern matches
        """
        # Directory-only patterns only match directories
        if pattern.directory_only and not is_dir:
            return False

        # Get the pattern string for matching
        pattern_str = pattern.pattern
        if not self._case_sensitive:
            pattern_str = pattern_str.lower()

        # For nested gitignore files, scope the pattern relative to its source directory
        # Patterns from /subdir/.gitignore only apply to files under /subdir/
        scoped_path_str = rel_path_str_match
        if pattern.source_depth > 0:
            # Calculate the relative path from the gitignore's directory
            source_dir = self._get_pattern_source_dir(pattern)
            if source_dir:
                source_dir_str = source_dir.replace("\\", "/")
                if not self._case_sensitive:
                    source_dir_str = source_dir_str.lower()
                
                # Check if the path is under the gitignore's directory
                if not rel_path_str_match.startswith(source_dir_str + "/") and rel_path_str_match != source_dir_str:
                    # Path is not under this gitignore's scope
                    return False
                
                # Make path relative to the gitignore's directory
                if rel_path_str_match.startswith(source_dir_str + "/"):
                    scoped_path_str = rel_path_str_match[len(source_dir_str) + 1:]
                elif rel_path_str_match == source_dir_str:
                    scoped_path_str = ""

        # Handle anchored patterns (must match from root of scope)
        if pattern.anchored:
            # Anchored pattern - must match from the start of the scoped path
            return self._match_anchored(pattern_str, scoped_path_str)
        else:
            # Non-anchored pattern - can match anywhere in the path
            return self._match_unanchored(pattern_str, scoped_path_str)

    def _get_pattern_source_dir(self, pattern: GitignorePattern) -> str | None:
        """
        Get the directory containing the pattern's source .gitignore file,
        relative to the root path.

        Args:
            pattern: The GitignorePattern

        Returns:
            Relative path string to the source directory, or None if at root
        """
        if pattern.source_depth <= 0:
            return None
        
        try:
            source_dir = pattern.source_path.parent
            rel_source_dir = source_dir.relative_to(self._root_path)
            return str(rel_source_dir).replace("\\", "/")
        except ValueError:
            return None

    def _match_anchored(self, pattern_str: str, rel_path_str: str) -> bool:
        """
        Match an anchored pattern (starts with /).

        Anchored patterns only match at the root level - they cannot match
        files in subdirectories unless the pattern itself contains path separators.

        Args:
            pattern_str: Pattern string (without leading /)
            rel_path_str: Relative path string

        Returns:
            True if pattern matches
        """
        try:
            # For anchored patterns without path separators, only match at root level
            if "/" not in pattern_str and "**" not in pattern_str:
                # Simple anchored pattern - only match if path has no directory component
                # or if the first component matches
                path_parts = rel_path_str.split("/")
                if len(path_parts) > 1:
                    # File is in a subdirectory - anchored pattern without / shouldn't match
                    return False
                # Match against the single component
                spec = pathspec.PathSpec.from_lines(
                    pathspec.patterns.GitWildMatchPattern, [pattern_str]
                )
                return spec.match_file(rel_path_str)
            else:
                # Pattern contains path separators or **, use full path matching
                spec = pathspec.PathSpec.from_lines(
                    pathspec.patterns.GitWildMatchPattern, [pattern_str]
                )
                return spec.match_file(rel_path_str)
        except Exception:
            return False

    def _match_unanchored(self, pattern_str: str, rel_path_str: str) -> bool:
        """
        Match an unanchored pattern.

        Unanchored patterns can match:
        - The full path
        - Any path component (basename)
        - Using ** for directory wildcards

        Args:
            pattern_str: Pattern string
            rel_path_str: Relative path string

        Returns:
            True if pattern matches
        """
        try:
            # If pattern contains a slash, it's path-relative
            if "/" in pattern_str:
                spec = pathspec.PathSpec.from_lines(
                    pathspec.patterns.GitWildMatchPattern, [pattern_str]
                )
                return spec.match_file(rel_path_str)
            else:
                # Pattern without slash matches basename anywhere in path
                # Check each path component
                path_parts = rel_path_str.split("/")
                spec = pathspec.PathSpec.from_lines(
                    pathspec.patterns.GitWildMatchPattern, [pattern_str]
                )
                
                # Match against basename
                if spec.match_file(path_parts[-1]):
                    return True
                
                # Also check if any directory component matches
                for part in path_parts[:-1]:
                    if spec.match_file(part):
                        return True
                
                return False
        except Exception:
            return False

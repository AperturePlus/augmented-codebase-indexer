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
from dataclasses import dataclass
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

        # Cache for single-pattern pathspec objects used during matching
        self._pattern_spec_cache: dict[str, pathspec.PathSpec] = {}

    def load_gitignore(self, gitignore_path: Path) -> int:
        """
        Load patterns from a .gitignore file.

        Handles the following error conditions gracefully:
        - Invalid UTF-8 encoding: Logs warning, returns 0
        - Malformed patterns: Logs warning, skips pattern, continues with others
        - Permission errors: Logs warning, returns 0
        - Empty files: Returns 0 (no patterns to load)
        - File not found: Logs debug, returns 0

        Args:
            gitignore_path: Path to the .gitignore file

        Returns:
            Number of patterns loaded

        Raises:
            No exceptions - all errors are logged and the method returns gracefully
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
        skipped_patterns = 0

        # Read file content with graceful error handling
        try:
            content = gitignore_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            logger.warning(
                f"Invalid UTF-8 encoding in {gitignore_path}, skipping file: {e}"
            )
            return 0
        except PermissionError as e:
            logger.warning(
                f"Permission denied reading {gitignore_path}, skipping file: {e}"
            )
            return 0
        except OSError as e:
            logger.warning(f"Error reading {gitignore_path}, skipping file: {e}")
            return 0

        # Handle empty files gracefully
        if not content or not content.strip():
            logger.debug(f"Empty gitignore file: {gitignore_path}")
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

            # Validate and parse pattern
            try:
                # Check for malformed patterns before parsing
                validation_error = self._validate_pattern(line)
                if validation_error:
                    logger.warning(
                        f"Malformed pattern '{line}' in {gitignore_path}: {validation_error}"
                    )
                    skipped_patterns += 1
                    continue

                pattern = GitignorePattern.parse(line, gitignore_path, source_depth)
                self._patterns.append(pattern)
                patterns_loaded += 1
            except Exception as e:
                logger.warning(
                    f"Failed to parse pattern '{line}' in {gitignore_path}: {e}"
                )
                skipped_patterns += 1
                continue

        if patterns_loaded > 0:
            self._pathspec_dirty = True
            self._pattern_spec_cache.clear()

        # Log summary at debug level (Requirement 6.1)
        logger.debug(
            f"Loaded {patterns_loaded} patterns from {gitignore_path}"
            + (f" ({skipped_patterns} skipped)" if skipped_patterns > 0 else "")
        )

        return patterns_loaded

    def _validate_pattern(self, pattern: str) -> str | None:
        """
        Validate a gitignore pattern for common malformations.

        Args:
            pattern: Raw pattern string to validate

        Returns:
            Error message if pattern is malformed, None if valid
        """
        # Check for patterns that are just special characters
        stripped = pattern.lstrip("!").rstrip("/")
        if not stripped:
            return "Pattern is empty after removing special characters"

        # Check for unbalanced brackets in glob patterns
        if "[" in pattern or "]" in pattern:
            open_count = pattern.count("[")
            close_count = pattern.count("]")
            if open_count != close_count:
                return f"Unbalanced brackets: {open_count} '[' vs {close_count} ']'"

        # Check for invalid escape sequences (backslash at end)
        if pattern.endswith("\\") and not pattern.endswith("\\\\"):
            return "Pattern ends with incomplete escape sequence"

        return None

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
            self._pattern_spec_cache.clear()

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

    def get_loaded_gitignore_summary(self) -> dict[Path, int]:
        """
        Get a summary of loaded .gitignore files and their pattern counts.

        This is useful for verbose mode reporting (Requirement 6.3).

        Returns:
            Dictionary mapping .gitignore file paths to number of patterns loaded
        """
        summary: dict[Path, int] = {}
        for pattern in self._patterns:
            source = pattern.source_path
            if source not in summary:
                summary[source] = 0
            summary[source] += 1
        return summary

    def log_verbose_summary(self) -> None:
        """
        Log a verbose summary of loaded .gitignore files.

        Logs which .gitignore files were loaded and how many patterns each contained.
        This is intended for verbose/debug mode (Requirement 6.3).
        """
        summary = self.get_loaded_gitignore_summary()
        if not summary:
            logger.info("No .gitignore files loaded")
            return

        logger.info(f"Loaded {len(summary)} .gitignore file(s):")
        for gitignore_path, count in sorted(summary.items(), key=lambda x: str(x[0])):
            logger.info(f"  {gitignore_path}: {count} pattern(s)")


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

        # Directory patterns (e.g., out/) should also match directory paths directly
        match_candidates = [rel_path_str_match]
        if is_dir and rel_path_str_match:
            match_candidates.append(f"{rel_path_str_match}/")

        # Track whether the path is currently ignored and which pattern matched
        ignored = False
        matching_pattern: GitignorePattern | None = None

        for pattern in self._patterns:
            if self._pattern_matches(pattern, match_candidates):
                # Pattern matches - update ignored status based on negation
                ignored = not pattern.negation
                matching_pattern = pattern

        # Log file exclusions with matching pattern (Requirement 6.2)
        if ignored and matching_pattern is not None:
            logger.debug(
                f"Excluding '{rel_path_str}' - matched pattern '{matching_pattern.raw}' "
                f"from {matching_pattern.source_path}"
            )

        return ignored

    def _pattern_matches(
        self,
        pattern: GitignorePattern,
        match_candidates: list[str],
    ) -> bool:
        """
        Check if a single pattern matches the given path.

        For nested gitignore files, patterns are scoped relative to the directory
        containing the .gitignore file. A pattern in /subdir/.gitignore only
        applies to files under /subdir/.

        Args:
            pattern: The GitignorePattern to check
            match_candidates: Normalized path candidates to evaluate

        Returns:
            True if the pattern matches
        """
        try:
            effective_pattern = self._build_effective_pattern(pattern)
            spec = self._get_or_build_pattern_spec(effective_pattern)

            for candidate in match_candidates:
                if not self._anchored_candidate_depth_ok(pattern, candidate):
                    continue
                if spec.match_file(candidate):
                    return True
            return False
        except Exception:
            return False

    def _anchored_candidate_depth_ok(
        self,
        pattern: GitignorePattern,
        candidate: str,
    ) -> bool:
        """
        Enforce scope-root depth for anchored non-directory patterns.

        PathSpec treats `/name` as matching a root directory and its descendants.
        For this codebase's semantics, anchored patterns should match only at
        their scope root depth (e.g., `/a` matches `a` but not `a/b/c/a`).
        """
        if not pattern.anchored:
            return True

        # Keep standard recursive behavior for anchored directory-only patterns.
        if pattern.directory_only:
            return True

        # Keep explicit recursive intent if pattern uses globstar.
        if "**" in pattern.pattern:
            return True

        source_dir = self._get_pattern_source_dir(pattern)
        if source_dir and not self._case_sensitive:
            source_dir = source_dir.lower()

        candidate_clean = candidate.rstrip("/")
        scoped_path = candidate_clean

        if source_dir:
            source_prefix = f"{source_dir}/"
            if not candidate_clean.startswith(source_prefix):
                return False
            scoped_path = candidate_clean[len(source_prefix):]

        # Anchored patterns only match paths at the scope root depth.
        pattern_depth = len([part for part in pattern.pattern.split("/") if part])
        candidate_depth = len([part for part in scoped_path.split("/") if part])
        return candidate_depth == pattern_depth

    def _build_effective_pattern(self, pattern: GitignorePattern) -> str:
        """Convert a parsed pattern to a root-scoped GitWildMatch pattern string."""
        base_pattern = pattern.pattern

        if not self._case_sensitive:
            base_pattern = base_pattern.lower()

        if pattern.directory_only:
            base_pattern = f"{base_pattern}/"

        source_dir = self._get_pattern_source_dir(pattern)
        if source_dir and not self._case_sensitive:
            source_dir = source_dir.lower()

        # Keep root-anchored patterns anchored with a leading slash.
        if not source_dir:
            if pattern.anchored:
                return f"/{base_pattern}"
            return base_pattern

        # Nested .gitignore patterns are scoped to their source directory.
        if pattern.anchored:
            return f"{source_dir}/{base_pattern}"

        # Unanchored patterns containing slashes are relative to source directory.
        if "/" in pattern.pattern:
            return f"{source_dir}/{base_pattern}"

        # Bare names in nested .gitignore files match at any depth within that subtree.
        return f"{source_dir}/**/{base_pattern}"

    def _get_or_build_pattern_spec(self, pattern_str: str) -> pathspec.PathSpec:
        """Get a cached single-pattern PathSpec matcher."""
        if pattern_str not in self._pattern_spec_cache:
            self._pattern_spec_cache[pattern_str] = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, [pattern_str]
            )
        return self._pattern_spec_cache[pattern_str]

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

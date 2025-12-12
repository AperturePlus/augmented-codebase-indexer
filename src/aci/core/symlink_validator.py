"""
SymlinkValidator module for Project ACI.

Provides symlink validation to prevent path traversal attacks during file scanning.
Validates that symlinks:
- Resolve to paths within the root directory boundary
- Do not point to sensitive directories
- Do not create cycles back to ancestor directories
"""

import fnmatch
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SymlinkValidationResult:
    """
    Result of symlink validation.

    Attributes:
        safe: True if symlink is safe to follow
        reason: Reason if unsafe (for logging), None if safe
        target_path: Resolved target path if safe, None if unsafe
    """

    safe: bool
    reason: str | None
    target_path: Path | None


class SymlinkValidator:
    """
    Validates symbolic links to prevent path traversal attacks.

    Checks:
    - Symlink target is within the root directory
    - Symlink target is not a sensitive directory
    - Symlink does not create a cycle
    """

    def __init__(self, root_path: Path, sensitive_patterns: frozenset[str] | None = None):
        """
        Initialize the SymlinkValidator.

        Args:
            root_path: Root directory boundary - symlinks must resolve within this
            sensitive_patterns: Patterns for sensitive directories/files to block.
                              If None, uses default sensitive patterns.
        """
        self._root_path = Path(root_path).resolve()
        self._sensitive_patterns = sensitive_patterns or self._default_sensitive_patterns()

    @staticmethod
    def _default_sensitive_patterns() -> frozenset[str]:
        """Return default sensitive patterns that should never be followed."""
        return frozenset([
            # SSH and GPG directories
            ".ssh",
            ".gnupg",
            # SSH key files
            "id_rsa",
            "id_rsa.pub",
            "id_ed25519",
            "id_ed25519.pub",
            "id_ecdsa",
            "id_ecdsa.pub",
            "id_dsa",
            "id_dsa.pub",
            # Certificates and keys (glob patterns)
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "*.crt",
            "*.keystore",
            # Environment files
            ".env",
            ".env.*",
            ".env.local",
            ".env.production",
            ".env.development",
            # Other sensitive files
            ".netrc",
            ".npmrc",
            ".pypirc",
        ])

    def is_safe_symlink(
        self, symlink_path: Path, visited: set[Path] | None = None
    ) -> SymlinkValidationResult:
        """
        Check if a symlink is safe to follow.

        Performs the following checks in order:
        1. Resolves the symlink to its real path
        2. Checks if target is within root directory boundary
        3. Checks if target matches sensitive patterns
        4. Checks for cycles (if visited set provided)

        Args:
            symlink_path: Path to the symlink to validate
            visited: Set of already-visited real paths (for cycle detection).
                    If None, cycle detection is skipped.

        Returns:
            SymlinkValidationResult with safe status, reason if unsafe, and target path
        """
        symlink_path = Path(symlink_path)

        # Check if it's actually a symlink
        if not symlink_path.is_symlink():
            return SymlinkValidationResult(
                safe=False,
                reason="Path is not a symlink",
                target_path=None,
            )

        # Try to resolve the symlink
        try:
            target_path = symlink_path.resolve()
        except OSError as e:
            reason = f"Failed to resolve symlink: {e}"
            logger.debug(f"Skipping symlink {symlink_path}: {reason}")
            return SymlinkValidationResult(
                safe=False,
                reason=reason,
                target_path=None,
            )

        # Check if target exists (broken symlink)
        if not target_path.exists():
            reason = "Broken symlink - target does not exist"
            logger.debug(
                f"Skipping symlink {symlink_path} -> {target_path}: {reason}"
            )
            return SymlinkValidationResult(
                safe=False,
                reason=reason,
                target_path=None,
            )

        # Check if target is within root directory boundary
        if not self._is_within_root(target_path):
            reason = f"Symlink target is outside root directory: {target_path}"
            logger.warning(
                f"Skipping symlink {symlink_path} -> {target_path}: {reason}"
            )
            return SymlinkValidationResult(
                safe=False,
                reason=reason,
                target_path=None,
            )

        # Check if target matches sensitive patterns
        if self._is_sensitive_target(target_path):
            reason = f"Symlink target matches sensitive pattern: {target_path.name}"
            logger.debug(
                f"Skipping symlink {symlink_path} -> {target_path}: {reason}"
            )
            return SymlinkValidationResult(
                safe=False,
                reason=reason,
                target_path=None,
            )

        # Check for cycles if visited set is provided
        if visited is not None and self._creates_cycle(target_path, visited):
            reason = "Symlink creates a cycle back to an ancestor directory"
            logger.debug(
                f"Skipping symlink {symlink_path} -> {target_path}: {reason}"
            )
            return SymlinkValidationResult(
                safe=False,
                reason=reason,
                target_path=None,
            )

        return SymlinkValidationResult(
            safe=True,
            reason=None,
            target_path=target_path,
        )

    def _is_within_root(self, target_path: Path) -> bool:
        """
        Check if a resolved path is within the root directory boundary.

        Args:
            target_path: Resolved (real) path to check

        Returns:
            True if target is within root directory
        """
        try:
            # Ensure both paths are resolved
            target_resolved = target_path.resolve()
            root_resolved = self._root_path.resolve()

            # Check if target is the root or a descendant of root
            target_resolved.relative_to(root_resolved)
            return True
        except ValueError:
            # relative_to raises ValueError if target is not under root
            return False

    def _is_sensitive_target(self, target_path: Path) -> bool:
        """
        Check if a path matches any sensitive pattern.

        Checks both the target path's name and all ancestor directory names
        to catch symlinks pointing into sensitive directories.

        Args:
            target_path: Path to check

        Returns:
            True if path matches a sensitive pattern
        """
        # Check the target name itself
        if self._matches_sensitive_pattern(target_path.name):
            return True

        # Check all ancestor directories
        for parent in target_path.parents:
            if self._matches_sensitive_pattern(parent.name):
                return True

        return False

    def _matches_sensitive_pattern(self, name: str) -> bool:
        """
        Check if a name matches any sensitive pattern.

        Args:
            name: File or directory name to check

        Returns:
            True if name matches a sensitive pattern
        """
        for pattern in self._sensitive_patterns:
            # Check exact match first
            if name == pattern:
                return True
            # Check glob pattern match
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _creates_cycle(self, target_path: Path, visited: set[Path]) -> bool:
        """
        Check if following a symlink would create a cycle.

        A cycle occurs when the symlink target is an ancestor of the current
        traversal path, which would cause infinite recursion.

        Args:
            target_path: Resolved target path of the symlink
            visited: Set of already-visited real paths in current traversal

        Returns:
            True if following this symlink would create a cycle
        """
        target_resolved = target_path.resolve()

        # Check if target is already in visited set (direct cycle)
        if target_resolved in visited:
            return True

        # Check if target is an ancestor of any visited path
        # This catches cases where symlink points to a parent directory
        for visited_path in visited:
            try:
                visited_path.relative_to(target_resolved)
                # If we get here, visited_path is under target_resolved
                # This means target is an ancestor - following would create cycle
                return True
            except ValueError:
                # Not an ancestor, continue checking
                continue

        return False

    @property
    def root_path(self) -> Path:
        """Return the root path boundary."""
        return self._root_path

    @property
    def sensitive_patterns(self) -> frozenset[str]:
        """Return the sensitive patterns."""
        return self._sensitive_patterns
